"""Structured DAG dispatch with isolated child agents and cooperative deadlines."""
import json
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from threading import Event, Lock, local
from typing import Callable, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from .agent import Agent
from .context import ContextManager
from .llm import ChatClient
from .tools import ToolRegistry

Role = Literal["orchestrator", "leaf"]
Status = Literal["pending", "running", "success", "failed", "skipped", "timed_out"]
_worker = local()


class DispatchError(Exception):
    """Role, depth, or worker dispatch rule rejected a plan."""


class PlanError(ValueError):
    """Invalid DAG plan."""


class AgentIdentity(BaseModel):
    agent_id: str
    parent_agent_id: str | None
    role: Role
    depth: int = Field(ge=0)


class TaskNode(BaseModel):
    task_id: str = Field(min_length=1)
    description: str
    depends_on: list[str] = Field(default_factory=list)
    timeout_s: float | None = Field(default=None, gt=0)
    role: Role = "leaf"
    subtasks: list["TaskNode"] = Field(default_factory=list)
    status: Status = "pending"
    result: str | None = None
    error: str | None = None


class SubAgentResult(BaseModel):
    task_id: str
    agent_id: str | None
    status: Literal["success", "failed", "skipped", "timed_out"]
    output: str | None = None
    error: str | None = None
    duration_ms: float = Field(ge=0)
    worker_running: bool = False


class AgentHandle:
    def __init__(self, identity: AgentIdentity, agent: Agent, factory: "AgentFactory"):
        self.identity, self.agent, self.factory = identity, agent, factory

    def dispatch(self, tasks: list[TaskNode], max_workers: int = 4) -> list[SubAgentResult]:
        if self.identity.role != "orchestrator":
            raise DispatchError("Leaf agents cannot dispatch sub-agents")
        if self.identity.depth >= self.factory.max_depth:
            raise DispatchError(f"Maximum agent depth {self.factory.max_depth} exceeded")
        scheduler = DAGScheduler(self.factory, max_workers)
        if getattr(_worker, "active", False):
            return scheduler.run_inline(self, tasks)
        return scheduler.run(self, tasks)

    def reason_with_results(self, prompt: str, results: list[SubAgentResult]) -> str:
        if self.identity.role != "orchestrator":
            raise DispatchError("Leaf agents cannot synthesize a plan")
        payload = json.dumps([item.model_dump() for item in results], ensure_ascii=False)
        return self.agent.run(f"{prompt}\n\nSub-agent results (JSON, in task order):\n{payload}")

    def dispatch_and_summarize(self, tasks: list[TaskNode], prompt: str,
                               max_workers: int = 4) -> tuple[list[SubAgentResult], str]:
        results = self.dispatch(tasks, max_workers)
        return results, self.reason_with_results(prompt, results)


class AgentFactory:
    def __init__(self, llm_factory: Callable[[], ChatClient], tools: ToolRegistry,
                 max_depth: int = 1, history_limit: int = 40, max_rounds: int = 8,
                 synchronize_tools: bool = True):
        if max_depth < 0 or history_limit < 1 or max_rounds < 1:
            raise ValueError("Invalid factory limits")
        self.llm_factory, self.tools = llm_factory, tools
        self.max_depth, self.history_limit, self.max_rounds = max_depth, history_limit, max_rounds
        self.synchronize_tools = synchronize_tools
        self._identity_lock = Lock()
        self._identities: dict[str, AgentIdentity] = {}

    @property
    def identities(self) -> dict[str, AgentIdentity]:
        with self._identity_lock:
            return {key: value.model_copy(deep=True) for key, value in self._identities.items()}

    def create(self, role: Role = "orchestrator", parent: AgentHandle | None = None) -> AgentHandle:
        depth = 0 if parent is None else parent.identity.depth + 1
        if depth > self.max_depth:
            raise DispatchError(f"Maximum agent depth {self.max_depth} exceeded")
        identity = AgentIdentity(agent_id=uuid4().hex,
                                 parent_agent_id=parent.identity.agent_id if parent else None,
                                 role=role, depth=depth)
        agent = Agent(self.llm_factory(), self.tools.clone(self.synchronize_tools),
                      ContextManager(self.history_limit), self.max_rounds)
        with self._identity_lock:
            self._identities[identity.agent_id] = identity.model_copy(deep=True)
        return AgentHandle(identity, agent, self)


class DAGScheduler:
    def __init__(self, factory: AgentFactory, max_workers: int = 4):
        if max_workers < 1:
            raise ValueError("max_workers must be positive")
        self.factory, self.max_workers = factory, max_workers

    @staticmethod
    def validate(tasks: list[TaskNode]) -> None:
        ids = [task.task_id for task in tasks]
        if len(ids) != len(set(ids)):
            raise PlanError("Duplicate task_id")
        by_id = {task.task_id: task for task in tasks}
        for task in tasks:
            if task.status != "pending":
                raise PlanError(f"Task {task.task_id} must start pending")
            if task.subtasks and task.role != "orchestrator":
                raise PlanError(f"Leaf task {task.task_id} cannot contain subtasks")
            if task.subtasks:
                DAGScheduler.validate(task.subtasks)
            for dependency in task.depends_on:
                if dependency not in by_id:
                    raise PlanError(f"Missing dependency {dependency} for {task.task_id}")
        colors: dict[str, int] = {}

        def visit(task_id: str) -> None:
            if colors.get(task_id) == 1:
                raise PlanError("DAG contains a cycle")
            if colors.get(task_id) == 2:
                return
            colors[task_id] = 1
            for dependency in by_id[task_id].depends_on:
                visit(dependency)
            colors[task_id] = 2

        for task_id in ids:
            visit(task_id)

    def run_inline(self, parent: AgentHandle, tasks: list[TaskNode]) -> list[SubAgentResult]:
        """Worker-side nested plan: same worker, no nested thread pool or deadlock."""
        self.validate(tasks)
        nodes = {task.task_id: task.model_copy(deep=True) for task in tasks}
        results: dict[str, SubAgentResult] = {}
        while len(results) < len(nodes):
            progressed = False
            for node in nodes.values():
                if node.task_id in results or any(dep not in results for dep in node.depends_on):
                    continue
                progressed = True
                bad = [dep for dep in node.depends_on if results[dep].status != "success"]
                if bad:
                    node.status, node.error = "skipped", f"Failed dependency: {', '.join(bad)}"
                    results[node.task_id] = SubAgentResult(task_id=node.task_id, agent_id=None,
                                                           status="skipped", error=node.error, duration_ms=0)
                    continue
                handle = self.factory.create(node.role, parent)
                start = time.monotonic()
                node.status = "running"
                try:
                    output = self._run_child(handle, node, [results[dep] for dep in node.depends_on],
                                             lambda: node.timeout_s is not None and time.monotonic() - start >= node.timeout_s)
                    if node.timeout_s is not None and time.monotonic() - start >= node.timeout_s:
                        node.status, node.error = "timed_out", f"Task budget {node.timeout_s}s exceeded"
                    else:
                        node.status, node.result = "success", output
                except Exception as exc:
                    if node.timeout_s is not None and time.monotonic() - start >= node.timeout_s:
                        node.status, node.error = "timed_out", f"Task budget {node.timeout_s}s exceeded"
                    else:
                        node.status, node.error = "failed", f"{type(exc).__name__}: {exc}"
                results[node.task_id] = SubAgentResult(task_id=node.task_id, agent_id=handle.identity.agent_id,
                    status=node.status, output=node.result, error=node.error,
                    duration_ms=(time.monotonic() - start) * 1000)
            if not progressed:
                raise RuntimeError("Nested DAG scheduler made no progress")
        return [results[task.task_id] for task in tasks]

    @staticmethod
    def _run_child(handle: AgentHandle, node: TaskNode, dependencies: list[SubAgentResult],
                   cancelled: Callable[[], bool]) -> str:
        if node.subtasks:
            nested = handle.dispatch(node.subtasks)
            return handle.reason_with_results(node.description, nested)
        prompt = node.description
        if dependencies:
            prompt += "\n\nCompleted dependencies (JSON):\n" + json.dumps(
                [item.model_dump() for item in dependencies], ensure_ascii=False)
        return handle.agent.run(prompt, cancelled=cancelled)

    def run(self, parent: AgentHandle, tasks: list[TaskNode]) -> list[SubAgentResult]:
        self.validate(tasks)
        if parent.identity.role != "orchestrator":
            raise DispatchError("Leaf agents cannot dispatch sub-agents")
        if parent.identity.depth >= self.factory.max_depth:
            raise DispatchError(f"Maximum agent depth {self.factory.max_depth} exceeded")
        nodes = {task.task_id: task.model_copy(deep=True) for task in tasks}
        results: dict[str, SubAgentResult] = {}
        active: dict[Future[str], tuple[TaskNode, AgentHandle, float, Event]] = {}
        result_lock = Lock()
        executor = ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="xcode-agent")

        def execute(handle: AgentHandle, node: TaskNode, stop: Event) -> str:
            _worker.active = True
            try:
                dependencies = [results[task_id] for task_id in node.depends_on]
                return self._run_child(handle, node, dependencies, stop.is_set)
            finally:
                _worker.active = False

        try:
            while len(results) < len(nodes):
                for node in nodes.values():
                    if node.status != "pending":
                        continue
                    bad = [dep for dep in node.depends_on if dep in results and results[dep].status != "success"]
                    if bad:
                        node.status, node.error = "skipped", f"Failed dependency: {', '.join(bad)}"
                        results[node.task_id] = SubAgentResult(task_id=node.task_id, agent_id=None,
                                                               status="skipped", error=node.error, duration_ms=0)
                for node in nodes.values():
                    if len(active) >= self.max_workers:
                        break
                    if node.status != "pending" or any(dep not in results for dep in node.depends_on):
                        continue
                    handle = self.factory.create(node.role, parent)
                    stop = Event()
                    start = time.monotonic()
                    node.status = "running"
                    future = executor.submit(execute, handle, node, stop)
                    active[future] = (node, handle, start, stop)
                if not active:
                    if len(results) != len(nodes):
                        raise RuntimeError("DAG scheduler made no progress")
                    break
                now = time.monotonic()
                remaining = [start + node.timeout_s - now for node, _, start, _ in active.values()
                             if node.timeout_s is not None]
                wait_timeout = max(0.0, min(remaining)) if remaining else None
                done, _ = wait(active, timeout=wait_timeout, return_when=FIRST_COMPLETED)
                now = time.monotonic()
                for future in list(active):
                    node, handle, start, stop = active[future]
                    expired = node.timeout_s is not None and now - start >= node.timeout_s
                    if future not in done and not future.done() and not expired:
                        continue
                    del active[future]
                    duration = (now - start) * 1000
                    if expired:
                        stop.set()
                        future.cancel()
                        node.status, node.error = "timed_out", f"Task budget {node.timeout_s}s exceeded; worker may still run"
                        item = SubAgentResult(task_id=node.task_id, agent_id=handle.identity.agent_id,
                                              status="timed_out", error=node.error, duration_ms=duration,
                                              worker_running=not future.done())
                        if item.worker_running:
                            def completed(_future: Future[str], _item: SubAgentResult = item) -> None:
                                with result_lock:
                                    _item.worker_running = False
                            future.add_done_callback(completed)
                    else:
                        try:
                            output = future.result()
                            node.status, node.result = "success", output
                            item = SubAgentResult(task_id=node.task_id, agent_id=handle.identity.agent_id,
                                                  status="success", output=output, duration_ms=duration)
                        except Exception as exc:
                            node.status, node.error = "failed", f"{type(exc).__name__}: {exc}"
                            item = SubAgentResult(task_id=node.task_id, agent_id=handle.identity.agent_id,
                                                  status="failed", error=node.error, duration_ms=duration)
                    results[node.task_id] = item
            return [results[task.task_id] for task in tasks]
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
