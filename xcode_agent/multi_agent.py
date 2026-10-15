"""Structured DAG dispatch with isolated child agents and cooperative deadlines."""
import json
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from threading import Event, Lock, local
from typing import Callable, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from .agent import Agent, AgentCancelled
from .context import ContextManager
from .llm import ChatClient
from .tools import ToolRegistry
from .tracing import TraceRecorder
from .memory import MemoryStore
from .skills import SkillRegistry
from .errors import AgentTimeoutError, DependencyFailedError

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
    run_id: str | None = None
    error_code: str | None = Field(default=None, exclude=True)


class AgentHandle:
    def __init__(self, identity: AgentIdentity, agent: Agent, factory: "AgentFactory"):
        self.identity, self.agent, self.factory = identity, agent, factory
        self.last_run_id: str | None = None

    def dispatch(self, tasks: list[TaskNode], max_workers: int = 4,
                 *, run_id: str | None = None) -> list[SubAgentResult]:
        if self.identity.role != "orchestrator":
            raise DispatchError("Leaf agents cannot dispatch sub-agents")
        if self.identity.depth >= self.factory.max_depth:
            raise DispatchError(f"Maximum agent depth {self.factory.max_depth} exceeded")
        scheduler = DAGScheduler(self.factory, max_workers)
        run_id = run_id or getattr(_worker, "run_id", None) or self.factory.trace.new_run_id()
        self.last_run_id = run_id
        if getattr(_worker, "active", False):
            return scheduler.run_inline(self, tasks, run_id)
        return scheduler.run(self, tasks, run_id)

    def reason_with_results(self, prompt: str, results: list[SubAgentResult]) -> str:
        if self.identity.role != "orchestrator":
            raise DispatchError("Leaf agents cannot synthesize a plan")
        payload = json.dumps([item.model_dump() for item in results], ensure_ascii=False)
        return self.agent.run(f"{prompt}\n\nSub-agent results (JSON, in task order):\n{payload}",
                              run_id=getattr(_worker, "run_id", None) or self.last_run_id)

    def dispatch_and_summarize(self, tasks: list[TaskNode], prompt: str,
                               max_workers: int = 4) -> tuple[list[SubAgentResult], str]:
        results = self.dispatch(tasks, max_workers)
        return results, self.reason_with_results(prompt, results)


class AgentFactory:
    def __init__(self, llm_factory: Callable[[], ChatClient], tools: ToolRegistry,
                 max_depth: int = 1, history_limit: int = 40, max_rounds: int = 8,
                 synchronize_tools: bool = True, trace: TraceRecorder | None = None,
                 memory: MemoryStore | None = None, shared_memory_namespace: str | None = None,
                 memory_top_k: int = 5, memory_char_budget: int = 2_000,
                 skills: SkillRegistry | None = None, max_skills: int = 8,
                 skill_char_budget: int = 8_000):
        if max_depth < 0 or history_limit < 1 or max_rounds < 1:
            raise ValueError("Invalid factory limits")
        if memory is None and shared_memory_namespace is not None:
            raise ValueError("shared_memory_namespace requires a memory store")
        if (shared_memory_namespace is not None and
                (not isinstance(shared_memory_namespace, str) or not shared_memory_namespace.strip() or
                 len(shared_memory_namespace) > 128)):
            raise ValueError("Invalid shared_memory_namespace")
        if (not isinstance(memory_top_k, int) or isinstance(memory_top_k, bool) or
                not 1 <= memory_top_k <= MemoryStore.MAX_TOP_K or
                not isinstance(memory_char_budget, int) or isinstance(memory_char_budget, bool) or
                memory_char_budget < 1):
            raise ValueError("Invalid memory retrieval limits")
        if (not isinstance(max_skills, int) or isinstance(max_skills, bool) or max_skills < 1 or
                not isinstance(skill_char_budget, int) or isinstance(skill_char_budget, bool) or
                skill_char_budget < 1):
            raise ValueError("Invalid Skill limits")
        if skills is not None and tools.get("load_skill") is not None:
            raise ValueError("Tool name load_skill is reserved when Skills are enabled")
        self.llm_factory, self.tools = llm_factory, tools
        self.max_depth, self.history_limit, self.max_rounds = max_depth, history_limit, max_rounds
        self.synchronize_tools = synchronize_tools
        self.trace = trace if trace is not None else TraceRecorder()
        self.memory = memory
        self.shared_memory_namespace = shared_memory_namespace
        self.memory_top_k, self.memory_char_budget = memory_top_k, memory_char_budget
        self.skills, self.max_skills, self.skill_char_budget = skills, max_skills, skill_char_budget
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
                      ContextManager(self.history_limit), self.max_rounds,
                      trace=self.trace, agent_id=identity.agent_id,
                      memory=self.memory,
                      memory_namespace=self.shared_memory_namespace if self.memory is not None and
                      self.shared_memory_namespace is not None else None,
                      memory_top_k=self.memory_top_k, memory_char_budget=self.memory_char_budget,
                      skills=self.skills, max_skills=self.max_skills,
                      skill_char_budget=self.skill_char_budget)
        with self._identity_lock:
            self._identities[identity.agent_id] = identity.model_copy(deep=True)
        return AgentHandle(identity, agent, self)


class DAGScheduler:
    def __init__(self, factory: AgentFactory, max_workers: int = 4):
        if max_workers < 1:
            raise ValueError("max_workers must be positive")
        self.factory, self.max_workers = factory, max_workers

    def _trace_task(self, run_id: str, parent: AgentHandle, node: TaskNode,
                    event_type: str, *, agent_id: str | None = None,
                    duration_ms: float | None = None, success: bool | None = None,
                    error_type: str | None = None, worker_running: bool | None = None,
                    error_code: str | None = None) -> None:
        parent_task_id = getattr(_worker, "task_id", None)
        metadata = {"task_id": node.task_id, "parent_agent_id": parent.identity.agent_id,
                    "parent_task_id": None if parent_task_id == node.task_id else parent_task_id,
                    "role": node.role, "error_code": error_code}
        if worker_running is not None:
            metadata["worker_running"] = worker_running
        self.factory.trace.record(run_id, agent_id or parent.identity.agent_id, event_type,
                                  duration_ms=duration_ms, success=success,
                                  error_type=error_type, metadata=metadata)

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

    def run_inline(self, parent: AgentHandle, tasks: list[TaskNode],
                   run_id: str | None = None) -> list[SubAgentResult]:
        """Worker-side nested plan: same worker, no nested thread pool or deadlock."""
        self.validate(tasks)
        run_id = run_id or getattr(_worker, "run_id", None) or self.factory.trace.new_run_id()
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
                                                           status="skipped", error=node.error, duration_ms=0,
                                                           run_id=run_id,
                                                           error_code=DependencyFailedError.error_code)
                    self._trace_task(run_id, parent, node, "task_skipped", success=False,
                                     error_type="DependencyFailedError",
                                     error_code=DependencyFailedError.error_code)
                    continue
                handle = self.factory.create(node.role, parent)
                self._trace_task(run_id, parent, node, "child_created", agent_id=handle.identity.agent_id)
                start = time.monotonic()
                node.status = "running"
                self._trace_task(run_id, parent, node, "child_start", agent_id=handle.identity.agent_id)
                failure_code: str | None = None
                try:
                    output = self._run_child(handle, node, [results[dep] for dep in node.depends_on],
                                             lambda: node.timeout_s is not None and time.monotonic() - start >= node.timeout_s,
                                             run_id)
                    if node.timeout_s is not None and time.monotonic() - start >= node.timeout_s:
                        node.status, node.error = "timed_out", f"Task budget {node.timeout_s}s exceeded"
                    else:
                        node.status, node.result = "success", output
                except Exception as exc:
                    if isinstance(exc, AgentCancelled):
                        self._trace_task(run_id, parent, node, "task_cancelled", agent_id=handle.identity.agent_id,
                                         duration_ms=(time.monotonic() - start) * 1000,
                                         success=False, error_type="AgentCancelled",
                                         error_code=AgentCancelled.error_code)
                    if node.timeout_s is not None and time.monotonic() - start >= node.timeout_s:
                        node.status, node.error = "timed_out", f"Task budget {node.timeout_s}s exceeded"
                    else:
                        node.status, node.error = "failed", f"{type(exc).__name__}: {exc}"
                        failure_code = getattr(exc, "error_code", "child_task_error")
                if node.status == "timed_out":
                    failure_code = AgentTimeoutError.error_code
                results[node.task_id] = SubAgentResult(task_id=node.task_id, agent_id=handle.identity.agent_id,
                    status=node.status, output=node.result, error=node.error,
                    duration_ms=(time.monotonic() - start) * 1000, run_id=run_id,
                    error_code=failure_code)
                self._trace_task(run_id, parent, node, "task_" + node.status,
                                 agent_id=handle.identity.agent_id,
                                 duration_ms=results[node.task_id].duration_ms,
                                 success=node.status == "success",
                                 error_type=None if node.status == "success" else
                                 ("Timeout" if node.status == "timed_out" else "ChildTaskError"),
                                 error_code=failure_code)
            if not progressed:
                raise RuntimeError("Nested DAG scheduler made no progress")
        return [results[task.task_id] for task in tasks]

    @staticmethod
    def _run_child(handle: AgentHandle, node: TaskNode, dependencies: list[SubAgentResult],
                   cancelled: Callable[[], bool], run_id: str | None = None) -> str:
        if node.subtasks:
            nested = handle.dispatch(node.subtasks)
            return handle.reason_with_results(node.description, nested)
        prompt = node.description
        if dependencies:
            prompt += "\n\nCompleted dependencies (JSON):\n" + json.dumps(
                [item.model_dump() for item in dependencies], ensure_ascii=False)
        return handle.agent.run(prompt, cancelled=cancelled, run_id=run_id)

    def run(self, parent: AgentHandle, tasks: list[TaskNode],
            run_id: str | None = None) -> list[SubAgentResult]:
        self.validate(tasks)
        run_id = run_id or self.factory.trace.new_run_id()
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
            _worker.run_id = run_id
            _worker.task_id = node.task_id
            try:
                self._trace_task(run_id, parent, node, "child_start", agent_id=handle.identity.agent_id)
                dependencies = [results[task_id] for task_id in node.depends_on]
                return self._run_child(handle, node, dependencies, stop.is_set, run_id)
            except AgentCancelled:
                self._trace_task(run_id, parent, node, "task_cancelled", agent_id=handle.identity.agent_id,
                                 success=False, error_type="AgentCancelled",
                                 error_code=AgentCancelled.error_code)
                raise
            finally:
                _worker.active = False
                _worker.run_id = None
                _worker.task_id = None

        try:
            while len(results) < len(nodes):
                for node in nodes.values():
                    if node.status != "pending":
                        continue
                    bad = [dep for dep in node.depends_on if dep in results and results[dep].status != "success"]
                    if bad:
                        node.status, node.error = "skipped", f"Failed dependency: {', '.join(bad)}"
                        results[node.task_id] = SubAgentResult(task_id=node.task_id, agent_id=None,
                                                               status="skipped", error=node.error, duration_ms=0,
                                                               run_id=run_id,
                                                               error_code=DependencyFailedError.error_code)
                        self._trace_task(run_id, parent, node, "task_skipped", success=False,
                                         error_type="DependencyFailedError",
                                         error_code=DependencyFailedError.error_code)
                for node in nodes.values():
                    if len(active) >= self.max_workers:
                        break
                    if node.status != "pending" or any(dep not in results for dep in node.depends_on):
                        continue
                    handle = self.factory.create(node.role, parent)
                    self._trace_task(run_id, parent, node, "child_created", agent_id=handle.identity.agent_id)
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
                        cancelled_before_start = future.cancel()
                        node.status, node.error = "timed_out", f"Task budget {node.timeout_s}s exceeded; worker may still run"
                        item = SubAgentResult(task_id=node.task_id, agent_id=handle.identity.agent_id,
                                              status="timed_out", error=node.error, duration_ms=duration,
                                              worker_running=not future.done(), run_id=run_id,
                                              error_code=AgentTimeoutError.error_code)
                        self._trace_task(run_id, parent, node, "task_timed_out", agent_id=handle.identity.agent_id,
                                         duration_ms=duration, success=False, error_type="Timeout",
                                         worker_running=item.worker_running,
                                         error_code=AgentTimeoutError.error_code)
                        if cancelled_before_start:
                            self._trace_task(run_id, parent, node, "task_cancelled", agent_id=handle.identity.agent_id,
                                             duration_ms=duration, success=False, error_type="FutureCancelled",
                                             error_code=AgentCancelled.error_code)
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
                                                  status="success", output=output, duration_ms=duration,
                                                  run_id=run_id)
                            self._trace_task(run_id, parent, node, "task_success", agent_id=handle.identity.agent_id,
                                             duration_ms=duration, success=True)
                        except Exception as exc:
                            node.status, node.error = "failed", f"{type(exc).__name__}: {exc}"
                            item = SubAgentResult(task_id=node.task_id, agent_id=handle.identity.agent_id,
                                                  status="failed", error=node.error, duration_ms=duration,
                                                  run_id=run_id,
                                                  error_code=getattr(exc, "error_code", "child_task_error"))
                            self._trace_task(run_id, parent, node, "task_failed", agent_id=handle.identity.agent_id,
                                             duration_ms=duration, success=False, error_type=type(exc).__name__,
                                             error_code=item.error_code)
                    results[node.task_id] = item
            return [results[task.task_id] for task in tasks]
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
