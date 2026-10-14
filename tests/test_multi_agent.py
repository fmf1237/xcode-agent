import json
from threading import Barrier, Event, Lock

import pytest

from xcode_agent import (AgentFactory, DAGScheduler, DispatchError, LLMResponse,
                         PlanError, TaskNode, ToolRegistry)


class FunctionLLM:
    def __init__(self, respond):
        self.respond = respond
        self.requests = []

    def chat(self, messages, tools):
        self.requests.append(messages)
        return self.respond(messages)


def setup(respond, max_depth=1):
    clients = []
    def build():
        client = FunctionLLM(respond)
        clients.append(client)
        return client
    factory = AgentFactory(build, ToolRegistry(), max_depth=max_depth)
    return factory, factory.create(), clients


def test_two_independent_tasks_run_concurrently():
    barrier = Barrier(2)
    def respond(messages):
        barrier.wait(timeout=3)
        return LLMResponse(content=messages[-1]["content"])
    _, root, _ = setup(respond)
    results = root.dispatch([TaskNode(task_id="a", description="A"), TaskNode(task_id="b", description="B")], max_workers=2)
    assert [item.status for item in results] == ["success", "success"]
    assert [item.output for item in results] == ["A", "B"]
    assert results[0].agent_id != results[1].agent_id


def test_dependency_order_and_results_passed_to_child():
    seen = []
    def respond(messages):
        prompt = messages[-1]["content"]
        seen.append(prompt)
        return LLMResponse(content="result A" if prompt == "A" else "result B")
    _, root, _ = setup(respond)
    results = root.dispatch([TaskNode(task_id="b", description="B", depends_on=["a"]),
                             TaskNode(task_id="a", description="A")], max_workers=2)
    assert [item.output for item in results] == ["result B", "result A"]
    assert seen[0] == "A"
    assert '"output": "result A"' in seen[1]


@pytest.mark.parametrize("tasks,fragment", [
    ([TaskNode(task_id="a", description="x"), TaskNode(task_id="a", description="y")], "Duplicate"),
    ([TaskNode(task_id="a", description="x", depends_on=["missing"])], "Missing"),
    ([TaskNode(task_id="a", description="x", depends_on=["b"]),
      TaskNode(task_id="b", description="y", depends_on=["a"])], "cycle"),
])
def test_invalid_dag(tasks, fragment):
    with pytest.raises(PlanError, match=fragment):
        DAGScheduler.validate(tasks)


def test_context_isolation_and_identity():
    factory, root, clients = setup(lambda messages: LLMResponse(content=str(len(messages))))
    results = root.dispatch([TaskNode(task_id="a", description="A"), TaskNode(task_id="b", description="B")])
    assert [item.output for item in results] == ["1", "1"]
    assert root.agent.context.messages == []
    assert len(clients) == 3  # root plus two private child clients
    child = factory.create("leaf", root)
    assert child.identity.parent_agent_id == root.identity.agent_id
    assert child.identity.depth == 1
    assert child.agent.context is not root.agent.context
    for result in results:
        recorded = factory.identities[result.agent_id]
        assert recorded.parent_agent_id == root.identity.agent_id
        assert recorded.role == "leaf" and recorded.depth == 1


def test_leaf_and_max_depth_reject_dispatch():
    factory, root, _ = setup(lambda _: LLMResponse(content="ok"), max_depth=1)
    leaf = factory.create("leaf", root)
    with pytest.raises(DispatchError, match="Leaf"):
        leaf.dispatch([])
    with pytest.raises(DispatchError, match="depth"):
        factory.create("orchestrator", leaf)
    factory2, root2, _ = setup(lambda _: LLMResponse(content="ok"), max_depth=0)
    with pytest.raises(DispatchError, match="depth"):
        root2.dispatch([TaskNode(task_id="a", description="A")])


def test_one_failure_does_not_lose_other_result_and_skips_dependents():
    def respond(messages):
        if messages[-1]["content"] == "fail":
            raise RuntimeError("simulated failure")
        return LLMResponse(content="ok")
    _, root, _ = setup(respond)
    results = root.dispatch([TaskNode(task_id="bad", description="fail"),
                             TaskNode(task_id="good", description="succeed"),
                             TaskNode(task_id="after", description="later", depends_on=["bad"])], max_workers=2)
    assert [item.status for item in results] == ["failed", "success", "skipped"]
    assert "bad" in results[2].error
    assert results[1].output == "ok"


def test_timeout_reports_running_worker_then_cooperative_stop():
    entered, release, finished = Event(), Event(), Event()
    def respond(_):
        entered.set()
        release.wait(timeout=3)
        finished.set()
        return LLMResponse(content="late")
    _, root, _ = setup(respond)
    try:
        result = root.dispatch([TaskNode(task_id="slow", description="slow", timeout_s=0.05)])[0]
        assert entered.is_set()
        assert result.status == "timed_out"
        assert result.worker_running
    finally:
        release.set()
    assert finished.wait(timeout=3)
    # Future completion callback updates the snapshot once the worker has stopped.
    for _ in range(1000):
        if not result.worker_running:
            break
        Event().wait(0.001)
    assert not result.worker_running


def test_max_concurrency_limit():
    lock, first_pair, release = Lock(), Event(), Event()
    active = 0
    maximum = 0
    def respond(_):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
            if active == 2:
                first_pair.set()
        assert first_pair.wait(timeout=3)
        assert release.wait(timeout=3)
        with lock:
            active -= 1
        return LLMResponse(content="ok")
    _, root, _ = setup(respond)
    import threading
    holder = []
    runner = threading.Thread(target=lambda: holder.extend(root.dispatch(
        [TaskNode(task_id=str(i), description=str(i)) for i in range(4)], max_workers=2)))
    runner.start()
    try:
        assert first_pair.wait(timeout=3)
        assert maximum == 2
    finally:
        release.set()
        runner.join(timeout=3)
    assert not runner.is_alive()
    assert len(holder) == 4 and all(item.status == "success" for item in holder)


def test_results_return_to_orchestrator_context():
    def respond(messages):
        return LLMResponse(content="summary" if "Sub-agent results" in messages[-1]["content"] else "child answer")
    _, root, _ = setup(respond)
    results, summary = root.dispatch_and_summarize([TaskNode(task_id="x", description="X")], "Summarize")
    assert summary == "summary"
    payload = root.agent.context.messages[0]["content"]
    assert json.loads(payload.split("Sub-agent results (JSON, in task order):\n", 1)[1])[0]["output"] == "child answer"
    assert results[0].agent_id is not None


def test_nested_orchestrator_plan_uses_worker_without_nested_pool():
    def respond(messages):
        prompt = messages[-1]["content"]
        return LLMResponse(content="summary" if "Sub-agent results" in prompt else prompt)
    _, root, _ = setup(respond, max_depth=2)
    plan = [TaskNode(task_id="group", description="Summarize", role="orchestrator",
                     subtasks=[TaskNode(task_id="inner", description="inner work")])]
    result = root.dispatch(plan, max_workers=1)[0]
    assert result.status == "success" and result.output == "summary"


def test_nested_plan_depth_failure_is_task_error():
    _, root, _ = setup(lambda _: LLMResponse(content="ok"), max_depth=1)
    result = root.dispatch([TaskNode(task_id="group", description="Group", role="orchestrator",
                                     subtasks=[TaskNode(task_id="inner", description="Inner")])])[0]
    assert result.status == "failed" and "depth" in result.error
