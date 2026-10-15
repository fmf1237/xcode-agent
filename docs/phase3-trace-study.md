# 第三阶段：Trace 运行追踪学习笔记

## 1. Trace 是什么，为什么需要？

Trace 是一次运行留下的**事件时间线**。例如，一个 Agent 回答问题时先请求模型、再调用文件工具、再请求模型，Trace 会留下“何时开始、是否成功、用了多久”的记录。它帮助我们回答：慢在哪里？失败发生在哪一步？并行的两个子 Agent 各自做了什么？

Trace 不是对话记忆。对话内容仍保存在 `ContextManager`；Trace 只存事件和少量标识，不保存完整问题、模型回复、工具参数或 API Key。

## 2. 新模块与重要函数

| 位置 | 函数或类 | 输入 | 输出与作用 |
| --- | --- | --- | --- |
| `xcode_agent/tracing.py:10` | `TraceEvent` | `run_id`、`agent_id`、事件类型等字段 | 一个不可重新赋值的 Pydantic 事件对象 |
| `xcode_agent/tracing.py:23` | `TraceRecorder` | 无；构造时建内存字典与线程锁 | 多线程可共用的事件记录器 |
| `xcode_agent/tracing.py:33` | `new_run_id()` | 无 | 随机的唯一运行 ID 字符串 |
| `xcode_agent/tracing.py:37` | `record(...)` | 运行 ID、Agent ID、事件类型、可选耗时和状态 | 保存事件并返回副本；过滤不允许的 metadata 键 |
| `xcode_agent/tracing.py:51` | `events(run_id)` | 运行 ID | 按记录顺序返回事件副本列表；未知 ID 返回空列表 |
| `xcode_agent/agent.py:41` | `Agent.run(...)` | 用户输入、可选取消函数、可选 `run_id` | 最终答案；失败时仍抛原有异常并留下 Trace |
| `xcode_agent/multi_agent.py:65` | `AgentHandle.dispatch(...)` | DAG 任务列表、线程数、可选 `run_id` | 子任务结果列表；结果含同一个 `run_id` |
| `xcode_agent/multi_agent.py:109` | `AgentFactory.create(...)` | 角色、可选父 Agent | 拥有独立上下文和 Agent ID 的 `AgentHandle`，共享工厂 Trace |
| `xcode_agent/multi_agent.py:130` | `_trace_task(...)` | 当前任务、事件类型、结果状态 | 在调度器里统一写入任务事件；无返回值 |

`Agent` 和 `AgentFactory` 的 `trace` 参数都可省略；省略时自动创建内存记录器。因此原有调用方式保持可用。

## 3. TraceEvent 字段

```python
event = trace.events(run_id)[0]
print(event.event_type, event.success, event.duration_ms)
```

| 字段 | 含义 |
| --- | --- |
| `run_id` | 一次 Agent 运行或一次 DAG 派发的 ID。父子任务共享它。 |
| `agent_id` | 产生事件的 Agent ID。并行子 Agent 各不相同。 |
| `event_type` | 事件名称，如 `llm_start`、`tool_end`、`task_timed_out`。 |
| `timestamp` | UTC 时区的事件时间。 |
| `duration_ms` | 有耗时的结束事件所用毫秒数；开始事件一般为 `None`。 |
| `success` | 适用时为 `True` 或 `False`；开始事件一般为 `None`。 |
| `error_type` | 异常类名或概括的失败类型，不包含异常正文。 |
| `metadata` | 白名单内的关联标识，例如 `round`、`tool_name`、`tool_call_id`、`task_id`、`parent_agent_id`。 |

记录器用 `RLock` 保护内存字典。`events()` 返回深拷贝，所以调用方修改列表或 `metadata` 不会改坏已保存的事件。进程退出后内存 Trace 消失；本阶段没有数据库持久化。

## 4. 单 Agent 调用链路

`Agent.run()` 先写 `agent_start`，再进入原有的模型与工具循环。每次调用模型都有 `llm_start → llm_end`；模型给出工具调用时，每个工具都有 `tool_start → tool_end`。得到最终回答写 `agent_end`。任何未恢复的异常写 `agent_error` 后继续抛给原调用方。

例如：

```python
from xcode_agent import Agent, ToolRegistry, TraceRecorder

trace = TraceRecorder()
agent = Agent(fake_llm, ToolRegistry(), trace=trace)
answer = agent.run("你好")
for event in trace.events(agent.last_run_id):
    print(event.event_type)
```

其中 `fake_llm` 只需实现现有 `chat(messages, tools)` 接口；离线测试中的 `FakeLLM` 就是具体例子。调用真实模型时，仍按原项目配置 `LLMClient`。

### 包含 LLM 与 Tool 的完整事件示例

假设模型第一轮调用 `double(value=2)`，第二轮回答“4”。下表是事件顺序示意；时间和耗时每次运行都不同。

| 顺序 | event_type | success | 关键 metadata |
| --- | --- | --- | --- |
| 1 | `agent_start` | 空 | 空 |
| 2 | `llm_start` | 空 | `round=1` |
| 3 | `llm_end` | `True` | `round=1` |
| 4 | `tool_start` | 空 | `tool_name=double`, `tool_call_id=c1` |
| 5 | `tool_end` | `True` | 同上，且有 `duration_ms` |
| 6 | `llm_start` | 空 | `round=2` |
| 7 | `llm_end` | `True` | `round=2` |
| 8 | `agent_end` | `True` | 有总耗时 |

Trace 中看不到 `value=2`、用户问题和最终回答；这些数据只走原有模型上下文和工具接口。

## 5. Multi-Agent 如何关联

一次 `dispatch()` 生成一个 `run_id`，返回的每个 `SubAgentResult.run_id` 相同。`AgentFactory` 创建的子 Agent 使用同一个 `TraceRecorder`，但各有自己的 `agent_id`。调度器记录 `child_created`、`child_start`、`task_success`、`task_failed`、`task_skipped`、`task_timed_out`；合作式取消完成时还会记录 `task_cancelled`。

任务事件的 `metadata.task_id` 标识 DAG 节点，`parent_agent_id` 指向创建它的父 Agent。嵌套 DAG 还用 `parent_task_id` 指向外层任务。Agent 的 LLM 和工具事件不重复存任务说明；通过同一 `run_id` 与子 Agent 的 `agent_id`，即可与 `child_created` 事件关联。

```python
results = root.dispatch(tasks, max_workers=2)
run_id = results[0].run_id
events = root.factory.trace.events(run_id)
child_events = [e for e in events if e.agent_id == results[0].agent_id]
```

并行线程的事件可能交错，这是正常的；同一 Agent 内的顺序仍可按事件列表查看。不要仅凭全局列表位置推断两个并行任务的先后因果。

## 6. 异常和超时流程

- **模型失败**：`llm_end(success=False, error_type=原异常类名)`，接着 `agent_error(error_type=AgentLLMError)`；调用方仍收到原项目的 `AgentLLMError`。
- **工具失败**：`ToolRegistry.execute()` 返回 `ok=False`，Trace 写 `tool_end(success=False)`；结果照常回传模型，因此 Agent 仍可能正常回答并写 `agent_end`。
- **最大轮数或取消**：写 `agent_error`，原有异常继续向外传播。
- **依赖失败**：调度器不创建该子 Agent，直接写 `task_skipped(error_type=DependencyFailed)`。
- **超时**：调度器先写 `task_timed_out` 并请求取消；若工作线程随后观察到取消，另写 `task_cancelled`。阻塞中的模型调用不能被强行中断，所以 `worker_running=True` 表示返回超时结果时线程尚未退出。

错误的详细文本仍存在原有异常或任务结果里，但不会进入 Trace。注意：`task_id`、工具名等标识会进入 Trace，命名时不要把密钥或个人信息写进这些标识。

## 7. 面试可能问的 5 道题

1. **Trace 和日志有什么区别？** Trace 用 `run_id`、`agent_id` 将多个步骤和并行子任务连成一次运行；普通日志通常只是文本行。两者可以并存。
2. **为什么每个 `start` 要有对应结束事件？** 否则无法区分步骤仍在执行、成功还是失败，也算不出准确耗时。模型和工具调用都在成功、失败分支写结束事件。
3. **为什么需要线程锁？** 多个子 Agent 的工作线程共享一个记录器，写入和读取事件列表需要同步，避免并发修改造成丢失或不一致。
4. **超时为什么不等于线程已经停止？** Python 线程池不能安全地强杀正在执行的函数。调度器先返回超时状态，工作线程在下一个取消检查点才停止。
5. **怎样避免 Trace 泄漏敏感信息？** 只记录白名单字段和错误类型，不记录消息正文、工具原始参数、工具输出与异常消息；同时不要把秘密放进任务 ID 等标识。
