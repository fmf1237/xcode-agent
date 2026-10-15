# 第三阶段：错误分类与可靠性学习笔记

## 1. 为什么要分类错误？

“请求失败”不等于“应该重试”。例如 HTTP 429 表示暂时被限流，通常可以稍后再请求；HTTP 400 常表示请求本身有问题，再发一次仍会失败。工具参数错误应该反馈给模型修正参数，不能重新执行工具。分类后的异常有稳定的 `error_code` 和 `retryable`，Agent 不必从异常文字里猜测。

```python
from xcode_agent import LLMRateLimitError

error = LLMRateLimitError("rate limited", retry_after_s=2)
print(error.error_code)  # llm_rate_limit
print(error.retryable)   # True
```

## 2. 异常类型与继承关系

`xcode_agent/errors.py` 定义如下结构：

```text
AgentError (error_code="agent_error", retryable=False)
├─ AgentLLMError (llm_error)
│  ├─ LLMTimeoutError (llm_timeout, retryable=True)
│  ├─ LLMRateLimitError (llm_rate_limit, retryable=True)
│  └─ LLMTransientError (llm_transient, retryable=True)
├─ ToolArgumentError (tool_argument_error)
├─ ToolExecutionError (tool_execution_error)
├─ AgentTimeoutError (agent_timeout)
├─ AgentCancelled (agent_cancelled)
└─ DependencyFailedError (dependency_failed)
```

`AgentLLMError` 和 `AgentCancelled` 仍可从旧位置导入。旧名 `xcode_agent.llm.LLMError` 是 `AgentLLMError` 的别名。已有 `MaxRoundsExceeded` 仍保持原来的独立异常类型。未知模型错误转为 `AgentLLMError(retryable=False)`，因此默认不会重试。

`ToolRegistry.execute()` 返回的 `ToolResult.error_code` 是结构化错误标识，但其 JSON 消息仍只有原来的 `ok`、`content`、`error` 三个键；这保证了已有模型工具反馈格式。多 Agent 的 `SubAgentResult.error_code` 可由 Python 调用方读取，但不进入原有依赖结果 JSON。

## 3. LLM 有限重试流程

`Agent.run()` 的**每一轮模型调用**最多尝试 3 次，包含首次请求。`LLMClient` 保留原参数 `retries=2`，并将它解释为“首次请求之外最多再试 2 次”；SDK 内置重试固定为 0，避免两层重试叠加。可通过 `Agent(max_llm_attempts=...)` 调整次数，或通过 `LLMClient(retries=...)` 调整实际客户端的次数。

```text
检查 Agent 截止时间与取消信号
  → llm_start
  → 调用非流式 chat()
  → llm_end(成功) → 继续原 Agent 流程
  → llm_end(失败) → 分类错误
      → 不可重试 / 次数用完 / 已有流式部分输出：抛出
      → 可重试：计算等待时间 → llm_retry → 分段等待并反复检查取消和截止时间 → 下一次尝试
```

可重试的明确情况：OpenAI SDK 的 `APITimeoutError`、`RateLimitError`，以及 `APIStatusError` 中的 408、429、500、502、503、504。其他状态、连接错误和普通 `RuntimeError` 默认不可重试。这里使用 SDK 异常类型或 HTTP 状态码，绝不解析“错误文字里是否写了 429”。本项目 `LLMClient.chat()` 只返回完整的非流式响应；标记 `streaming=True` 的自定义客户端，或异常标记 `partial_output=True` 时，Agent 不自动重试。

### 指数退避怎么算？

假设基础等待为 0.5 秒、抖动刚好取中间值、最多等待 8 秒：第一次失败后等待 `0.5 × 2⁰ = 0.5` 秒；第二次失败后等待 `0.5 × 2¹ = 1` 秒。实际等待再乘一个 0.8～1.2 的随机因子，以免很多客户端同时重发。若 429 的 `Retry-After` 明确给出 2 秒，优先用 2 秒；等待仍受 `retry_max_delay_s` 上限和 Agent 剩余时间限制。`Retry-After` 可以是秒数，也可以是 HTTP 日期。

下面用假时钟代替真实等待：

```python
class FakeClock:
    now = 0.0
    def __call__(self):
        return self.now
    def wait(self, seconds):
        self.now += seconds

clock = FakeClock()
# Agent(fake_llm, tools, clock=clock, wait_fn=clock.wait,
#       jitter_fn=lambda: 0.5, retry_base_delay_s=0.5)
```

离线测试使用类似方法，所以测试不会因为退避而真的等待数秒。

## 4. Tool Error 和 LLM Error 有什么区别？

模型请求失败时，Agent 没有可用响应，最终抛出 `AgentLLMError` 或它的子类。工具参数或执行失败时，`ToolRegistry.execute()` 返回 `ToolResult(ok=False, error_code=...)`；Agent 照常写入 tool 消息，再让模型决定下一步。工具处理函数只执行一次，**不会自动重试**。这很重要，因为工具可能有写文件、发请求等副作用。

```python
result = registry.execute("read_file", '{"path": "../outside"}')
print(result.ok)          # False
print(result.error_code)  # tool_execution_error（路径检查由工具函数执行）
```

参数 JSON 解析或 Pydantic 校验失败对应 `tool_argument_error`；工具函数抛异常对应 `tool_execution_error`。未知工具会得到 `tool_not_found`。这些代码不靠错误字符串判断。

## 5. 三种“时间/停止”概念

1. **单次 LLM 调用超时**：`LLMClient(timeout=30)` 控制一次 SDK 请求。SDK 抛 `APITimeoutError` 时变为 `LLMTimeoutError`；如果还有预算，非流式请求可以重试。
2. **Agent 总超时**：`agent.run("问题", total_timeout_s=5)` 为整个 run 设截止时间，包含多轮模型调用、工具和重试等待。达到后抛不可重试的 `AgentTimeoutError`，阻止下一次请求。
3. **多 Agent 子任务超时与取消**：`TaskNode(timeout_s=...)` 是调度器的任务预算。调度器到期后返回 `timed_out`，对工作线程发协作式取消信号。线程在 Agent 的下一个检查点才停止；已经阻塞中的模型或工具函数无法被 Python 安全地强制中断。

主动取消使用 `agent.run(..., cancelled=lambda: stop_event.is_set())`。重试等待按短间隔检查这个函数；观察到取消即抛 `AgentCancelled(retryable=False)`，不会再发送下一次请求。多 Agent 调度器原有的 `Event` 继续作为取消来源。

## 6. Trace 与错误系统

原有 `agent_start`、`llm_start`、`llm_end`、`tool_start`、`tool_end`、`agent_end`、`agent_error` 事件继续存在。每次模型请求尝试各有一组 `llm_start → llm_end`，并在 `metadata` 中记录 `attempt`、`max_attempts` 和失败的 `error_code`。决定重试时新增 `llm_retry`，记录下一次等待的 `delay_ms`。`agent_error` 记录总超时、取消或最终模型错误代码；`tool_end` 记录工具错误代码。依赖失败的 `task_skipped` 记录 `dependency_failed`。

Trace 只保存错误类型和代码，不保存 API Key、完整用户消息、工具原始参数、模型回答或异常正文。某些标识（如 `task_id`、`tool_name`）仍会入 Trace；不要把秘密放进标识里。

## 7. 关键代码位置、输入和输出

| 位置 | 输入 | 输出或作用 |
| --- | --- | --- |
| `xcode_agent/errors.py:74` `classify_llm_exception(exc)` | SDK 或其他异常 | 返回有 `error_code`、`retryable` 的 `AgentLLMError` 子类；未知错误不可重试 |
| `xcode_agent/errors.py:57` `parse_retry_after(value, now)` | 秒数或 HTTP 日期 | 非负等待秒数；无效时 `None` |
| `xcode_agent/llm.py:28` `LLMClient(...)` | Key、base URL、模型、单次超时、额外重试次数 | 非流式客户端；SDK 内置重试关闭 |
| `xcode_agent/llm.py:40` `chat(messages, tools)` | 上下文消息与工具 Schema | 完整 `LLMResponse` 或已分类的模型异常 |
| `xcode_agent/agent.py:46` `run(user_input, cancelled, run_id, total_timeout_s)` | 用户输入、取消函数、可选追踪 ID 和总时限 | 最终回答或结构化异常；在调用循环中执行有限重试 |
| `xcode_agent/agent.py:62` `wait_for_retry(delay)` | 预计等待秒数 | 无返回值；分段等待，期间检查取消与截止时间 |
| `xcode_agent/tools.py:56` `execute(name, raw_arguments)` | 工具名、JSON 参数 | `ToolResult`；参数和执行失败仍是可恢复反馈 |
| `xcode_agent/multi_agent.py:50` `SubAgentResult` | 调度结果字段 | `status`、`error_code` 等结构化结果 |
| `xcode_agent/multi_agent.py:132` `_trace_task(...)` | 子任务和状态 | 写入任务 Trace，包括依赖失败、超时和取消类别 |
| `xcode_agent/tracing.py:38` `record(...)` | 运行 ID、事件类型、少量元数据 | 保存事件并返回副本；只允许白名单元数据 |

## 8. 面试可能问的 5 道题

1. **为什么不能根据异常字符串决定重试？** 文本会随 SDK、语言和服务商变化，还可能包含敏感信息；异常类型和 HTTP 状态码更稳定。
2. **为什么要关闭 SDK 自己的重试？** 两层重试会让实际请求次数相乘，并且 Agent 看不到内部尝试，难以落实总时限、取消与 Trace。
3. **工具执行失败为何不自动重试？** 工具可能有副作用，失败时不一定知道副作用是否已发生；反馈模型比盲目重做更安全。
4. **Agent 总超时能否立刻停止正在进行的同步网络请求？** 不能强行中断；单次 SDK 超时控制阻塞上限，总截止时间在请求返回及重试等待检查点生效。
5. **429 的等待时间从哪里来？** 先解析服务端 `Retry-After`，再受项目最大等待时间和任务剩余时间限制；没有有效值时用指数退避加抖动。
