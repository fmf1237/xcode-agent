# 第二阶段：多 Agent DAG 派发

## 1. 与单 Agent 的区别

第一阶段的 `Agent.run()` 在一条会话里反复进行模型与工具调用。第二阶段保留这个循环，把多个独立 `Agent` 实例作为 DAG 节点的执行者。主 orchestrator 持有计划和聚合结果；每个子 Agent 只获得自己的任务描述、已完成依赖的结果和私有会话。`leaf` 不可派发；`orchestrator` 可继续派发。

## 2. ThreadPoolExecutor

`DAGScheduler` 在一次顶层派发中创建一个 `ThreadPoolExecutor(max_workers=N)`。准备好的节点最多提交 N 个；`Future` 记录完成、异常和运行状态。线程适合等待模型 API 或模拟 I/O；CPU 密集型 Python 工作不一定加速。嵌套计划在已占用的工作线程内顺序执行，避免每个子 Agent 再创建线程池导致资源耗尽或互等死锁。嵌套层的独立节点因此不并行。

任务超时从提交时计预算。调度器在预算到期时设置取消事件，并返回 `timed_out`；`worker_running=true` 明确表示底层线程仍可能在模型或工具的阻塞调用中。`Agent.run()` 会在模型请求及每次工具调用边界检查取消。线程无法被强制杀死；调用真正结束后完成回调把 `worker_running` 改为 `false`。若模型永久阻塞，Python 进程仍可能等待该线程退出。生产环境需要外部进程隔离或服务端超时保证。嵌套顺序任务的超时在阻塞操作返回后确认，因此其返回可能晚于预算。

## 3. DAG 调度

`TaskNode` 包含 `task_id`、`description`、`depends_on`、`status`、`result`、`error`，并可指定 `timeout_s`、`role`、`subtasks`。调度前检查重复 ID、缺失依赖、环、非法 leaf 子计划和非 pending 初态。调度器按输入顺序寻找依赖已成功的节点，将无依赖节点并行提交。依赖失败或超时的节点标记为 `skipped`，错误包含依赖 ID。每个节点最终为 `success`、`failed`、`skipped` 或 `timed_out`，返回列表保持输入顺序。

`SubAgentResult` 包含任务 ID、Agent ID、状态、输出、错误、耗时（毫秒）与工作线程是否仍在运行。某个节点失败不会吞掉其他节点结果。

## 4. 角色和深度

角色阻止 leaf 递归创建 Agent。`AgentFactory(max_depth=...)` 记录根 Agent 深度 0、子 Agent 深度 1，超限时抛出 `DispatchError`；若发生在一个任务内，该任务返回明确的失败信息。结构化 `subtasks` 可让 orchestrator 节点继续派发，直到深度上限。

## 5. 上下文和共享组件

工厂为每个子 Agent 创建新的 `ContextManager` 与新的 LLM 客户端，并通过 `AgentFactory.identities` 保留 Agent ID、父 ID、角色及深度的记录。`llm_factory` 必须返回独立客户端；不能把同一个带内部状态的 FakeLLM 或客户端实例返回给多个子 Agent。工具注册表复制定义；默认 `synchronize_tools=True`，通过共享锁串行调用原始处理函数，保护未知线程安全性的处理函数。若处理函数本身明确线程安全，可关闭此选项提高工具并发。文件工具为只读，但仍保留工作目录边界。消息副本由 `ContextManager.messages` 返回，子 Agent 不能修改兄弟 Agent 历史。

## 6. 完整输入输出

```python
from xcode_agent import AgentFactory, TaskNode, ToolRegistry, LLMClient
import os

def new_llm():
    return LLMClient(api_key=os.environ["OPENAI_API_KEY"],
                     base_url=os.environ["OPENAI_BASE_URL"],
                     model=os.environ["OPENAI_MODEL"])

root = AgentFactory(new_llm, ToolRegistry(), max_depth=2).create()
plan = [
    TaskNode(task_id="a", description="收集事实"),
    TaskNode(task_id="b", description="基于事实形成摘要", depends_on=["a"]),
]
results, answer = root.dispatch_and_summarize(plan, "综合各任务结果回答用户", max_workers=2)
```

调用顺序是：主 Agent 提交计划 → 工厂创建具有 `parent_agent_id` 的子 Agent → 子 Agent 的 user 消息包含任务描述及依赖结果 → 子 Agent 运行原有 `Agent.run()` → 调度器收集 `SubAgentResult` → `reason_with_results()` 把按输入顺序排列的 JSON 结果作为主 Agent 的新 user 消息，请主 Agent 推理并输出答案。这里的计划由调用者提供，尚无 LLM 自动规划。

## 7. 模拟性能实验

运行 `python examples/phase2_benchmark.py`。同样的四个本地模拟 I/O 任务，每次模型调用用约 150 ms 的 `sleep` 代替，完全离线。一次实际运行：

| 模式 | 最大工作线程 | 总耗时 | 各任务耗时 | 加速比 |
| --- | ---: | ---: | --- | ---: |
| 串行 | 1 | 602.49 ms | 156、157、140、157 ms | 基准 |
| 并行 | 4 | 151.42 ms | 140、140、156、156 ms | 3.98× |

这是本地模拟等待的单次测量，不代表真实 LLM API 的速度、限流或稳定性；每次运行数字会变动。

## 8. 局限性与面试追问

- 如何处理永久阻塞的 LLM？当前仅协作取消，无法杀死 Python 线程；可讨论进程隔离、HTTP 超时、服务端取消和幂等任务。
- 为什么嵌套任务顺序执行？共享固定线程预算可避免嵌套线程池死锁；更大规模可考虑单一事件循环或中央队列。
- 如何处理 API 限流？当前无全局限流与退避协调；`LLMClient` 仅做单次请求的有限重试。
- 工具为何默认加锁？未知处理函数可能有共享可变状态；代价是工具执行被串行化。真实实现应按工具粒度定义线程安全策略。
- 计划如何生成和校验？当前由调用者提供结构化 DAG，尚无模型自动拆解、计划审查或持久化恢复。
- 当前耗时从提交起计算，包括极短的排队时间；未提供分布式执行、长期记忆、RAG、Skills 或 MCP。
