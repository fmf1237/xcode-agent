# 第三阶段：联调与性能评测学习笔记

## 1. 两类验证不能混为一谈

`examples/phase3_benchmark.py` 使用固定 Mock LLM 和 Mock Tool，测量本机调度与人为等待的 wall-clock 时间。它不会访问模型 API，数字**不是**真实模型延迟、吞吐量或价格。`examples/phase3_live_smoke.py` 是独立的真实 API 冒烟示例，必须显式加 `--live`；它检查接口联通、简单回答、只读工具调用和 Trace，不负责性能基准。

本次环境未配置 `.env`、`OPENAI_API_KEY` 和 `OPENAI_MODEL`，因此真实 API 测试结果为 **not_run**，没有真实模型性能数据。

## 2. 评测输入与输出

默认输入：任务数量 2、4、8；每组重复 3 次；最多 4 个工作线程；Mock LLM 每次调用等待 15 ms，Mock Tool 等待 5 ms，模拟 429 后 `Retry-After` 为 5 ms。每个任务先调用模型获取工具调用，再执行一个无状态 Mock Tool，最后再调用模型得到答案。重试模式会在每个子 Agent 第一次模型请求上额外抛一次模拟限流异常。

两种图结构：

- `independent`：所有任务没有依赖，可同时就绪。
- `layered_dag`：前半任务是根节点，后半任务分别依赖一个根节点；有可并行的任务，也有必须等待的任务。

每种图都分别以 1 个线程（串行）和 `min(任务数, 4)` 个线程（并行）运行。Mock Tool 是无共享状态的，因此评测显式设置 `synchronize_tools=False`；正式项目的工厂默认会同步共享工具处理函数。这个选择会影响并行加速，不能把结果直接套用到有锁或有真实 I/O 的工具上。

运行命令：

```powershell
python examples/phase3_benchmark.py
```

脚本输出 [JSON 原始报告](../reports/phase3/benchmark.json) 和 [Markdown 汇总](../reports/phase3/benchmark.md)。可以用 `--task-counts`、`--repeats`、`--max-workers`、`--llm-delay-ms`、`--tool-delay-ms`、`--retry-delay-ms`、`--output-dir` 调整输入。固定工作负载可重复运行；操作系统调度使实测时间不会每次完全相同。

| 函数 | 输入 | 输出 |
| --- | --- | --- |
| `make_tasks(count, topology)` | 任务数、图类型 | `TaskNode` 列表 |
| `run_case(count, topology, retry, workers, ...)` | 单次场景参数 | wall-clock 毫秒、成功任务数、状态、Trace 计数 |
| `summarize_runs(samples, workers)` | 多次单场景记录 | 中位耗时、样本、成功率、平均尝试和重试次数 |
| `build_report(...)` | 全部场景参数 | 可序列化的汇总字典 |
| `markdown_report(report)` | 汇总字典 | Markdown 表格文本 |

## 3. 指标怎么算？

- **wall-clock 耗时**：`(time.perf_counter() 结束值 - 开始值) × 1000`，从调用 `dispatch()` 前到它返回后；包含线程创建、调度、Mock 等待和 Trace 开销。报告保留每次样本，并使用样本中位数比较。
- **成功率**：所有重复运行中 `status == "success"` 的任务数 ÷ 已调度任务数。
- **加速比**：同一任务数、图结构和失败模式下，串行耗时中位数 ÷ 并行耗时中位数。加速比小于 1 表示并行反而较慢。
- **LLM 尝试次数**：同一运行 ID 下 `llm_start` 事件数。一个任务正常需要两次成功的模型调用，所以是 `2 × 任务数`。
- **重试次数**：同一运行 ID 下 `llm_retry` 事件数。模拟限流模式每个任务增加一次失败尝试和一次重试，所以是 `任务数`。

计数从 Trace 读取，不根据预期值直接填表。`tests/test_phase3_examples.py` 用零延迟场景验证了这些关系。

## 4. 本次实际离线结果

下面是本次默认配置的中位 wall-clock 时间，单位毫秒；每行都重复 3 次。全部场景成功率为 **100%**。原始样本见 JSON 报告。

| 任务数 | 图结构 | 模式 | 串行 ms | 并行 ms | 加速比 | 尝试/运行 | 重试/运行 |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2 | independent | 正常 | 76.486 | 38.695 | 1.977 | 4 | 0 |
| 2 | independent | 模拟 429 | 118.942 | 59.725 | 1.991 | 6 | 2 |
| 2 | layered_dag | 正常 | 74.752 | 73.992 | 1.010 | 4 | 0 |
| 2 | layered_dag | 模拟 429 | 117.880 | 117.520 | 1.003 | 6 | 2 |
| 4 | independent | 正常 | 150.981 | 38.696 | 3.902 | 8 | 0 |
| 4 | independent | 模拟 429 | 235.607 | 60.080 | 3.922 | 12 | 4 |
| 4 | layered_dag | 正常 | 151.141 | 76.048 | 1.987 | 8 | 0 |
| 4 | layered_dag | 模拟 429 | 236.681 | 118.955 | 1.990 | 12 | 4 |
| 8 | independent | 正常 | 300.429 | 77.821 | 3.861 | 16 | 0 |
| 8 | independent | 模拟 429 | 468.433 | 120.800 | 3.878 | 24 | 8 |
| 8 | layered_dag | 正常 | 302.062 | 78.504 | 3.848 | 16 | 0 |
| 8 | layered_dag | 模拟 429 | 469.412 | 119.523 | 3.927 | 24 | 8 |

**为什么加速比不会随 Agent 数量线性增长？** 线程池最多 4 个工作线程；8 个独立任务至少分两批完成。DAG 的依赖也会让一部分任务必须等待前置任务，例如 2 个任务的分层 DAG 基本只能串行。线程创建、调度、Trace、Python 解释器和共享工具锁都会带来开销。真实模型还会受到网络、服务端队列、限流和 token 数量影响。因此这份报告只说明固定 Mock 工作负载下的本机现象。

## 5. 真实 API 冒烟测试如何运行？

把真实配置写在项目 `.env` 或同名环境变量中：`OPENAI_API_KEY`、`OPENAI_MODEL`、`OPENAI_BASE_URL`。脚本默认不访问网络；要主动测试时运行：

```powershell
python examples/phase3_live_smoke.py --live
```

可用 `--request-timeout` 设置单次 SDK 请求上限，用 `--total-timeout` 设置每个 Agent 运行的总时限。脚本使用现有 `LLMClient`，先要求简单回答，再要求模型调用 `read_file` 读取项目内的 `examples/demo.txt`。它只输出状态、事件类型、尝试次数等摘要，不输出 Key、完整提示词、模型答案或原始异常正文。若模型没有实际完成工具调用，结果为 `failed`，不会假报通过。

本次运行 `python examples/phase3_live_smoke.py --live` 的结果是 `not_run`，原因是缺少 API Key 或模型配置；没有真实 API 请求。

## 6. 可靠性边界检查

- **同步调用超时**：`Agent` 只能在调用前后及重试等待期间检查总时限。`tests/test_checkpoint_history.py` 用假时钟验证：模型阻塞到截止时间后返回时，Trace 先记录一次成功的 `llm_end`，随后以 `agent_timeout` 结束 Agent。单次 SDK 请求超时限制这段阻塞，二者作用不同。
- **失败与重试 Trace**：失败尝试有 `llm_end(success=False)`，决定重试时有 `llm_retry`，成功尝试另有 `llm_end(success=True)`；最后是 `agent_end` 或 `agent_error`。
- **并行与竞态**：子 Agent 各有独立上下文和客户端；TraceRecorder 用锁保护事件存储。超时后调度器可能先返回 `timed_out`，工作线程稍后才观察到取消并补记事件，这是协作式取消的预期时序。
- **重复副作用**：LLM 请求重试不会自动重复已完成的工具调用，已有离线测试验证。若模型在新的响应中再次要求执行同一有副作用工具，框架目前不提供幂等去重；正在执行的工具也不能被强制中断。
- **工具调用组完整性**：此次边界检查先用失败测试复现了“多个工具调用之间取消会留下缺失 tool 回复”的问题，随后在 Agent 的异常路径为未完成调用补上失败回复。未执行的工具不会被启动，后续会话仍保有合法的 `tool_call_id` 对应关系。

## 7. 验证结果

本次执行 `python -m pytest -q`：**58 passed，1 skipped**。跳过的是当前 Windows 环境无法创建符号链接的原有文件工具测试。离线评测已运行并生成上述两个报告；真实 API 冒烟未执行，因为没有配置密钥和模型。
