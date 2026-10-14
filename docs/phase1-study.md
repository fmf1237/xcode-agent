# 第一阶段学习笔记

## 模块职责和接口

| 模块 | 职责 | 输入 | 输出 |
| --- | --- | --- | --- |
| `config.py` | 从 `.env` / 环境变量构造配置并验证范围 | 环境变量 | `Settings` |
| `llm.py` | 把 OpenAI 兼容 Chat Completions 转为统一响应，配置超时及 SDK 有限重试 | 消息、工具 Schema | `LLMResponse` 或 `LLMError` |
| `tools.py` | 注册工具、生成 JSON Schema、校验参数、隔离执行异常 | 工具名、JSON 字符串 | `ToolResult` |
| `file_tools.py` | 仅在工作目录内列文件和读取 UTF-8 文本 | 相对路径 | 文件名列表或文本 |
| `context.py` | 保存会话消息，按完整用户轮次裁剪历史 | user / assistant / tool 消息 | 可传给模型的消息副本 |
| `agent.py` | 反复调用模型、执行所有工具调用、返回最终答案 | `run(user_input)` | 答案字符串，或模型/轮数异常 |
| `main.py` | 交互式命令行与会话 reset | 终端输入 | 终端输出 |

## 调用关系

`main.py` 创建 `Settings → LLMClient + ToolRegistry + ContextManager → Agent`。`Agent.run()` 把 user 消息放入上下文并请求 LLM。如果模型给出 `tool_calls`，先完整保存 assistant 工具调用消息，再依次执行每个工具，并按原始 `tool_call_id` 保存对应的 tool 消息。所有结果回传后再次请求 LLM，直到得到最终 assistant 文本或达到轮数上限。

工具参数是 JSON 对象，由 Pydantic 模型验证。未知工具、无效参数和 Python 函数异常都成为 `ok=false` 的工具结果，因此模型可以看到错误并继续回答。模型请求失败则抛出 `AgentLLMError`，不会伪装为工具错误。上下文是会话私有对象；未来并发派发时，各 Agent 实例应持有独立的 `ContextManager`。

历史上限按完整 user 轮次删除，从而不拆开 assistant 的 `tool_calls` 与关联 tool 消息。若单轮本身超过上限，保留整轮，这是软上限。文件工具拒绝绝对路径、`..`、解析后离开工作目录的符号链接；限制目录为 100 个条目、文件为 1 MB、读取内容为 20,000 字符。只允许 UTF-8 文本。此方案是应用层路径限制，不把工作目录当作操作系统级沙箱。

## 固定模拟流程

运行 `python examples/fake_demo.py`：

1. 用户输入“Read the demo file”。
2. FakeLLM 首次返回 `read_file({"path":"examples/demo.txt"})`，调用 ID 为 `demo-1`。
3. `ToolRegistry` 验证参数并执行文件工具，返回 `Hello from xcode-agent.`。
4. Agent 保存与 `demo-1` 对应的 tool 消息，再次调用 FakeLLM。
5. FakeLLM 返回最终文本。整个演示不访问网络。

## 本阶段边界

不提供 shell、任意代码执行、多 Agent、DAG、数据库记忆、RAG、Skills 或 MCP。未来可以在现有 `ChatClient` 和工具注册接口外围添加这些能力。
