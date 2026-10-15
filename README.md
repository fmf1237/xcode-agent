# xcode-agent（第一阶段）

一个使用原生 Python 实现的 Agent 项目。第一阶段支持 OpenAI 兼容 Chat Completions、连续会话、多轮及并列工具调用和受限只读文件工具。第二阶段增加结构化 DAG、子 Agent 角色与深度、并行派发及结果回传。长期记忆、Skills 和 MCP 尚未实现。

## 环境与安装

需要 Python 3.11+。在 Windows PowerShell 或 PyCharm 终端运行：

```powershell
git clone https://github.com/fmf1237/xcode-agent.git
cd xcode-agent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

编辑 `.env`：设置 `OPENAI_API_KEY`、`OPENAI_BASE_URL` 和 `OPENAI_MODEL`。`AGENT_WORKSPACE` 是允许文件工具访问的工作目录；相对值按启动进程的当前目录解析。不要提交 `.env`。

## 使用

```powershell
python main.py
```

输入问题后可连续对话；输入 `reset` 清空会话，`exit` 退出。`AGENT_VERBOSE=true` 会打印模型轮次、工具调用和结果。`AGENT_MAX_ROUNDS` 控制每次 `run()` 最多调用模型的次数，`AGENT_HISTORY_LIMIT` 是消息数的软上限，历史只按完整用户轮次裁剪。

`OPENAI_TIMEOUT` 是单次模型请求超时。`OPENAI_RETRIES=2` 表示首次请求之外最多重试 2 次；仅对明确可重试的非流式模型请求生效，SDK 内置重试已关闭。程序调用方还可使用 `agent.run(question, total_timeout_s=5)` 设置整次 Agent 运行的截止时间。工具操作不会自动重试。

## 离线演示与测试

```powershell
python examples/fake_demo.py
python examples/phase2_benchmark.py
python -m pytest -q
```

演示与基准都不需要 API Key 或网络。示例文件是 `examples/demo.txt`。模块与消息流程见 [第一阶段学习笔记](docs/phase1-study.md)、[第二阶段学习笔记](docs/phase2-study.md)、[Trace 学习笔记](docs/phase3-trace-study.md) 和 [可靠性学习笔记](docs/phase3-reliability-study.md)。

## 扩展接口

`Agent` 仅依赖具有 `chat(messages, tools) -> LLMResponse` 方法的客户端及 `ToolRegistry`。`AgentFactory` 通过客户端工厂创建独立子 Agent；`AgentHandle.dispatch()` 接收 `TaskNode` 列表并按输入顺序返回 `SubAgentResult`，`dispatch_and_summarize()` 可将结果送回主 Agent 推理。工具通过 Pydantic 参数模型和 Python 函数注册；未来工具来源可以使用同一注册接口。当前上下文仅保存在内存中。
