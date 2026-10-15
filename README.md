# xcode-agent

一个使用原生 Python 实现的 Agent 项目。支持 OpenAI 兼容 Chat Completions、连续会话、多轮及并列工具调用、受限只读文件工具、Multi-Agent DAG、可选 SQLite FTS5 长期记忆、本地 Skills 和可信本地 stdio MCP 工具。

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
python examples/phase3_benchmark.py
python examples/mcp_demo_client.py
python -m pytest -q
```

演示与离线评测都不需要 API Key 或网络。第三阶段评测结果写入 `reports/phase3/benchmark.json` 和 `reports/phase3/benchmark.md`，仅代表固定 Mock 工作负载。真实 API 冒烟测试须手动运行 `python examples/phase3_live_smoke.py --live`；没有配置 Key 与模型时不会发送请求。示例文件是 `examples/demo.txt`。模块与消息流程见 [第一阶段学习笔记](docs/phase1-study.md)、[第二阶段学习笔记](docs/phase2-study.md)、[Trace 学习笔记](docs/phase3-trace-study.md)、[可靠性学习笔记](docs/phase3-reliability-study.md) 和 [评测学习笔记](docs/phase3-benchmark-study.md)。

## 可选长期记忆

第四阶段增加基于标准库 SQLite FTS5 的显式记忆存储。默认关闭，不会自动保存对话。调用方可用 `MemoryStore("memory.sqlite3").save("project", "文本")` 保存，再用 `Agent(..., memory=store, memory_namespace="project")` 在运行时按关键词检索。多 Agent 默认各用自身 ID 隔离 namespace，只有配置 `AgentFactory(..., shared_memory_namespace="project")` 才共享。接口和中文分词限制见 [长期记忆学习笔记](docs/phase4-memory-study.md)。

## 可选本地 Skills

将每个 Skill 放在独立子目录的 `SKILL.md` 中，并在代码中创建 `SkillRegistry(skills_root)`，再传给 `Agent(..., skills=registry)` 或 `AgentFactory(..., skills=registry)`。默认关闭。模型首先只看到名称和描述；调用内置 `load_skill` 后才在本轮执行中临时获得全文。Skill 不能自动安装、运行脚本或获取额外工具权限。格式和完整示例见 [Skills 学习笔记](docs/phase5-skills-study.md)。

## 可选本地 MCP 工具

`MCPServerConfig` 中显式指定可信 Server 的命令、参数和 `allowed_tools`。在 `with MCPClient(config) as client:` 内调用 `client.register_tools(registry)`，然后运行 Agent；离开 `with` 时关闭 ClientSession 和 stdio 子进程。示例 `python examples/mcp_demo_client.py` 使用本地 FastMCP 的只读 `add`、`echo` 工具和 Mock LLM，无需 API Key。工具名称按 `mcp__server_name__tool_name` 隔离；MCP 调用不会自动重试。详见 [MCP 学习笔记](docs/phase5-mcp-study.md)。

## 扩展接口

`Agent` 仅依赖具有 `chat(messages, tools) -> LLMResponse` 方法的客户端及 `ToolRegistry`。`AgentFactory` 通过客户端工厂创建独立子 Agent；`AgentHandle.dispatch()` 接收 `TaskNode` 列表并按输入顺序返回 `SubAgentResult`，`dispatch_and_summarize()` 可将结果送回主 Agent 推理。工具通过 Pydantic 参数模型和 Python 函数注册；未来工具来源可以使用同一注册接口。会话上下文仅保存在内存中。
