# 第五阶段第二步：本地 stdio MCP 工具接入

## 1. MCP、Client、Server 与 stdio

MCP（Model Context Protocol）让程序用统一协议发现并调用外部工具。**Server** 暴露工具；**Client** 启动或连接 Server，读取工具列表并发起调用。这里仅支持可信的本地 Server。`stdio` 表示两个进程通过标准输入和标准输出交换协议消息：Client 向 Server 的 stdin 写请求，从 stdout 读响应。Server 不应把普通日志写入 stdout，以免破坏协议；Server 的 stderr 在本实现中默认丢弃，避免把其自行打印的参数带入应用日志。

MCP 消息使用 JSON-RPC：请求包含方法和 ID，响应使用同一 ID 返回结果或错误。官方 SDK 的 `ClientSession` 负责初始化、请求响应配对和消息读写，项目无需手写 JSON-RPC 协议。代码依据 [官方 Python SDK v1 文档](https://github.com/modelcontextprotocol/python-sdk/blob/v1.30.0/README.md)和[官方 stdio Client 示例](https://github.com/modelcontextprotocol/python-sdk/blob/v1.30.0/examples/snippets/clients/stdio_client.py)。官方 v2 已调整高层 Client API；本项目为明确使用 `ClientSession`，将依赖限定为 `mcp>=1.30,<2`。

## 2. MCP 与普通 Tool Calling 的关系

模型看到的仍是 OpenAI 兼容函数工具 Schema。普通工具由 `ToolRegistry` 调用本地 Python 函数；MCP 工具由 `ToolRegistry` 校验参数后转交 `MCPClient`，再由 `ClientSession.call_tool()` 发送给 Server。模型只能选择已注册的工具，不能提供命令来启动任意程序。

工具名使用 `mcp__{server_name}__{tool_name}`，因此两个 Server 的 `add` 不会重名。注册时复制 Server 提供的 JSON Schema，保留 `required`、类型、范围等约束；本地使用 `jsonschema` 校验原始 JSON 参数。MCP 结果通过现有 `ToolResult` 回到 Agent。`isError`、连接异常、调用超时使用不同 `error_code`，工具失败不自动重试。

## 3. 同步 Agent 如何使用异步 ClientSession

`Agent.run()` 和 `ToolRegistry.execute()` 是同步函数，而官方 `stdio_client`、`ClientSession.initialize()`、`list_tools()`、`call_tool()` 是异步操作。`MCPClient` 为每个 Server 启动**一条专用线程和一个长期存活的事件循环**。它在同一事件循环中进入 stdio 与 Session 的异步上下文，并完成初始化和发现。同步调用使用 `asyncio.run_coroutine_threadsafe()` 把协程提交到该循环，等待结果。它不会为每次工具调用建立新的 Session。

生命周期由调用方显式管理：

```python
with MCPClient(config) as client:
    client.register_tools(registry)
    agent.run("请计算 3 + 5")
# 无论 Agent 正常返回还是抛异常，这里都会关闭 Session 和子进程。
```

当多个子 Agent 共用工具注册器时，不能让其中一个 Agent 提前关闭整个 Server；所以连接由外层 `with` 拥有。运行 `AgentHandle.dispatch()` 的代码也应置于同一个 `with` 内。`ToolRegistry.clone()` 会为共享外部 handler 使用锁，避免多个子 Agent 同时驱动同一 Session；MCPClient 自身也串行提交工具调用。

## 4. 每层输入输出与关键源码位置

| 位置 | 输入 | 输出或作用 |
|---|---|---|
| [mcp_client.py:22](../xcode_agent/mcp_client.py) `MCPServerConfig` | Server 名、可信命令、参数、`allowed_tools`、数量与超时 | 固定配置；不接受模型产生的启动命令 |
| [mcp_client.py:35](../xcode_agent/mcp_client.py) `MCPToolInfo` | 原工具名、描述、Schema | 带唯一限定名称的元数据 |
| [mcp_client.py:60](../xcode_agent/mcp_client.py) `_serve()` | 固定 Server 配置 | 在专用事件循环中保持 stdio 与 ClientSession 存活 |
| [mcp_client.py:100](../xcode_agent/mcp_client.py) `connect()` | 无 | 初始化并返回已允许的工具；失败抛 `MCPConnectionError` |
| [mcp_client.py:117](../xcode_agent/mcp_client.py) `list_tools()` | 无 | `list[MCPToolInfo]` |
| [mcp_client.py:120](../xcode_agent/mcp_client.py) `register_tools(registry)` | 现有 `ToolRegistry` | 注册已允许工具，返回限定名称列表 |
| [mcp_client.py:143](../xcode_agent/mcp_client.py) `call_tool(name, arguments)` | 原 MCP 工具名、参数字典 | 标准 `ToolResult`，含内容或结构化错误码 |
| [mcp_client.py:166](../xcode_agent/mcp_client.py) `close()` | 无 | 退出 Session、关闭 stdio 子进程和线程 |
| [tools.py:43](../xcode_agent/tools.py) `register_external(...)` | 限定名称、描述、原 JSON Schema、handler | 保存外部工具定义 |
| [tools.py:84](../xcode_agent/tools.py) `schemas()` | 无 | 给 LLM 的函数工具 Schema 列表 |
| [tools.py:96](../xcode_agent/tools.py) `execute(name, raw_arguments)` | 模型工具名与 JSON 字符串 | 参数校验后执行并返回 `ToolResult` |
| [agent.py:131](../xcode_agent/agent.py) `run(...)` | 用户输入 | 原有 Tool Calling 循环；Trace 记录工具名、耗时和状态 |

## 5. `add(3, 5)` 返回 `8` 的完整链路

先安装项目依赖，再运行：

```powershell
python -m pip install -e ".[dev]"
python examples/mcp_demo_client.py
```

[mcp_demo_server.py](../examples/mcp_demo_server.py) 用官方 `FastMCP` 定义无副作用的 `add` 和 `echo`。[mcp_demo_client.py](../examples/mcp_demo_client.py) 使用固定的 `sys.executable` 和 Server 文件路径，显式允许这两个工具。连接后获得 `mcp__demo__add`，Mock LLM 给出 `{"a":3,"b":5}`；注册器按 Server Schema 验证整数参数；ClientSession 发送 `call_tool("add", ...)`；Server 返回结构化结果；`ToolResult(ok=True, content=8)` 回到 Agent 的 `tool` 消息；下一次 Mock LLM 回答 `3 + 5 = 8`。整个示例不使用真实 LLM、API Key 或外部网络。

## 6. 错误和安全边界

- 启动、初始化或发现失败：`connect()` 抛 `MCPConnectionError`（`error_code=mcp_connection_error`），并尝试关闭线程与子进程。
- JSON 参数不符合 Server Schema：注册器返回 `tool_argument_error`，不会发送调用。
- Server 返回 `isError`：返回 `mcp_tool_error`；传输或请求异常返回 `mcp_call_error`；超时返回 `mcp_timeout`。错误反馈不包含原始异常正文。
- 超时只说明客户端停止等待；Server 端操作可能已经开始。本项目不自动重试 MCP 工具，尤其不能假定有副作用的操作安全可重发。
- `allowed_tools` 和 `max_tools` 控制暴露范围。示例工具标记为只读且无副作用；对其它 Server 的标注仍需人工核实。MCP 内容作为外部 `tool` 数据交给模型，不获得系统指令权限。
- Agent 的 Trace 不保存原始参数或结果；MCP 工具的详细结果也不会写入 Agent 的执行日志。Server 的 stderr 默认丢弃；这减少意外泄露，也意味着连接失败时只能看到结构化错误码。
- 只实现本地 stdio。不支持远程 HTTP、OAuth、动态安装或模型自行启动 Server。连接共享时，应用须保持外层 `with` 覆盖所有 Agent 调用；进程被强杀时不能保证正常清理。

## 7. 离线验证

本阶段在 Windows 上运行 `python -m pytest -q`：**113 passed, 1 skipped**。跳过的是既有文件工具的符号链接测试，原因是当前环境不能创建该符号链接。MCP 测试使用本地 FastMCP 子进程，不依赖网络或真实模型。

## 8. 面试题与参考答案

1. **MCP 和 Tool Calling 是否是同一层？** 不是。Tool Calling 是模型提出函数调用；MCP 是 Client 与 Server 之间发现和执行工具的协议。注册器把两层接起来。
2. **为什么不能每次调用都 `asyncio.run()`？** MCP 的 stdio 流和 Session 属于同一个事件循环；每次新建循环会破坏会话生命周期并反复启动进程。专用循环线程可复用同一 Session。
3. **如何防止不同 Server 的 `add` 冲突？** 暴露给模型的名称加 Server 前缀，例如 `mcp__left__add` 与 `mcp__right__add`；调用时映射回对应 Session 的原始 `add`。
4. **如何处理 MCP 超时？** 返回 `mcp_timeout`，取消客户端等待，不自动重试。超时无法证明 Server 没有执行操作。
5. **为什么需要白名单？** Server 可以暴露很多工具；应用只把明确允许的名称注册给模型。模型无法通过工具参数决定启动哪个可执行程序。
