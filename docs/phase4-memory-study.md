# 第四阶段第一步：SQLite FTS5 长期记忆

## 1. 本阶段做了什么？

短期上下文 `ContextManager` 保存当前 Agent 会话的 user、assistant、tool 消息，并按完整用户轮次裁剪。长期记忆 `MemoryStore` 则把**调用方主动选择保存的文本**放入 SQLite 文件。Agent 可以在一次 `run()` 开始时检索相关文本，把有限长度的结果临时放进 LLM 请求。本阶段不自动保存对话、不做摘要，也没有 Embedding 或向量搜索。

只有传入 `memory=MemoryStore(...)` 才会开启检索；原来的 `Agent(llm, tools)` 完全保持原样。

## 2. 模块、函数的输入与输出

| 代码位置 | 职责 | 输入 | 输出 |
| --- | --- | --- | --- |
| `xcode_agent/memory.py:12` `MemoryItem` | 一条记忆的数据结构 | ID、namespace、内容、时间、可选分数 | Pydantic 对象 |
| `xcode_agent/memory.py:25` `MemoryStore(path)` | 打开 SQLite 文件并创建表、FTS5 索引和触发器 | 文件路径；默认 `:memory:` | 可跨线程共用的存储对象 |
| `xcode_agent/memory.py:95` `save(namespace, content, item_id=None)` | 新增或按相同 namespace 和 ID 更新 | namespace、文本、可选 ID | `MemoryItem`；更新时保留创建时间 |
| `xcode_agent/memory.py:117` `search(namespace, query, top_k=5)` | 关键词检索 | namespace、查询文本、条数上限 | 按 BM25 排序的 `MemoryItem` 列表，带 `score` |
| `xcode_agent/memory.py:139` `delete(namespace, item_id)` | 删除一条记忆 | namespace、ID | 是否实际删到记录的 `bool` |
| `xcode_agent/agent.py:62` `_memory_message(query)` | 检索并按字符预算整理参考文本 | 当前用户输入 | 临时 user 消息或 `None` |
| `xcode_agent/agent.py:83` `_with_memory(messages, memory_message)` | 将参考文本插入本次请求的消息副本 | 当前历史副本、记忆消息 | 发送给 LLM 的消息列表 |
| `xcode_agent/agent.py:89` `run(...)` | 原有 Agent 循环，启动时可选检索一次 | 用户输入、原有取消和超时参数 | 最终回答；不自动写入记忆 |
| `xcode_agent/multi_agent.py:94` `AgentFactory(...)` | 创建子 Agent 并配置 namespace 策略 | 原有参数、可选存储和共享 namespace | 工厂对象 |
| `xcode_agent/context.py:20` `messages` | 给 Agent 提供会话历史副本 | 无 | 消息字典列表；不包含临时记忆消息 |
| `xcode_agent/tracing.py:38` `record(...)` | 现有运行追踪 | 运行 ID、事件等 | TraceEvent；不记录记忆正文 |

## 3. 保存、更新和删除为何能保持索引一致？

主表 `memories` 保存 namespace、ID、正文和时间；`memories_fts` 是 SQLite FTS5 的外部内容索引（`xcode_agent/memory.py:52`）。`AFTER INSERT`、`AFTER DELETE` 和 `AFTER UPDATE OF content` 触发器分别写入、移除或替换索引项（`memory.py:55`、`:58`、`:62`）。`save()` 和 `delete()` 用 SQLite 事务执行主表操作，触发器与主表变更在同一事务内完成。更新用相同 `(namespace, item_id)`，不会把另一 namespace 的同名 ID 改掉。

```python
from xcode_agent import MemoryStore

with MemoryStore(":memory:") as store:
    item = store.save("demo", "alpha launch", item_id="note-1")
    assert store.search("demo", "alpha")[0].id == "note-1"
    store.save("demo", "beta release", item_id="note-1")
    assert store.search("demo", "alpha") == []
    assert store.search("demo", "beta")[0].id == "note-1"
    store.delete("demo", "note-1")
    assert store.search("demo", "beta") == []
```

默认 `:memory:` 数据库只在当前进程和连接存活时存在。传入文件路径，例如 `MemoryStore("agent-memory.sqlite3")`，重开后仍可检索。存储对象用锁保护共用连接；SQLite 文件本身仍受 SQLite 的跨进程锁和磁盘限制。

## 4. 检索如何工作？

`search()` 从查询提取普通 Unicode 字词，去重后最多使用前 16 个，组成安全的字面量 OR 查询。namespace 作为 SQL 参数过滤，结果按 `bm25(memories_fts)` **升序**返回，`top_k` 限制条数。BM25 分数越小，当前 FTS5 排序越靠前；这不是“相似度百分比”。空白或全标点查询返回空列表。公开接口还限制查询最多 1000 字符、`top_k` 为 1～100、单条正文最多 100000 字符。

FTS5 默认 `unicode61` 分词器不做中文词语切分。测试中，保存 `北京天气` 后查询 `北京` **不能**匹配；保存 `北京 天气` 后查询 `北京` 才能匹配。它只做词项全文检索，**不是语义检索**：同义词、相关概念和中文子词不会自动召回。namespace 能隔离返回的记录，但当前单个 FTS 索引的 BM25 语料统计覆盖所有 namespace，分数可能受其他 namespace 的记录数量影响。

## 5. 从保存到注入 LLM：完整离线例子

```python
from xcode_agent import Agent, LLMResponse, MemoryStore, ToolRegistry

class FakeLLM:
    def chat(self, messages, tools):
        for message in messages:
            if "Retrieved saved notes" in (message.get("content") or ""):
                print(message["content"])  # 例子中是固定非敏感文本
        return LLMResponse(content="已参考保存的日期。")

with MemoryStore(":memory:") as store:
    store.save("project-a", "launch date is Friday", item_id="schedule")
    agent = Agent(FakeLLM(), ToolRegistry(), memory=store,
                  memory_namespace="project-a", memory_top_k=3,
                  memory_char_budget=200)
    answer = agent.run("What is the launch date?")
    print(answer)
    print([m["role"] for m in agent.context.messages])  # ['user', 'assistant']
```

执行顺序是：调用方显式 `save()` → `Agent.run()` 用当前提问做一次 `search()` → `_memory_message()` 取排名靠前的结果并裁到 **包含标题在内最多 200 个 Python 字符** → `_with_memory()` 把它作为低优先级的参考 user 消息插入本次 LLM 请求 → 模型回答。参考消息不写回 `ContextManager`，下一轮会按新问题重新检索。若预算太小，可能不注入任何记忆。较长问题只用前 1000 个字符构造检索词，但原问题完整发给 LLM。

## 6. 多 Agent 的 namespace 规则

```python
store = MemoryStore("agent-memory.sqlite3")
factory = AgentFactory(make_llm, tools, memory=store)
# 每个创建出的 Agent 默认使用自身 agent_id 作为 namespace。

shared = AgentFactory(make_llm, tools, memory=store,
                      shared_memory_namespace="team-project-a")
# 只有这里明确指定后，子 Agent 才共享 team-project-a。
```

`AgentFactory` 可共用一个线程安全的 `MemoryStore` 实例，但不因此共享全部记忆。没有显式共享 namespace 时，每个 Agent 只检索以自身 ID 命名的 namespace。调用方要提前把记忆保存在目标 namespace；框架不会自动把任务结果或对话写入数据库。

## 7. 异常、测试与限制

参数无效会抛 `ValueError`；存储已关闭或数据库无法打开会抛 `MemoryStoreError`。启用记忆后，检索失败会让本次 Agent 运行报错并写入原有 `agent_error` Trace；错误正文和记忆文本不会写入 Trace。默认关闭时没有额外数据库访问。

本次运行 `python -m pytest -q`：**76 passed，1 skipped**。新增测试覆盖索引一致性、文件持久化、namespace 隔离、BM25 与 top_k、中文局限、并发共用存储、字符预算、异常、默认 API 兼容和显式共享配置。跳过的是原有 Windows 符号链接用例。

仍需注意：SQLite 数据库未加密；注入的外部文本虽然标为参考数据，模型仍可能受恶意内容影响；当前没有语义检索、中文专用分词、自动摘要、自动保存或记忆清理策略。
