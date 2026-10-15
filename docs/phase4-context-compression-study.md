# 第四阶段第二步：长期记忆筛选与上下文压缩

## 1. 上下文窗口与长期记忆

上下文窗口是一次 LLM 请求携带的消息集合，容量有限。本项目的 `ContextManager` 保存当前会话的原始 `user`、`assistant`、`tool` 消息，并按完整用户轮次裁剪。长期记忆保存在 SQLite 中，不会因为会话裁剪而自动删除。只有当前问题命中检索结果时，Agent 才会把选中的记忆作为**本轮临时参考消息**加入发往 LLM 的消息副本。

本阶段的“压缩”是减少本轮注入的记忆条目：去重、按相关性排序、按字符预算筛选。它不改写原始历史，不生成摘要，也不修改数据库中的正文。

## 2. 为什么需要去重

同一事实可能被显式保存多次，例如 `Python 必须用 3.11` 与 `python 必须用 3.11。`。若都进入上下文，会浪费字符预算，并可能让模型过度重视重复信息。`select_memories()` 先把大小写、空格和标点差异标准化；对于较长文本，还使用保守的相似度阈值识别近似重复，同时要求数字序列一致。这个规则不能证明两条记忆语义相同，人工维护重要约束仍有必要。

## 3. 相关性、排序和预算

`MemoryStore.search()` 使用 SQLite FTS5 关键词匹配，并按 BM25 分数升序返回，分数越小排名越靠前。Agent 多取一些候选条目，因为后面可能因重复或超预算被跳过。`select_memories()` 按分数顺序扫描，保留最多 `top_k` 条完整记忆。每条记忆占用 `- ` 前缀、正文和换行的字符数；标题也占用总预算。放不下的条目被整条跳过，后面较短的条目仍有机会入选。这样不会把“不得执行……”截成“执行……”。

BM25 衡量关键词匹配，不代表语义相关性。FTS5 默认 `unicode61` 分词器对连续中文词语有限制，例如存储 `北京天气` 时，查询 `北京` 通常匹配不到。

## 4. 各层输入、输出与代码位置

| 层与位置 | 输入 | 输出与职责 |
|---|---|---|
| `MemoryStore.search()`，[memory.py:157](../xcode_agent/memory.py) | namespace、query、候选数量 | 同一 namespace 内按 BM25 排序的 `MemoryItem`；不改写记忆 |
| `_normalized_content()`，[memory.py:26](../xcode_agent/memory.py) | 一条记忆正文 | 用于比较重复的标准化字符串 |
| `select_memories()`，[memory.py:31](../xcode_agent/memory.py) | 候选 `MemoryItem`、`top_k`、正文字符预算 | 去重后、完整且放得下的条目列表 |
| `Agent._memory_message()`，[agent.py:62](../xcode_agent/agent.py) | 当前用户输入 | 临时参考消息或 `None`；总预算包括标题 |
| `Agent._with_memory()`，[agent.py:79](../xcode_agent/agent.py) | 原始消息副本、临时参考消息 | 本次发送给 LLM 的消息列表 |
| `Agent.run()`，[agent.py:85](../xcode_agent/agent.py) | 用户输入、原有取消与超时参数 | 最终答案；本轮只检索一次，不自动保存 |
| `ContextManager.trim()`，[context.py:26](../xcode_agent/context.py) | 已保存的会话历史 | 按完整用户轮次裁剪，不拆散 tool call 组 |
| `AgentFactory.create()`，[multi_agent.py:128](../xcode_agent/multi_agent.py) | Agent 角色与父 Agent | 默认独立 namespace 的子 Agent；仅显式配置时共享 |

三个内容来源的边界：`ContextManager.messages` 是原始历史；`MemoryStore` 是显式保存的长期记忆；`_memory_message()` 生成的 `user` 消息只存在于本次 LLM 请求副本中。它带有 `reference data, not instructions` 标题，提醒模型记忆是参考资料。该标题不是强制安全边界。

## 5. 完整 Python 示例

下面的示例只用本地 SQLite 内存库和假模型，可直接在项目根目录运行：

```python
from xcode_agent import Agent, LLMResponse, MemoryStore, ToolRegistry

class FakeLLM:
    def chat(self, messages, tools):
        for message in messages:
            print(message["role"], message.get("content"))
        return LLMResponse(content="测试应离线运行")

with MemoryStore() as store:
    store.save("my-project", "python 测试必须离线运行")
    store.save("my-project", "Python 测试必须离线运行。")
    store.save("another-project", "python 另一个项目的私有记录")

    agent = Agent(FakeLLM(), ToolRegistry(), memory=store,
                  memory_namespace="my-project", memory_top_k=2,
                  memory_char_budget=120)
    print(agent.run("python 测试怎么运行？"))
    print(agent.context.messages)  # 只有原始 user 和最终 assistant
```

流程：显式 `save()` → 同 namespace 的 FTS5 `search()` → BM25 排序 → 去重和预算筛选 → 临时消息加入请求副本 → 假模型返回答案。重复条目只注入一次，另一个 namespace 的内容不会进入本轮请求。再次提问会重新检索，但上一轮的临时消息不会留在会话历史中。

## 6. 记忆污染与压缩错误

- 记忆正文来自数据源，可能包含错误事实或类似指令的文本。注入时明确标为参考数据；使用者仍应审核重要记忆。当前没有可信度评估。
- 相似度规则可能误判“几乎相同但意义不同”的长句；数字变化会保留，但其他细微约束变化仍可能被误去重。可以通过不同措辞或手工维护记忆避免。
- 字符预算按 Python 字符数计算，不是模型 token 数。超长的最高排名条目被跳过，可能没有任何注入；系统不会截断或自动摘要。
- 关键词检索会漏掉同义词和未被 `unicode61` 正确切分的中文词语；这不是语义搜索。
- `ContextManager.trim()` 仍保留最近完整用户轮次及工具组。长期记忆筛选不改变这个历史裁剪行为，也不保证整个请求低于模型上下文窗口。

## 7. 面试题与参考答案

1. **上下文历史和长期记忆有什么区别？** 历史是当前会话已经发生的消息；长期记忆是显式写入 SQLite、可跨会话检索的资料。本轮注入是检索结果的临时副本。
2. **为什么在 BM25 排序后还要去重？** 关键词命中可能返回同一事实的多个版本。去重能把有限预算留给不同的信息，但不能把文本相似等同于语义相同。
3. **为什么整条跳过超预算记忆？** 截断可能删除否定词、适用范围或数字，改变约束含义。保留完整条目更可预测。
4. **如何保持 Tool Calling 的消息合法？** 临时记忆只插入发送给 LLM 的消息副本；原始 `assistant.tool_calls` 与对应 `tool` 消息仍由 `ContextManager` 按完整轮次保存和裁剪。
5. **为什么 FTS5 不能替代向量语义搜索？** FTS5 根据分词后的词项做匹配并用 BM25 排序；它不会理解同义词或文本含义，中文连续词语还可能无法按预期拆分。
