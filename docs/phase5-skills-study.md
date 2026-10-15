# 第五阶段第一步：Skills 动态发现与按需加载

## 1. Skills 与 Tools 的区别

**Skill** 是保存在本地 `SKILL.md` 的操作说明，例如“审查代码时先检查输入校验”。它是给模型阅读的外部材料。**Tool** 是实际可调用的 Python 函数，例如 `read_file(path)`，由 `ToolRegistry` 校验参数并执行。Skill 文字不能注册工具，也不能赋予模型新的文件、网络或代码执行权限。

默认情况下 Agent 不启用 Skills。配置 `SkillRegistry` 后，第一轮只发送 Skill 的名称和简短说明。模型认为需要时，调用受控的 `load_skill`，Agent 才读取全文并在后续本轮 LLM 请求中临时加入。这样无关 Skill 不占据上下文窗口，也避免把全文永久写入会话历史。

## 2. SKILL.md 文件结构

每个 Skill 位于根目录下的一个子目录：

```text
skills/
└── code-review/
    └── SKILL.md
```

```markdown
---
name: code-review
description: Review Python changes for correctness
---
# Review steps
Check input validation and error handling first.
```

当前轻量解析器支持 YAML Frontmatter 中的**单行字符串**：无引号、双引号或单引号；`name` 和 `description` 必填。复杂 YAML 对象、列表和多行块字符串不在本版支持范围内。Skill 名称限英文字母开头，后面可含数字、下划线和连字符。文件必须为 UTF-8，单文件最多 64 KiB，Frontmatter 最多 4 KiB。无效格式、重复名称和过大文件会抛出 `SkillError`。

## 3. 关键类和函数：输入与输出

| 代码位置 | 输入 | 输出与职责 |
|---|---|---|
| [skills.py:12](../xcode_agent/skills.py) `SkillInfo` | name、description、path | Skill 的 Pydantic 元数据；不含全文 |
| [skills.py:23](../xcode_agent/skills.py) `SkillRegistry(root)` | 本地 Skills 根目录 | 完成初次扫描的注册器 |
| [skills.py:35](../xcode_agent/skills.py) `_safe_file(folder)` | 子目录路径 | 验证路径和大小后的 `SKILL.md` 路径；拒绝符号链接与越界 |
| [skills.py:55](../xcode_agent/skills.py) `_frontmatter(raw, folder_name)` | 文件开头字节 | `(name, description)`；格式错误抛 `SkillError` |
| [skills.py:97](../xcode_agent/skills.py) `scan()` | 无；重新扫描根目录 | `list[SkillInfo]`，刷新可用 Skill |
| [skills.py:119](../xcode_agent/skills.py) `list()` | 无 | 所有 Skill 的元数据副本 |
| [skills.py:122](../xcode_agent/skills.py) `load(name)` | 已发现的 Skill 名称 | 完整 `SKILL.md` 字符串；读取前再次检查路径和大小 |
| [agent.py:97](../xcode_agent/agent.py) `_with_skills(messages, catalog, loaded)` | 当前消息副本、目录说明、已加载内容 | 只影响本次请求的消息列表 |
| [agent.py:107](../xcode_agent/agent.py) `_load_skill(raw_arguments, available, loaded)` | 模型的 JSON 参数、允许的名称、当前已加载内容 | 标准 `ToolResult`；历史中只保存加载状态 |
| [agent.py:131](../xcode_agent/agent.py) `run(user_input, ...)` | 用户输入及既有控制参数 | 最终答案；每次运行维护独立的已加载 Skill 集合 |
| [multi_agent.py:138](../xcode_agent/multi_agent.py) `AgentFactory.create()` | Agent 角色和父 Agent | 子 Agent；可使用同一注册器，但加载状态和会话独立 |

`ToolRegistry.schemas()` 与 `execute()` 位于 [tools.py:49](../xcode_agent/tools.py) 和 [tools.py:56](../xcode_agent/tools.py)。启用 Skills 时，Agent 在普通工具 Schema 之外临时提供 `load_skill` Schema；它由 Agent 内部处理，不会注册一个可执行任意代码的普通工具。`ContextManager.trim()` 位于 [context.py:26](../xcode_agent/context.py)，仍按完整用户轮次裁剪 `assistant.tool_calls` 和对应 `tool` 消息。

## 4. 一次完整的加载过程

下面示例只使用假模型，不需要 API Key 或网络：

```python
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from xcode_agent import Agent, LLMResponse, SkillRegistry, ToolCall, ToolRegistry

class FakeLLM:
    def __init__(self):
        self.calls = 0

    def chat(self, messages, tools):
        self.calls += 1
        print("LLM request", self.calls, [m["role"] for m in messages])
        if self.calls == 1:
            assert "Check input validation" not in str(messages)
            return LLMResponse(tool_calls=[ToolCall(
                id="load-1", name="load_skill",
                arguments=json.dumps({"name": "code-review"})
            )])
        assert "Check input validation" in str(messages)
        return LLMResponse(content="先检查输入校验。")

with TemporaryDirectory() as temporary:
    folder = Path(temporary) / "code-review"
    folder.mkdir()
    (folder / "SKILL.md").write_text(
        "---\nname: code-review\ndescription: Review Python code\n---\n"
        "Check input validation first.\n", encoding="utf-8")
    agent = Agent(FakeLLM(), ToolRegistry(), skills=SkillRegistry(temporary))
    print(agent.run("请审查这段代码"))
    print(agent.context.messages)  # 有 load_skill 调用与简短结果，没有全文
```

步骤：① `scan()` 仅读取文件开头的元数据；② Agent 发出名称、描述和 `load_skill` Schema；③ 模型用名称选择 Skill；④ `_load_skill()` 校验名称、数量与字符预算，再调用 `load()` 读取全文；⑤ 后续 LLM 请求临时包含全文；⑥ Tool Calling 历史中只写入 `{"name": "code-review", "status": "loaded"}` 等状态。下次 `run()` 会重新开始选择，不继承已加载全文。

## 5. 与 Multi-Agent 和长期记忆的关系

可把同一个 `SkillRegistry` 传给 `AgentFactory(..., skills=registry)`。每个子 Agent 只共享可用 Skill 目录，不共享“本轮已经加载了什么”的状态；它们的 `ContextManager` 仍独立。长期记忆由 `MemoryStore` 提供，按 namespace 检索；Skill 是指导如何完成任务的文档。两者都使用临时消息副本，不会改变原始用户消息或自动保存对话。Multi-Agent 记忆 namespace 仍须显式配置才能共享。

## 6. 注入的安全边界与限制

- 文件路径来自已扫描的注册表，不从模型参数拼接路径；名称校验拒绝 `../`。扫描与加载均拒绝符号链接，检查文件是否处于根目录内，并限制大小。
- Skill 内容属于外部材料，以低于系统消息优先级的临时 `user` 角色消息提供；系统规则仍应优先。模型遵守优先级取决于模型行为，不能仅靠提示文本保证安全。
- `load_skill` 只返回状态，不返回全文到 `tool` 历史。Trace 记录工具名称与结果，不记录 Skill 全文。
- `max_skills` 限制展示及本轮可加载的 Skill 数量；`skill_char_budget` 限制本轮加载的全文消息总字符数。超预算时整篇拒绝，不截断可能改变意思的步骤。
- 不执行 Skill 中的脚本，不读取其它附件，不联网安装 Skill。当前解析器只支持单行字符串 Frontmatter；扫描需要显式调用 `scan()` 刷新。若不可信进程可在验证与打开之间并发替换文件，仍存在文件系统竞态，应将 Skills 目录交给可信维护者管理。

## 7. 面试题与参考答案

1. **Skill 和 Tool 的核心区别？** Skill 是供模型阅读的说明；Tool 是由程序执行的受控函数。读到 Skill 不代表得到新的工具权限。
2. **为什么采用渐进式加载？** 只给目录可减少初始请求的字符量；模型选中后再读取全文，让上下文更聚焦。
3. **如何避免路径穿越？** 模型只传名称；注册器用名称查已扫描的路径，并校验名称、真实路径、符号链接和文件大小，不把名称拼成任意读取路径。
4. **如何保证 Tool Calling 历史合法？** 模型的 `load_skill` 仍生成标准 assistant 工具调用和匹配 `tool_call_id` 的工具结果，全文只在临时请求副本中出现。
5. **多个子 Agent 是否共享已激活 Skill？** 不共享。工厂可共享只读的 Skill 注册器；每个 Agent 的每次 `run()` 自行选择并管理已加载内容。
