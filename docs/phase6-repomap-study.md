# 第六阶段第一步：轻量级 RepoMap

## 1. RepoMap 有什么用？

`list_files` 只能列出文件名，`read_file` 返回一个文件的文本。Coding Agent 若想知道“哪个文件定义了 `Parser.parse`”，逐个读文件既慢又占上下文。RepoMap 先给出仓库的目录、类、函数和方法位置，让模型决定是否再读取具体文件。本版只支持 Python，且需要显式注册工具；Agent 不会每轮自动注入仓库地图。

## 2. AST 是什么？

`ast` 是 Python 标准库的抽象语法树解析器。它把源码变成 `ClassDef`、`FunctionDef`、`AsyncFunctionDef` 等节点。节点有 `name` 和 `lineno`。例如：

```python
import ast

tree = ast.parse("def greet(): return 'hello'")
function = tree.body[0]
print(function.name, function.lineno)  # greet 1
```

RepoMap 只输出名称、类型和行号，不输出 `return` 后的字符串或其他源码内容。AST 无法可靠分析动态导入、运行时生成的定义或跨文件调用关系。

## 3. 三个文件的扫描例子

假设工作目录有以下三个 Python 文件：

```text
project/
  app.py       # def main(): ...
  models.py    # class User: def save(self): ...
  util.py      # def normalize(text): ...
```

```python
from xcode_agent import RepoMap

mapper = RepoMap("project", max_files=20, max_file_bytes=50_000,
                 max_output_chars=2_000)
result = mapper.scan(".")
print(result.text)
print(result.files[0].symbols[0].qualified_name)
```

文本结果类似：

```text
Python repository map (names and locations only):
app.py
  function main @ 1
models.py
  class User @ 1
  method User.save @ 2
util.py
  function normalize @ 1
```

`result.files` 是结构化结果；每个文件带路径和符号列表。`result.issues` 记录无法解析、过大或不可读的文件。`result.truncated` 表示文件数或文本预算使结果不完整。

## 4. 输入、处理与输出

| 层 | 输入 | 处理 | 输出 |
| --- | --- | --- | --- |
| `RepoMap(...)` | 工作目录和三个上限 | 检查目录、限制值 | 可复用的扫描器 |
| `scan(path=".")` | 工作目录内的相对目录 | 验证路径、遍历 `.py`、逐文件解析 | `RepoMapResult` |
| `_read_file(file)` | 一个候选 `.py` 文件 | 检查链接和字节数、UTF-8 解码、`ast.parse` | `list[RepoSymbol]` |
| `_Definitions` | AST 节点 | 递归访问类、函数、方法 | 名称、限定名、类型、行号 |
| `_render(files)` | 文件和符号列表 | 按字符预算保留完整行 | `(text, truncated)` |
| `register_repo_map_tool(...)` | `ToolRegistry`、工作目录、上限 | 注册只读 `repo_map` 工具 | `RepoMap` 实例 |

关键位置：[`repo_map.py`](../xcode_agent/repo_map.py) 第 18–39 行定义 Pydantic 数据；第 43–66 行访问 AST；第 69–83 行配置扫描器；第 86–103 行验证相对路径；第 106–126 行读取与解析；第 128–150 行控制文本预算；第 152–181 行遍历仓库；第 189–201 行注册工具。

调用链：模型返回 `repo_map` Tool Call → [`agent.py`](../xcode_agent/agent.py) 第 131 行起的 `Agent.run` 循环 → [`tools.py`](../xcode_agent/tools.py) 第 96 行 `ToolRegistry.execute` 校验参数 → `show(path)` → `RepoMap.scan` → 工具结果作为 `tool` 消息返回模型 → 模型生成最终答复。[`context.py`](../xcode_agent/context.py) 第 23–26 行维护消息和工具组；[`multi_agent.py`](../xcode_agent/multi_agent.py) 第 138 行创建子 Agent 时沿用注册表机制，因此每个子 Agent 都能查询同一只读地图。Trace 延用 Agent 现有 `tool_start` / `tool_end` 事件，不记录源码内容。

显式启用方式：

```python
from xcode_agent import Agent, ToolRegistry, register_repo_map_tool

registry = ToolRegistry()
register_repo_map_tool(registry, "project")
agent = Agent(llm_client, registry)  # llm_client 使用现有 ChatClient 接口
answer = agent.run("这个项目有哪些主要函数？")
```

未调用 `register_repo_map_tool` 时，模型看不到 `repo_map`。它也不能选择任意工作目录；根目录由应用代码指定。查询参数只能是该根目录内的相对目录。

## 5. 与 Aider 实现的对照

本研究查看了本地参考源码 `D:\workspace\Agent\aider-source\aider-main`。Aider 的 `aider/repomap.py` 第 42 行定义 `RepoMap`，第 103 行 `get_repo_map` 组织入口；第 279–363 行 `get_tags_raw` 使用 Tree-sitter 查询提取定义与引用，并有 Pygments 引用回退。第 365–574 行 `get_ranked_tags` 根据文件、标识符提及和引用关系建立图，用 PageRank 排名；第 629–705 行依据 token 预算寻找可容纳的输出。第 233–264 行缓存标签，第 787 行有目录扫描函数。`aider/coders/base_coder.py` 第 709–760 行取得 RepoMap 并放入模型消息。相关测试在 `tests/basic/test_repomap.py`，包括基本文件、标识符、聊天文件排除和样例仓库。

我们的版本只用 `ast` 获取 Python 定义，不提取引用、不建立图、不做 PageRank、无多语言 Tree-sitter 支持，也不按真实模型 token 计数。输出使用**字符预算**。Aider 的优先级和自动上下文集成不适用于本次“按需只读查询”的范围。大型仓库可能因 `max_files` 截断；单个文件的语法错误只进入 `issues`，其他文件仍继续处理。

## 6. 预算与安全边界

- 默认最多扫描 200 个候选 Python 文件，单文件最多 100,000 字节，文本最多 8,000 个字符。上限可在构造器中调整；字符数不等于 token 数。
- 跳过 `.git`、虚拟环境、缓存、构建目录；显式查询这些目录也会被拒绝。拒绝绝对路径、`..` 和符号链接目录；符号链接文件不解析。
- 跳过常见配置/凭据 Python 文件，如 `secrets.py`、`settings.py`、`config.py`。源码必须被读取才能交给 AST 解析，但工具结果只含路径、符号名称和行号；不返回源码或字面量。这是最小保护，**敏感信息若写入文件名、类名或函数名仍可能出现在地图中**。使用者应只对可信仓库开放工具。
- 输出预算满时不截断单行，但后续文件或符号可能省略；结构化 `truncated` 标志提示结果不完整。解析失败原因用固定短代码，不向模型暴露源代码或完整异常消息。
- 此工具只读，没有代码编辑、Shell 执行或 Git 提交能力。

## 7. 面试题与参考答案

1. **为什么 RepoMap 比逐个 `read_file` 更适合寻找入口？** 先给出符号索引，模型可定位相关文件后再精读，减少无关文本进入上下文。
2. **AST 能查到调用关系吗？** AST 能识别语法上的调用节点，但静态调用目标可能动态绑定；本版只提取定义，不做跨文件引用分析。
3. **如何避免一个坏文件使整张地图失败？** 每个文件单独解析；失败转换为 `RepoIssue`，扫描器继续处理其他文件。
4. **为什么字符预算不等于 token 预算？** 分词器会把不同语言和符号分成不同数量的 token，同样字符数可对应不同 token 数。本版只保证输出字符上限。
5. **多 Agent 使用 RepoMap 会出现什么共享问题？** 子 Agent 的工具注册表可分别克隆，但查询指向同一工作目录；工具只读，因此不会互相修改仓库。Trace 仍按各 Agent 的 ID 记录调用。
