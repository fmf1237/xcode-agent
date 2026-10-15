"""Trusted local stdio MCP server. Keep stdout reserved for MCP messages."""
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations


mcp = FastMCP("xcode-agent-demo")
readonly = ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False)


@mcp.tool(annotations=readonly)
def add(a: int, b: int) -> int:
    """Return the sum of two integers without side effects."""
    return a + b


@mcp.tool(annotations=readonly)
def echo(text: str) -> str:
    """Return the supplied text without side effects."""
    return text


if __name__ == "__main__":
    mcp.run(transport="stdio")
