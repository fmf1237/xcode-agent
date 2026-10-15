"""Small read-only Python repository map based on the standard-library AST."""
import ast
import os
from pathlib import Path, PurePosixPath, PureWindowsPath

from pydantic import BaseModel, ConfigDict, Field

from .tools import ToolRegistry


IGNORED_DIRS = frozenset({".git", ".hg", ".svn", ".venv", "venv", "env",
                          "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
                          "build", "dist", "node_modules", ".tox"})
SENSITIVE_FILES = frozenset({"secrets.py", "credentials.py", "settings.py",
                             "local_settings.py", "config.py"})


class RepoSymbol(BaseModel):
    name: str
    qualified_name: str
    kind: str
    line: int = Field(ge=1)
    is_async: bool = False


class RepoFile(BaseModel):
    path: str
    symbols: list[RepoSymbol] = Field(default_factory=list)


class RepoIssue(BaseModel):
    path: str
    reason: str


class RepoMapResult(BaseModel):
    files: list[RepoFile]
    issues: list[RepoIssue]
    text: str
    truncated: bool = False


class _Definitions(ast.NodeVisitor):
    def __init__(self) -> None:
        self.symbols: list[RepoSymbol] = []
        self.parents: list[tuple[str, str]] = []

    def _add(self, node: ast.AST, name: str, kind: str, is_async: bool = False) -> None:
        qualified = ".".join([parent[1] for parent in self.parents] + [name])
        self.symbols.append(RepoSymbol(name=name, qualified_name=qualified, kind=kind,
                                       line=node.lineno, is_async=is_async))
        self.parents.append((kind, name))
        for statement in node.body:
            self.visit(statement)
        self.parents.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._add(node, node.name, "class")

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        kind = "method" if self.parents and self.parents[-1][0] == "class" else "function"
        self._add(node, node.name, kind)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        kind = "method" if self.parents and self.parents[-1][0] == "class" else "function"
        self._add(node, node.name, kind, is_async=True)


class RepoMap:
    """Scan a configured workspace; never return source lines or literal values."""
    def __init__(self, workspace: str | Path, *, max_files: int = 200,
                 max_file_bytes: int = 100_000, max_output_chars: int = 8_000):
        source = Path(workspace)
        if source.is_symlink():
            raise ValueError("RepoMap workspace must not be a symlink")
        self.workspace = source.resolve(strict=True)
        if not self.workspace.is_dir():
            raise ValueError("RepoMap workspace must be a directory")
        if (not all(isinstance(value, int) and not isinstance(value, bool) and value > 0
                    for value in (max_files, max_file_bytes, max_output_chars))):
            raise ValueError("RepoMap limits must be positive integers")
        self.max_files = max_files
        self.max_file_bytes = max_file_bytes
        self.max_output_chars = max_output_chars

    def _subtree(self, path: str) -> Path:
        if not isinstance(path, str) or not path:
            raise ValueError("RepoMap path must be a relative directory")
        relative = Path(path)
        windows = PureWindowsPath(path)
        if (relative.is_absolute() or windows.is_absolute() or windows.drive or
                PurePosixPath(path).is_absolute() or ".." in relative.parts or ".." in windows.parts):
            raise ValueError("RepoMap path must stay inside workspace")
        current = self.workspace
        for component in relative.parts:
            if component.casefold() in IGNORED_DIRS:
                raise ValueError("RepoMap path is ignored")
            current = current / component
            if current.is_symlink():
                raise ValueError("RepoMap symlink directories are forbidden")
        target = current.resolve(strict=True)
        if not target.is_relative_to(self.workspace) or not target.is_dir():
            raise ValueError("RepoMap path must be a workspace directory")
        return target

    def _read_file(self, file: Path) -> list[RepoSymbol]:
        # Enumeration never follows symlinks; recheck before opening the file.
        if file.is_symlink() or not file.resolve(strict=True).is_relative_to(self.workspace):
            raise ValueError("symlink_skipped")
        if file.stat().st_size > self.max_file_bytes:
            raise ValueError("too_large")
        with file.open("rb") as stream:
            data = stream.read(self.max_file_bytes + 1)
        if len(data) > self.max_file_bytes or b"\x00" in data:
            raise ValueError("too_large_or_binary")
        try:
            source = data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError("invalid_utf8") from exc
        try:
            tree = ast.parse(source, filename=file.name)
        except (SyntaxError, ValueError, RecursionError) as exc:
            raise ValueError("parse_error") from exc
        visitor = _Definitions()
        visitor.visit(tree)
        return visitor.symbols

    def _render(self, files: list[RepoFile]) -> tuple[str, bool]:
        header = "Python repository map (names and locations only):\n"
        if len(header) > self.max_output_chars:
            return "", True
        lines = [header]
        used = len(header)
        truncated = False
        for file in files:
            line = f"{file.path}\n"
            if used + len(line) > self.max_output_chars:
                truncated = True
                continue
            lines.append(line)
            used += len(line)
            for symbol in file.symbols:
                suffix = " async" if symbol.is_async else ""
                line = f"  {symbol.kind} {symbol.qualified_name} @ {symbol.line}{suffix}\n"
                if used + len(line) > self.max_output_chars:
                    truncated = True
                    continue
                lines.append(line)
                used += len(line)
        return "".join(lines), truncated

    def scan(self, path: str = ".") -> RepoMapResult:
        """Return structure and a bounded text map for a relative subtree."""
        target = self._subtree(path)
        files: list[RepoFile] = []
        issues: list[RepoIssue] = []
        truncated = False
        for directory, dirs, names in os.walk(target, followlinks=False):
            dirs[:] = sorted(name for name in dirs
                             if name.casefold() not in IGNORED_DIRS and
                             not (Path(directory) / name).is_symlink())
            for name in sorted(names):
                if not name.endswith(".py") or name.casefold() in SENSITIVE_FILES or name.startswith(".env"):
                    continue
                file = Path(directory) / name
                relative = file.relative_to(self.workspace).as_posix()
                if len(files) + len(issues) >= self.max_files:
                    truncated = True
                    break
                try:
                    symbols = self._read_file(file)
                except (OSError, ValueError) as exc:
                    reason = str(exc) if isinstance(exc, ValueError) else "unreadable"
                    issues.append(RepoIssue(path=relative, reason=reason))
                    continue
                files.append(RepoFile(path=relative, symbols=symbols))
            if truncated:
                break
        text, output_truncated = self._render(files)
        return RepoMapResult(files=files, issues=issues, text=text,
                             truncated=truncated or output_truncated)


class RepoMapArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(default=".", description="Relative directory within the configured workspace")


def register_repo_map_tool(registry: ToolRegistry, workspace: str | Path, *,
                           max_files: int = 200, max_file_bytes: int = 100_000,
                           max_output_chars: int = 8_000) -> RepoMap:
    """Explicitly register the read-only repo_map tool; disabled otherwise."""
    mapper = RepoMap(workspace, max_files=max_files, max_file_bytes=max_file_bytes,
                     max_output_chars=max_output_chars)

    def show(path: str = ".") -> str:
        return mapper.scan(path).text

    registry.register("repo_map", "Show Python files, classes and functions in a workspace subtree",
                      RepoMapArgs, show)
    return mapper
