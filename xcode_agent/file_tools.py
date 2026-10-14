"""Read-only file tools restricted to a configured workspace."""
from pathlib import Path, PurePosixPath, PureWindowsPath

from pydantic import BaseModel, ConfigDict, Field

from .tools import ToolRegistry

MAX_FILES = 100
MAX_FILE_BYTES = 1_000_000
MAX_CONTENT_CHARS = 20_000


class PathArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(min_length=1, description="Relative path inside the configured workspace")


class FileTools:
    def __init__(self, workspace: Path):
        self.workspace = workspace.resolve(strict=True)
        if not self.workspace.is_dir():
            raise ValueError("workspace must be a directory")

    def _resolve(self, path: str) -> Path:
        relative = Path(path)
        if relative.is_absolute() or PureWindowsPath(path).is_absolute() or PureWindowsPath(path).drive or PurePosixPath(path).is_absolute():
            raise ValueError("Absolute paths are forbidden")
        if ".." in relative.parts or ".." in PureWindowsPath(path).parts:
            raise ValueError("Path traversal is forbidden")
        target = (self.workspace / relative).resolve(strict=True)
        if not target.is_relative_to(self.workspace):
            raise ValueError("Path escapes workspace")
        return target

    def list_files(self, path: str) -> list[str]:
        directory = self._resolve(path)
        if not directory.is_dir():
            raise ValueError("Path is not a directory")
        entries = sorted(directory.iterdir(), key=lambda item: item.name)
        if len(entries) > MAX_FILES:
            raise ValueError(f"Directory exceeds {MAX_FILES} entries")
        result = []
        for item in entries:
            try:
                resolved = item.resolve(strict=True)
                if resolved.is_relative_to(self.workspace):
                    result.append(item.name + ("/" if item.is_dir() else ""))
            except (OSError, RuntimeError):
                continue
        return result

    def read_file(self, path: str) -> str:
        file = self._resolve(path)
        if not file.is_file():
            raise ValueError("Path is not a regular file")
        if file.stat().st_size > MAX_FILE_BYTES:
            raise ValueError(f"File exceeds {MAX_FILE_BYTES} bytes")
        with file.open("rb") as stream:
            data = stream.read(MAX_FILE_BYTES + 1)
        if len(data) > MAX_FILE_BYTES or b"\x00" in data:
            raise ValueError("File is too large or binary")
        try:
            content = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("File is not UTF-8 text") from exc
        if len(content) > MAX_CONTENT_CHARS:
            raise ValueError(f"Content exceeds {MAX_CONTENT_CHARS} characters")
        return content


def register_file_tools(registry: ToolRegistry, workspace: Path) -> None:
    files = FileTools(workspace)
    registry.register("list_files", "List files in a workspace directory", PathArgs, files.list_files)
    registry.register("read_file", "Read a UTF-8 text file in the workspace", PathArgs, files.read_file)
