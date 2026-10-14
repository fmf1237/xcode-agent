import json

import pytest

from xcode_agent.file_tools import FileTools, MAX_FILE_BYTES, register_file_tools
from xcode_agent.tools import ToolRegistry


def test_file_tools_inside_workspace(tmp_path):
    (tmp_path / "note.txt").write_text("hello", encoding="utf-8")
    tools = FileTools(tmp_path)
    assert tools.list_files(".") == ["note.txt"]
    assert tools.read_file("note.txt") == "hello"


@pytest.mark.parametrize("path", ["../outside.txt", "/etc/passwd", "C:\\Windows\\win.ini"])
def test_reject_path_escape(tmp_path, path):
    with pytest.raises(ValueError):
        FileTools(tmp_path).read_file(path)


def test_reject_symlink_outside_workspace(tmp_path):
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    link = tmp_path / "link.txt"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks unavailable")
    with pytest.raises(ValueError, match="escapes"):
        FileTools(tmp_path).read_file("link.txt")
    assert "link.txt" not in FileTools(tmp_path).list_files(".")


def test_file_size_limit(tmp_path):
    (tmp_path / "large.txt").write_bytes(b"x" * (MAX_FILE_BYTES + 1))
    with pytest.raises(ValueError, match="exceeds"):
        FileTools(tmp_path).read_file("large.txt")


def test_registry_schema_and_validation(tmp_path):
    (tmp_path / "a.txt").write_text("A", encoding="utf-8")
    registry = ToolRegistry()
    register_file_tools(registry, tmp_path)
    assert {entry["function"]["name"] for entry in registry.schemas()} == {"list_files", "read_file"}
    assert registry.execute("read_file", json.dumps({"path": "a.txt"})).content == "A"
    assert not registry.execute("read_file", json.dumps({"path": "a.txt", "extra": 1})).ok
