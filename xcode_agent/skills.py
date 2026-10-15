"""Discover local SKILL.md files and load their guidance on demand."""
from pathlib import Path
import re

from pydantic import BaseModel


class SkillError(ValueError):
    """A skill is invalid, unsafe, unavailable, or too large."""


class SkillInfo(BaseModel):
    name: str
    description: str
    path: Path


class SkillRegistry:
    MAX_FILE_BYTES = 64 * 1024
    MAX_FRONTMATTER_BYTES = 4096
    _NAME = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,63}\Z")

    def __init__(self, root: str | Path):
        if Path(root).is_symlink():
            raise SkillError("Skills root must not be a symbolic link")
        try:
            self.root = Path(root).resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise SkillError("Skills root does not exist or cannot be resolved") from exc
        if not self.root.is_dir():
            raise SkillError("Skills root must be a directory")
        self._skills: dict[str, SkillInfo] = {}
        self.scan()

    def _safe_file(self, folder: Path) -> Path:
        path = folder / "SKILL.md"
        if folder.is_symlink() or path.is_symlink():
            raise SkillError(f"Symbolic links are not allowed: {folder.name}")
        try:
            if not path.resolve(strict=True).is_relative_to(self.root):
                raise SkillError(f"Skill path escapes root: {folder.name}")
        except (OSError, RuntimeError) as exc:
            raise SkillError(f"Could not resolve skill path: {folder.name}") from exc
        if not path.is_file():
            raise SkillError(f"Missing SKILL.md: {folder.name}")
        try:
            size = path.stat().st_size
        except OSError as exc:
            raise SkillError(f"Could not stat SKILL.md: {folder.name}") from exc
        if size > self.MAX_FILE_BYTES:
            raise SkillError(f"SKILL.md exceeds {self.MAX_FILE_BYTES} bytes: {folder.name}")
        return path

    @classmethod
    def _frontmatter(cls, raw: bytes, folder_name: str) -> tuple[str, str]:
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeError as exc:
            raise SkillError(f"SKILL.md must be UTF-8: {folder_name}") from exc
        lines = text.splitlines()
        if not lines or lines[0] != "---":
            raise SkillError(f"Missing YAML frontmatter: {folder_name}")
        try:
            end = lines.index("---", 1)
        except ValueError as exc:
            raise SkillError(f"Unclosed YAML frontmatter: {folder_name}") from exc
        fields: dict[str, str] = {}
        for line in lines[1:end]:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_-]*):\s*(.*)", line)
            if match is None:
                raise SkillError(f"Invalid YAML frontmatter: {folder_name}")
            key, value = match.groups()
            if key in fields:
                raise SkillError(f"Duplicate YAML field {key}: {folder_name}")
            if value.startswith('"') and value.endswith('"'):
                import json
                try:
                    value = json.loads(value)
                except ValueError as exc:
                    raise SkillError(f"Invalid YAML string: {folder_name}") from exc
            elif value.startswith("'") and value.endswith("'"):
                value = value[1:-1].replace("''", "'")
            elif value.startswith(('"', "'")) or value.endswith(('"', "'")):
                raise SkillError(f"Invalid YAML string: {folder_name}")
            elif value.startswith(("[", "{", "|", ">", "!", "&", "*")):
                raise SkillError(f"Unsupported YAML value: {folder_name}")
            fields[key] = value
        name, description = fields.get("name", ""), fields.get("description", "")
        if not isinstance(name, str) or not cls._NAME.fullmatch(name):
            raise SkillError(f"Invalid skill name: {folder_name}")
        if not isinstance(description, str) or not description.strip() or len(description) > 500:
            raise SkillError(f"Invalid skill description: {folder_name}")
        return name, description

    def scan(self) -> list[SkillInfo]:
        """Refresh metadata only; the instruction body is read by load()."""
        discovered: dict[str, SkillInfo] = {}
        for folder in sorted(self.root.iterdir()):
            if not folder.is_dir() and not folder.is_symlink():
                continue
            path = self._safe_file(folder)
            try:
                with path.open("rb") as handle:
                    prefix = handle.read(self.MAX_FRONTMATTER_BYTES + 1)
            except OSError as exc:
                raise SkillError(f"Could not read SKILL.md: {folder.name}") from exc
            first = prefix.find(b"\n---", 3)
            if first < 0 or first + 4 > self.MAX_FRONTMATTER_BYTES:
                raise SkillError(f"Frontmatter exceeds limit or is unclosed: {folder.name}")
            name, description = self._frontmatter(prefix[:first + 4], folder.name)
            if name in discovered:
                raise SkillError(f"Duplicate skill name: {name}")
            discovered[name] = SkillInfo(name=name, description=description, path=path)
        self._skills = discovered
        return self.list()

    def list(self) -> list[SkillInfo]:
        return [info.model_copy(deep=True) for info in self._skills.values()]

    def load(self, name: str) -> str:
        """Return full SKILL.md text after rechecking the path and size."""
        if not isinstance(name, str) or not self._NAME.fullmatch(name):
            raise SkillError("Invalid skill name")
        info = self._skills.get(name)
        if info is None:
            raise SkillError(f"Unknown skill: {name}")
        path = self._safe_file(info.path.parent)
        try:
            with path.open("rb") as handle:
                raw = handle.read(self.MAX_FILE_BYTES + 1)
        except OSError as exc:
            raise SkillError(f"Could not read SKILL.md: {name}") from exc
        if len(raw) > self.MAX_FILE_BYTES:
            raise SkillError(f"SKILL.md exceeds {self.MAX_FILE_BYTES} bytes: {name}")
        parsed_name, _ = self._frontmatter(raw[:self.MAX_FRONTMATTER_BYTES], path.parent.name)
        if parsed_name != name:
            raise SkillError(f"Skill name changed; rescan required: {name}")
        try:
            return raw.decode("utf-8-sig")
        except UnicodeError as exc:
            raise SkillError(f"SKILL.md must be UTF-8: {name}") from exc
