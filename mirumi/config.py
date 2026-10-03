"""Small, environment-driven configuration for local/offline operation."""

from dataclasses import dataclass
import os
from pathlib import Path


def _integer(name, default, minimum, maximum):
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


@dataclass(frozen=True)
class Settings:
    root: Path
    llama_cli: Path
    model_path: Path
    context_size: int = 2048
    max_new_tokens: int = 180
    threads: int = 4
    timeout_seconds: int = 180
    max_history_chars: int = 1800

    @classmethod
    def from_environment(cls, project_root):
        root = Path(os.environ.get("MIRUMI_HOME", project_root)).expanduser()
        llama_cli = Path(
            os.environ.get(
                "MIRUMI_LLAMA_CLI",
                Path.home() / "llama.cpp" / "build" / "bin" / "llama-cli",
            )
        ).expanduser()
        model_path = Path(
            os.environ.get(
                "MIRUMI_MODEL",
                Path.home() / "models" / "Qwen3-1.7B-Q4_K_M.gguf",
            )
        ).expanduser()
        cpu_count = os.cpu_count() or 4
        return cls(
            root=root,
            llama_cli=llama_cli,
            model_path=model_path,
            context_size=_integer("MIRUMI_CONTEXT_SIZE", 2048, 1024, 8192),
            max_new_tokens=_integer("MIRUMI_MAX_TOKENS", 180, 32, 1024),
            threads=_integer("MIRUMI_THREADS", min(4, cpu_count), 1, 32),
            timeout_seconds=_integer("MIRUMI_TIMEOUT_SECONDS", 180, 10, 1800),
            max_history_chars=_integer(
                "MIRUMI_MAX_HISTORY_CHARS", 1800, 300, 12000
            ),
        )

    @property
    def memory_dir(self):
        return self.root / "memory"

    @property
    def conversations_dir(self):
        return self.memory_dir / "conversations"

    def prepare_runtime_directories(self):
        self.conversations_dir.mkdir(parents=True, exist_ok=True)
        (self.memory_dir / "self_development").mkdir(parents=True, exist_ok=True)