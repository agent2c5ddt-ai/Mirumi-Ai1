"""Inference adapter for a local llama.cpp command-line installation."""

import subprocess


class ModelError(RuntimeError):
    """Raised when the local model cannot produce a usable response."""


class ModelInterface:
    """Minimal interface expected by the conversation orchestrator."""

    def generate(self, system_prompt, prompt, max_tokens=None):
        raise NotImplementedError


class LlamaCppCLI(ModelInterface):
    def __init__(self, settings):
        self.settings = settings

    def generate(self, system_prompt, prompt, max_tokens=None):
        if not self.settings.llama_cli.is_file():
            raise ModelError(
                "llama-cli was not found at "
                f"{self.settings.llama_cli}. Set MIRUMI_LLAMA_CLI to its path."
            )
        if not self.settings.model_path.is_file():
            raise ModelError(
                "The configured GGUF model was not found at "
                f"{self.settings.model_path}. Set MIRUMI_MODEL to its path."
            )

        token_limit = max_tokens or self.settings.max_new_tokens
        command = [
            str(self.settings.llama_cli),
            "-m",
            str(self.settings.model_path),
            "-c",
            str(self.settings.context_size),
            "-b",
            "128",
            "-ub",
            "64",
            "-t",
            str(self.settings.threads),
            "-n",
            str(token_limit),
            "--no-show-timings",
            "--no-display-prompt",
            "-sys",
            system_prompt,
            "-p",
            prompt,
        ]

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.settings.timeout_seconds,
                check=False,
            )
        except FileNotFoundError as exc:
            raise ModelError(f"Unable to start llama.cpp: {exc}") from exc
        except subprocess.TimeoutExpired as exc:
            raise ModelError(
                "llama.cpp exceeded the configured "
                f"{self.settings.timeout_seconds}-second limit."
            ) from exc
        except OSError as exc:
            raise ModelError(f"Unable to run llama.cpp: {exc}") from exc

        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()
            if len(detail) > 1200:
                detail = detail[-1200:]
            raise ModelError(
                detail or f"llama-cli exited with status {result.returncode}."
            )

        answer = (result.stdout or "").replace("\x00", "").strip()
        if not answer:
            raise ModelError("llama.cpp returned an empty response.")
        return answer