import asyncio
import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_BINARY = Path(__file__).resolve().parent / "llm-adapter"
_DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"


class LLMAdapterClient:

    def __init__(
        self,
        binary_path: str | Path | None = None,
        config_path: str | Path | None = None,
        timeout: int = 300,
    ):
        self.binary_path = Path(binary_path) if binary_path else _DEFAULT_BINARY
        self.config_path = Path(config_path) if config_path else _DEFAULT_CONFIG
        self.timeout = timeout

        if not self.binary_path.exists():
            raise FileNotFoundError(
                f"llm-adapter binary not found at {self.binary_path}"
            )
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"llm-adapter config not found at {self.config_path}"
            )

        logger.info(
            "LLMAdapterClient initialized: binary=%s config=%s timeout=%ds",
            self.binary_path, self.config_path, self.timeout,
        )

    async def complete(
        self,
        messages: list[dict],
        model: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict:
        request: dict = {"messages": messages}
        if model:
            request["model"] = model
        if temperature is not None:
            request["temperature"] = temperature
        if max_tokens is not None:
            request["max_tokens"] = max_tokens

        request_json = json.dumps(request, ensure_ascii=False)

        msg_info = {
            "role": messages[-1]["role"] if messages else "unknown",
            "chars": len(str(messages[-1]["content"])) if messages else 0,
            "temp": temperature,
            "max_tokens": max_tokens,
        }
        logger.info(
            "→ LLM-adapter request | %s: %d chars temp=%s max_tokens=%s",
            msg_info["role"], msg_info["chars"], msg_info["temp"], msg_info["max_tokens"],
        )

        start_time = time.monotonic()
        try:
            process = await asyncio.create_subprocess_exec(
                str(self.binary_path),
                "-c", str(self.config_path),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except (OSError, PermissionError) as e:
            raise RuntimeError(
                f"Failed to launch llm-adapter binary at {self.binary_path}: {e}"
            ) from e

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(input=request_json.encode()),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            elapsed = time.monotonic() - start_time
            raise TimeoutError(
                f"llm-adapter timed out after {elapsed:.0f}s (limit={self.timeout}s)"
            )

        elapsed = time.monotonic() - start_time

        if process.returncode != 0:
            stderr_text = stderr.decode(errors="replace").strip()
            logger.error(
                "← LLM-adapter failed (exit=%d, %.1fs): %s",
                process.returncode, elapsed, stderr_text[:500],
            )
            raise RuntimeError(
                f"llm-adapter exited with code {process.returncode}: {stderr_text[:500]}"
            )

        stdout_text = stdout.decode(errors="replace").strip()
        if not stdout_text:
            raise RuntimeError(
                f"llm-adapter returned empty output (elapsed={elapsed:.1f}s)"
            )

        try:
            response = json.loads(stdout_text)
        except json.JSONDecodeError as e:
            logger.error(
                "← LLM-adapter returned invalid JSON (first 300 chars): %s",
                stdout_text[:300],
            )
            raise RuntimeError(
                f"llm-adapter returned invalid JSON: {e}"
            ) from e

        normalized = self._normalize_response(response, elapsed)
        logger.info(
            "← LLM-adapter response (%.1fs) model=%s",
            elapsed, normalized.get("model", "?"),
        )
        logger.debug("← LLM-adapter raw output (%d chars):\n%s", len(stdout_text), stdout_text)
        return normalized

    def _normalize_response(self, raw: dict, elapsed: float) -> dict:
        content = ""
        reasoning_content = None
        usage = raw.get("usage", {})
        model = raw.get("model", "")

        choices = raw.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            content = message.get("content", "")
            reasoning_content = message.get("reasoning_content")

        result = {
            "choices": [
                {
                    "message": {"content": content},
                    "finish_reason": choices[0].get("finish_reason") if choices else None,
                }
            ],
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
            "model": model,
        }

        if reasoning_content is not None:
            result["choices"][0]["message"]["reasoning_content"] = reasoning_content

        return result