"""HTTP client for experiment LLM calls (OpenAI-compatible chat completions)."""

from __future__ import annotations

import json
import random
import re
import sys
import threading
import time
from dataclasses import dataclass

import requests

from runtime.config import GenerationSettings, RuntimeLLMConfig


DEFAULT_SYSTEM_INSTRUCTIONS = "You are a careful and precise assistant."


class LLMClientError(RuntimeError):
    """Raised when the LLM provider fails or returns invalid content."""


@dataclass(frozen=True)
class LLMResponse:
    text: str
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    reasoning_text: str = ""
    reasoning_tokens: int = 0
    finish_reason: str = ""


class LLMClient:
    """Small wrapper around the active experiment LLM provider."""

    CONNECT_TIMEOUT_SECONDS = 10.0
    REQUEST_TIMEOUT_SECONDS = 45.0
    # Provider 层只处理短暂故障；更高层会从任务初始状态公平重跑。
    MAX_RETRIES = 2
    RETRY_BACKOFF_SECONDS = 2.0
    MAX_RETRY_AFTER_SECONDS = 10.0

    def __init__(self, config: RuntimeLLMConfig) -> None:
        self.config = config
        self._usage_lock = threading.Lock()
        self._usage_totals = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "call_count": 0,
            "latency_ms": 0.0,
        }
        from runtime.config import SUPPORTED_PROVIDERS
        if config.provider not in SUPPORTED_PROVIDERS:
            raise LLMClientError(
                "Unsupported provider `%s`. Must be one of %s." % (config.provider, sorted(SUPPORTED_PROVIDERS))
            )

    def usage_snapshot(self) -> dict[str, int | float]:
        """Return a process-local cumulative usage snapshot.

        Runners use before/after deltas so post-runtime compiler calls and
        successful calls made before a task-level retry cannot disappear from
        the episode record.
        """
        with self._usage_lock:
            return dict(self._usage_totals)

    def _record_response_usage(self, response: LLMResponse) -> None:
        with self._usage_lock:
            self._usage_totals["prompt_tokens"] += int(response.prompt_tokens or 0)
            self._usage_totals["completion_tokens"] += int(response.completion_tokens or 0)
            self._usage_totals["total_tokens"] += int(
                response.total_tokens
                or response.prompt_tokens + response.completion_tokens
            )
            self._usage_totals["call_count"] += 1
            self._usage_totals["latency_ms"] += float(response.latency_ms or 0.0)

    def _sanitize(self, text: str) -> str:
        sanitized = re.sub(r"sk-[A-Za-z0-9_-]+", "[REDACTED_API_KEY]", text)
        sanitized = re.sub(r"Bearer\s+[A-Za-z0-9._-]+", "Bearer [REDACTED_TOKEN]", sanitized)
        sanitized = re.sub(r"eyJ[A-Za-z0-9._-]+", "[REDACTED_TOKEN]", sanitized)
        return sanitized[:600]

    def _usage_int(self, usage: object, field: str) -> int:
        if isinstance(usage, dict):
            return int(usage.get(field, 0) or 0)
        return int(getattr(usage, field, 0) or 0)

    def _is_transient_transport_error(self, exc: Exception) -> bool:
        transient_types = (
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
            requests.exceptions.SSLError,
            requests.exceptions.ChunkedEncodingError,
        )
        return isinstance(exc, transient_types)

    def _is_transient_status(self, status_code: int) -> bool:
        return status_code in {408, 409, 425, 429} or 500 <= status_code < 600

    def _retry_after_seconds(self, response: requests.Response) -> float | None:
        header_value = str(response.headers.get("retry-after", "") or "").strip()
        if not header_value:
            return None
        try:
            parsed = float(header_value)
        except ValueError:
            return None
        return max(0.0, min(parsed, self.MAX_RETRY_AFTER_SECONDS))

    def _retry_backoff(self, retry_index: int, *, response: requests.Response | None = None) -> None:
        retry_after = self._retry_after_seconds(response) if response is not None else None
        base_delay = retry_after if retry_after is not None else self.RETRY_BACKOFF_SECONDS * (2 ** max(retry_index - 1, 0))
        jitter = 0.2 * random.random() * max(base_delay, 1.0)
        time.sleep(min(base_delay + jitter, self.MAX_RETRY_AFTER_SECONDS))

    def _report_retry(self, retry_index: int, reason: str) -> None:
        """显式记录有限重试，避免长请求被误判为静默卡死。"""
        print("[llm-retry] provider attempt %d/%d: %s" % (
            retry_index, self.MAX_RETRIES, self._sanitize(reason)),
            file=sys.stderr, flush=True)

    def _request_timeout(self, settings: GenerationSettings) -> tuple[float, float]:
        read_timeout = (float(settings.request_timeout_seconds)
                        if settings.request_timeout_seconds is not None
                        else self.REQUEST_TIMEOUT_SECONDS)
        return (self.CONNECT_TIMEOUT_SECONDS, max(1.0, read_timeout))

    def generate(self, *, instructions: str, input_text: str, settings: GenerationSettings) -> LLMResponse:
        if self.config.provider == "openrouter":
            return self._generate_openrouter(
                instructions=instructions,
                input_text=input_text,
                settings=settings,
            )
        raise LLMClientError(
            "Unsupported provider `%s` at generation time." % self.config.provider
        )

    # ------------------------------------------------------------------
    # OpenRouter (OpenAI Chat Completions compatible)
    # ------------------------------------------------------------------

    def _generate_openrouter(
        self, *, instructions: str, input_text: str, settings: GenerationSettings,
    ) -> LLMResponse:
        effective_instructions = (instructions.strip() or DEFAULT_SYSTEM_INSTRUCTIONS)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": effective_instructions},
            {"role": "user", "content": input_text},
        ]
        payload: dict[str, object] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": settings.temperature,
            "max_tokens": settings.max_output_tokens,
        }
        # DeepSeek 官方接口默认启用 thinking。ALFWorld 每一步只需从合法命令
        # 中选择一个动作，开启长推理会把 256 token 全耗在 reasoning_content，
        # 导致 content=""/finish_reason=length。仅对显式请求且为 DeepSeek 时关闭。
        if (settings.thinking in {"enabled", "disabled"}
                and ("deepseek" in self.config.base_url.lower()
                     or "deepseek" in self.config.model.lower())):
            payload["thinking"] = {"type": settings.thinking}
            if settings.thinking == "enabled" and settings.reasoning_effort:
                payload["reasoning_effort"] = settings.reasoning_effort
        url = "%s/chat/completions" % self.config.base_url.rstrip("/")
        headers = {
            "Authorization": "Bearer %s" % self.config.api_key,
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/DEFENSE-SEU/FlowEvo",
            "X-Title": self.config.app_name or "FlowEvo",
        }
        started = time.perf_counter()
        transient_retry_count = 0
        last_error = "retry budget exhausted"
        # 普通调用保留旧的 length 扩容容错。严格结构化 Extractor
        # 可要求在“CoT 非空 + 正文为空 + length”时改用非思考 JSON 重试。
        max_tokens = int(settings.max_output_tokens)
        max_tokens_bumped = False
        thinking_fallback_used = False

        for attempt in range(self.MAX_RETRIES + 1):
            payload["max_tokens"] = max_tokens
            try:
                request_kwargs = {
                    "headers": headers, "json": payload,
                    "timeout": self._request_timeout(settings),
                }
                if settings.stream_response:
                    payload["stream"] = True
                    request_kwargs["stream"] = True
                response = requests.post(url, **request_kwargs)
            except Exception as exc:  # noqa: BLE001
                last_error = self._sanitize("%s: %s" % (type(exc).__name__, exc))
                if self._is_transient_transport_error(exc) and transient_retry_count < self.MAX_RETRIES:
                    transient_retry_count += 1
                    self._report_retry(transient_retry_count, last_error)
                    self._retry_backoff(transient_retry_count)
                    continue
                raise LLMClientError(last_error) from exc

            if not response.ok:
                last_error = self._sanitize(
                    "HTTP %d for %s. Body: %s" % (response.status_code, url, response.text[:500])
                )
                if self._is_transient_status(response.status_code) and transient_retry_count < self.MAX_RETRIES:
                    transient_retry_count += 1
                    self._report_retry(transient_retry_count, last_error)
                    self._retry_backoff(transient_retry_count, response=response)
                    continue
                raise LLMClientError(last_error)

            try:
                data = (self._consume_chat_stream(response)
                        if settings.stream_response else response.json())
            except Exception as exc:  # noqa: BLE001
                last_error = self._sanitize(
                    "Stream/JSON response error: %s: %s"
                    % (type(exc).__name__, exc))
                if (self._is_transient_transport_error(exc)
                        and transient_retry_count < self.MAX_RETRIES):
                    transient_retry_count += 1
                    self._report_retry(transient_retry_count, last_error)
                    self._retry_backoff(transient_retry_count)
                    continue
                raise LLMClientError(last_error) from exc

            choices = data.get("choices") or []
            text = ""
            finish_reason = ""
            reasoning_content = ""
            if choices:
                message = choices[0].get("message") or {}
                text = str(message.get("content") or "").strip()
                reasoning_content = str(message.get("reasoning_content") or "").strip()
                finish_reason = str(choices[0].get("finish_reason") or "")
            if not text:
                # 间歇性空响应（provider 过载等）：按瞬时错误重试，避免一次空响应
                # 就终止整个 episode（与 v2.0 runtime 的容错行为对齐）。
                if (finish_reason == "length" and reasoning_content
                        and settings.fallback_disable_thinking_on_length
                        and not thinking_fallback_used
                        and settings.thinking == "enabled"):
                    thinking_fallback_used = True
                    payload["thinking"] = {"type": "disabled"}
                    payload.pop("reasoning_effort", None)
                    if settings.fallback_response_format:
                        payload["response_format"] = {
                            "type": settings.fallback_response_format}
                    last_error = self._sanitize(
                        "empty content after reasoning exhausted output budget "
                        "(finish_reason=length) for %s -> retry with "
                        "thinking=disabled%s"
                        % (url, " and response_format=%s"
                           % settings.fallback_response_format
                           if settings.fallback_response_format else "")
                    )
                elif finish_reason == "length" and not max_tokens_bumped:
                    max_tokens_bumped = True
                    # 旧实现对初始预算 >1024 的任务反而会降到1024。
                    max_tokens = min(max_tokens * 2, 8192)
                    last_error = self._sanitize(
                        "empty content (finish_reason=length) for %s -> retry with max_tokens=%d. Body: %s"
                        % (url, max_tokens, json.dumps(data, ensure_ascii=False)[:200])
                    )
                else:
                    last_error = self._sanitize(
                        "empty content (finish_reason=%s) for %s. Body: %s"
                        % (finish_reason, url, json.dumps(data, ensure_ascii=False)[:300])
                    )
                if transient_retry_count < self.MAX_RETRIES:
                    transient_retry_count += 1
                    self._report_retry(transient_retry_count, last_error)
                    self._retry_backoff(transient_retry_count, response=response)
                    continue
                raise LLMClientError(last_error)

            usage = data.get("usage") or {}
            usage_details = (usage.get("completion_tokens_details") or {}
                             if isinstance(usage, dict) else {})
            latency_ms = (time.perf_counter() - started) * 1000.0
            llm_response = LLMResponse(
                text=text,
                provider=self.config.provider,
                model=self.config.model,
                prompt_tokens=self._usage_int(usage, "prompt_tokens"),
                completion_tokens=self._usage_int(usage, "completion_tokens"),
                total_tokens=self._usage_int(usage, "total_tokens"),
                latency_ms=latency_ms,
                reasoning_text=reasoning_content,
                reasoning_tokens=self._usage_int(usage_details, "reasoning_tokens"),
                finish_reason=finish_reason,
            )
            self._record_response_usage(llm_response)
            return llm_response

        raise LLMClientError(
            self._sanitize(
                "Exceeded retry limit after %d transient retries. Last error: %s"
                % (transient_retry_count, last_error)
            )
        )

    @staticmethod
    def _consume_chat_stream(response: requests.Response) -> dict:
        """Consume OpenAI-compatible SSE while preserving CoT and final content.

        A streaming read prevents a long reasoning generation from being treated as
        one silent HTTP read: requests' read timeout now applies between SSE chunks.
        """
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        finish_reason = ""
        usage: dict = {}
        for raw_line in response.iter_lines(decode_unicode=True):
            if raw_line is None:
                continue
            line = (raw_line.decode("utf-8", errors="replace")
                    if isinstance(raw_line, bytes) else str(raw_line)).strip()
            if not line or not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            chunk = json.loads(body)
            if isinstance(chunk.get("usage"), dict):
                usage = chunk["usage"]
            choices = chunk.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta") or choice.get("message") or {}
            if delta.get("reasoning_content"):
                reasoning_parts.append(str(delta["reasoning_content"]))
            if delta.get("content"):
                content_parts.append(str(delta["content"]))
            if choice.get("finish_reason"):
                finish_reason = str(choice["finish_reason"])
        return {
            "choices": [{
                "message": {
                    "content": "".join(content_parts),
                    "reasoning_content": "".join(reasoning_parts),
                },
                "finish_reason": finish_reason,
            }],
            "usage": usage,
        }
