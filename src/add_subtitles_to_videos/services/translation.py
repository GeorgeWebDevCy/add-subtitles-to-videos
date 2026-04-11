from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
import os
from threading import local
from typing import Any

import requests

from ..config import DEFAULT_MAX_PARALLEL_TRANSLATION_BATCHES

from ..languages import language_label
from ..models import SubtitleSegment, TranslationSegment
from . import OperationCancelledError

LogReporter = Callable[[str], None]
CancelChecker = Callable[[], bool]


class TranslationServiceError(RuntimeError):
    pass


class TranslationTransportError(TranslationServiceError):
    pass


@dataclass(slots=True)
class TranslationProviderConfig:
    base_url: str
    api_key: str
    model: str
    timeout_seconds: int = 120

    @classmethod
    def from_values(
        cls,
        *,
        base_url: str,
        api_key: str,
        model: str,
    ) -> TranslationProviderConfig:
        return cls(
            base_url=base_url.strip() or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            api_key=api_key.strip() or os.environ.get("OPENAI_API_KEY", ""),
            model=model.strip() or os.environ.get("OPENAI_TRANSLATION_MODEL", "gpt-4.1-mini"),
        )


class TranslationService(ABC):
    provider_name: str

    @abstractmethod
    def is_configured(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def configuration_status(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def translate_segments(
        self,
        segments: list[SubtitleSegment],
        *,
        source_language: str,
        target_language: str,
        log: LogReporter | None = None,
        cancel_requested: CancelChecker | None = None,
    ) -> list[TranslationSegment]:
        raise NotImplementedError

    def test_connection(
        self,
        *,
        source_language: str = "en",
        target_language: str = "de",
        log: LogReporter | None = None,
    ) -> str:
        result = self.translate_segments(
            [
                SubtitleSegment(
                    start_seconds=0.0,
                    end_seconds=1.0,
                    text="This is a connection test.",
                )
            ],
            source_language=source_language,
            target_language=target_language,
            log=log,
        )
        return result[0].translated_text


class OpenAICompatibleTranslationService(TranslationService):
    provider_name = "openai_compatible"

    def __init__(
        self,
        config: TranslationProviderConfig,
        *,
        max_parallel_batches: int = DEFAULT_MAX_PARALLEL_TRANSLATION_BATCHES,
    ) -> None:
        self._config = config
        self._max_parallel_batches = max(1, max_parallel_batches)
        self._session_local = local()

    def is_configured(self) -> bool:
        return bool(self._config.api_key and self._config.base_url and self._config.model)

    def configuration_status(self) -> str:
        if not self._config.base_url:
            return "Translation unavailable: missing base URL."
        if not self._config.model:
            return "Translation unavailable: missing model."
        if not self._config.api_key:
            return "Translation unavailable: missing API key."
        return f"Configured: {self.provider_name} ({self._config.model})"

    def translate_segments(
        self,
        segments: list[SubtitleSegment],
        *,
        source_language: str,
        target_language: str,
        log: LogReporter | None = None,
        cancel_requested: CancelChecker | None = None,
    ) -> list[TranslationSegment]:
        self._ensure_configured()
        translated_segments: list[TranslationSegment] = []
        source_label = language_label(source_language)
        target_label = language_label(target_language)
        batches = list(enumerate(_chunk_segments(segments), start=1))
        translated_batches = self._translate_batches(
            batches,
            source_label=source_label,
            target_label=target_label,
            log=log,
            cancel_requested=cancel_requested,
        )

        for batch_index, batch in batches:
            _check_cancel(cancel_requested)
            translated_texts = translated_batches[batch_index]
            _check_cancel(cancel_requested)

            if len(translated_texts) != len(batch):
                raise TranslationServiceError(
                    "Translation provider returned a different number of subtitle lines than expected."
                )

            for segment, translated_text in zip(batch, translated_texts, strict=True):
                translated_segments.append(
                    TranslationSegment(
                        start_seconds=segment.start_seconds,
                        end_seconds=segment.end_seconds,
                        source_text=segment.text,
                        translated_text=translated_text.strip(),
                    )
                )

        return translated_segments

    def _translate_batches(
        self,
        batches: list[tuple[int, list[SubtitleSegment]]],
        *,
        source_label: str,
        target_label: str,
        log: LogReporter | None = None,
        cancel_requested: CancelChecker | None = None,
    ) -> dict[int, list[str]]:
        if len(batches) <= 1 or self._max_parallel_batches == 1:
            return self._translate_batches_sequential(
                batches,
                source_label=source_label,
                target_label=target_label,
                log=log,
                cancel_requested=cancel_requested,
            )

        max_workers = min(self._max_parallel_batches, len(batches))
        if log is not None:
            log(f"Running up to {max_workers} translation batches in parallel.")

        results: dict[int, list[str]] = {}
        executor = ThreadPoolExecutor(max_workers=max_workers)
        futures = {
            executor.submit(
                self._translate_batch_job,
                batch_index,
                batch,
                source_label=source_label,
                target_label=target_label,
                log=log,
            ): batch_index
            for batch_index, batch in batches
        }

        fallback_required = False
        try:
            for future in as_completed(futures):
                _check_cancel(cancel_requested)
                batch_index = futures[future]
                try:
                    results[batch_index] = future.result()
                except TranslationTransportError as exc:
                    fallback_required = True
                    if log is not None:
                        log(
                            f"{exc} Falling back to sequential translation for the rest of this file."
                        )
                    break
        finally:
            executor.shutdown(wait=not fallback_required, cancel_futures=fallback_required)

        if not fallback_required:
            return results

        remaining_batches = [
            (batch_index, batch)
            for batch_index, batch in batches
            if batch_index not in results
        ]
        results.update(
            self._translate_batches_sequential(
                remaining_batches,
                source_label=source_label,
                target_label=target_label,
                log=log,
                cancel_requested=cancel_requested,
            )
        )
        return results

    def _translate_batches_sequential(
        self,
        batches: list[tuple[int, list[SubtitleSegment]]],
        *,
        source_label: str,
        target_label: str,
        log: LogReporter | None = None,
        cancel_requested: CancelChecker | None = None,
    ) -> dict[int, list[str]]:
        results: dict[int, list[str]] = {}
        for batch_index, batch in batches:
            _check_cancel(cancel_requested)
            results[batch_index] = self._translate_batch_job(
                batch_index,
                batch,
                source_label=source_label,
                target_label=target_label,
                log=log,
            )
        return results

    def _translate_batch_job(
        self,
        batch_index: int,
        batch: list[SubtitleSegment],
        *,
        source_label: str,
        target_label: str,
        log: LogReporter | None = None,
    ) -> list[str]:
        if log is not None:
            log(
                f"Submitting translation batch {batch_index} ({len(batch)} segments) "
                f"from {source_label} to {target_label} via {self.provider_name}"
            )
            log(
                f"Translation request: model={self._config.model} "
                f"endpoint={self._config.base_url.rstrip('/')}/chat/completions"
            )
        return self._translate_batch_adaptive(
            batch,
            source_label=source_label,
            target_label=target_label,
            log=log,
        )

    def _translate_batch_adaptive(
        self,
        segments: list[SubtitleSegment],
        *,
        source_label: str,
        target_label: str,
        log: LogReporter | None = None,
    ) -> list[str]:
        try:
            translated_texts = self._translate_batch(
                segments,
                source_label=source_label,
                target_label=target_label,
                log=log,
            )
        except TranslationServiceError as exc:
            if len(segments) == 1 or not _should_retry_with_smaller_batch(str(exc)):
                raise
            midpoint = max(1, len(segments) // 2)
            if log is not None:
                log(
                    "Translation batch could not be parsed reliably. "
                    f"Retrying with smaller batches ({midpoint} + {len(segments) - midpoint})."
                )
            return self._translate_batch_adaptive(
                segments[:midpoint],
                source_label=source_label,
                target_label=target_label,
                log=log,
            ) + self._translate_batch_adaptive(
                segments[midpoint:],
                source_label=source_label,
                target_label=target_label,
                log=log,
            )

        if len(translated_texts) == len(segments):
            return translated_texts

        if len(segments) == 1:
            raise TranslationServiceError(
                "Translation provider returned a different number of subtitle lines than expected."
            )

        midpoint = max(1, len(segments) // 2)
        if log is not None:
            log(
                "Translation batch returned the wrong number of subtitle lines. "
                f"Retrying with smaller batches ({midpoint} + {len(segments) - midpoint})."
            )
        return self._translate_batch_adaptive(
            segments[:midpoint],
            source_label=source_label,
            target_label=target_label,
            log=log,
        ) + self._translate_batch_adaptive(
            segments[midpoint:],
            source_label=source_label,
            target_label=target_label,
            log=log,
        )

    def _translate_batch(
        self,
        segments: list[SubtitleSegment],
        *,
        source_label: str,
        target_label: str,
        log: LogReporter | None = None,
    ) -> list[str]:
        endpoint = self._config.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self._config.model,
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a professional subtitle translator.\n"
                        "\n"
                        "Translate subtitle text from the source language to the target language.\n"
                        "\n"
                        "Rules:\n"
                        "- Preserve the exact number of subtitle items.\n"
                        "- Preserve the exact order of subtitle items.\n"
                        "- Translate only subtitle text, not metadata.\n"
                        "- Do not merge, split, reorder, or omit items.\n"
                        "- Keep names, brands, numbers, URLs, and proper nouns accurate.\n"
                        "- Keep tone, meaning, and speaker intent faithful.\n"
                        "- Prefer natural subtitle phrasing over overly literal translation.\n"
                        "- If a phrase should remain unchanged, keep it unchanged rather than guessing.\n"
                        "- Do not add explanations, notes, markdown, or code fences.\n"
                        "- Return valid JSON only using this exact shape:\n"
                        "{\"translations\":[{\"text\":\"...\"}]}"
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "source_language": source_label,
                            "target_language": target_label,
                            "instruction": (
                                "Translate each subtitle item into subtitle-ready language for the target audience. "
                                "Return JSON only."
                            ),
                            "schema": {"translations": [{"text": "translated subtitle line"}]},
                            "segments": [
                                {"index": index, "text": segment.text}
                                for index, segment in enumerate(segments, start=1)
                            ],
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
        }

        session = self._session()
        try:
            response = session.post(
                endpoint,
                json=payload,
                timeout=self._config.timeout_seconds,
            )
        except requests.Timeout as exc:
            raise TranslationTransportError("Translation provider request failed: timed out") from exc
        except requests.ConnectionError as exc:
            raise TranslationTransportError("Translation provider request failed: connection error") from exc
        except requests.RequestException as exc:
            raise TranslationTransportError(
                f"Translation provider request failed: {exc}"
            ) from exc

        if response.status_code >= 400:
            detail = response.text
            if response.status_code in {408, 429} or response.status_code >= 500:
                raise TranslationTransportError(
                    f"Translation provider returned HTTP {response.status_code}.\n{detail}"
                )
            raise TranslationServiceError(
                f"Translation provider returned HTTP {response.status_code}.\n{detail}"
            )

        try:
            raw_payload = response.json()
        except json.JSONDecodeError as exc:
            raise TranslationServiceError(
                "Translation provider returned invalid JSON at the HTTP layer."
            ) from exc

        content = _extract_message_text(raw_payload)
        if log is not None:
            preview = " ".join(content.split())
            if len(preview) > 240:
                preview = preview[:237] + "..."
            log(f"Translation response preview: {preview}")
        parsed = _extract_json(content)
        translations = parsed.get("translations")
        if not isinstance(translations, list):
            raise TranslationServiceError("Translation provider did not return a translations array.")

        result: list[str] = []
        for item in translations:
            text = _translation_text_from_item(item)
            if not isinstance(text, str) or not text.strip():
                raise TranslationServiceError("Translation provider returned an empty translated subtitle.")
            result.append(text)
        return result

    def _ensure_configured(self) -> None:
        if not self.is_configured():
            raise TranslationServiceError(self.configuration_status())

    def _session(self) -> requests.Session:
        session = getattr(self._session_local, "session", None)
        if session is None:
            session = requests.Session()
            session.headers.update(
                {
                    "Authorization": f"Bearer {self._config.api_key}",
                    "Content-Type": "application/json",
                }
            )
            self._session_local.session = session
        return session


def _chunk_segments(
    segments: list[SubtitleSegment],
    *,
    max_segments: int = 20,
    max_characters: int = 1800,
) -> list[list[SubtitleSegment]]:
    chunks: list[list[SubtitleSegment]] = []
    current_chunk: list[SubtitleSegment] = []
    current_length = 0

    for segment in segments:
        segment_length = len(segment.text)
        if current_chunk and (
            len(current_chunk) >= max_segments or current_length + segment_length > max_characters
        ):
            chunks.append(current_chunk)
            current_chunk = []
            current_length = 0

        current_chunk.append(segment)
        current_length += segment_length

    if current_chunk:
        chunks.append(current_chunk)
    return chunks


def _extract_message_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise TranslationServiceError("Translation provider response did not include choices.")

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise TranslationServiceError("Translation provider returned an invalid choice payload.")

    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise TranslationServiceError("Translation provider response did not include a message payload.")

    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise TranslationServiceError("Translation provider response did not include translated text.")
    return content


def _extract_json(content: str) -> dict[str, Any]:
    candidate = content.strip()
    if candidate.startswith("```"):
        parts = [part for part in candidate.split("```") if part.strip()]
        candidate = parts[0]
        if candidate.lower().startswith("json"):
            candidate = candidate[4:].strip()

    decoder = json.JSONDecoder()
    try:
        parsed, _ = decoder.raw_decode(candidate)
    except json.JSONDecodeError as exc:
        raise TranslationServiceError(
            "Translation provider did not return valid JSON for subtitle translation."
        ) from exc
    if not isinstance(parsed, dict):
        raise TranslationServiceError("Translation provider returned a non-object JSON payload.")
    return parsed


def _translation_text_from_item(item: Any) -> str | None:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in ("text", "translated_text", "translation"):
            value = item.get(key)
            if isinstance(value, str):
                return value
    raise TranslationServiceError("Translation provider returned an invalid translation item.")


def _check_cancel(cancel_requested: CancelChecker | None) -> None:
    if cancel_requested is not None and cancel_requested():
        raise OperationCancelledError("Processing stopped by user.")


def _should_retry_with_smaller_batch(message: str) -> bool:
    lowered = message.casefold()
    return not (
        lowered.startswith("translation provider returned http")
        or lowered.startswith("translation provider request failed")
        or lowered.startswith("translation provider timed out")
        or lowered.startswith("translation unavailable:")
    )
