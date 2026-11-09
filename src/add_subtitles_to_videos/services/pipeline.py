from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

from ..config import is_audio_file
from ..languages import language_label, MIXED_LANGUAGE_CODE, OPPOSITE_LANGUAGE_CODE
from ..models import (
    OutputMode,
    PipelineResult,
    ProcessingOptions,
    SubtitleSegment,
    TranscriptionMetadata,
    TranscriptionResult,
    TranslationSegment,
)
from . import OperationCancelledError, ffmpeg
from .subtitles import (
    parse_srt_text,
    segments_to_srt_text,
    translation_segments_to_srt_text,
    validate_review_srt_text,
)
from .media_edit import format_clock, keep_segments, total_duration
from .translation import TranslationService
from .whisper import WhisperService

ProgressReporter = Callable[[float, str], None]
LogReporter = Callable[[str], None]
CancelChecker = Callable[[], bool]


class SubtitlePipeline:
    def __init__(
        self,
        whisper_service: WhisperService | None = None,
        translation_service: TranslationService | None = None,
    ) -> None:
        self._whisper_service = whisper_service or WhisperService()
        self._translation_service = translation_service

    def process_video(
        self,
        video_path: Path,
        options: ProcessingOptions,
        *,
        progress: ProgressReporter | None = None,
        log: LogReporter | None = None,
        cancel_requested: CancelChecker | None = None,
    ) -> PipelineResult:
        transcription = self.transcribe(
            video_path,
            options,
            progress=progress,
            log=log,
            cancel_requested=cancel_requested,
        )
        return self.finalize(
            transcription,
            transcription.translated_srt_text,
            options,
            progress=progress,
            log=log,
            cancel_requested=cancel_requested,
        )

    def transcribe(
        self,
        video_path: Path,
        options: ProcessingOptions,
        *,
        progress: ProgressReporter | None = None,
        log: LogReporter | None = None,
        cancel_requested: CancelChecker | None = None,
    ) -> TranscriptionResult:
        input_video = video_path.expanduser().resolve()
        stage_durations: dict[str, float] = {}

        if options.remove_ranges or is_audio_file(input_video):
            self._emit(progress, 0.01, "Cutting and preparing media")
            prepare_started_at = perf_counter()
            input_video = self._prepare_media(
                input_video,
                options,
                log=log,
                cancel_requested=cancel_requested,
            )
            stage_durations["media_prepare_seconds"] = perf_counter() - prepare_started_at
            self._emit_log(
                log,
                f"Media preparation completed in {stage_durations['media_prepare_seconds']:.2f}s",
            )
            self._ensure_not_cancelled(cancel_requested)

        self._emit(progress, 0.03, "Preparing audio")
        self._emit_log(log, f"Using FFmpeg binary at {ffmpeg.ffmpeg_binary()}")

        with TemporaryDirectory(prefix="subtitle-foundry-audio-") as temp_dir:
            audio_path = Path(temp_dir) / "audio.wav"
            self._emit_log(log, f"Extracting mono audio from {input_video.name}")
            extract_started_at = perf_counter()
            ffmpeg.extract_audio(input_video, audio_path, cancel_requested=cancel_requested)
            stage_durations["audio_extraction_seconds"] = perf_counter() - extract_started_at
            self._emit_log(
                log,
                f"Audio extraction completed in {stage_durations['audio_extraction_seconds']:.2f}s",
            )
            self._ensure_not_cancelled(cancel_requested)

            self._emit(progress, 0.22, "Running Whisper")
            whisper_started_at = perf_counter()
            source_segments, metadata = self._whisper_service.transcribe(
                audio_path,
                options,
                log=log,
                cancel_requested=cancel_requested,
            )
            stage_durations["whisper_seconds"] = perf_counter() - whisper_started_at
            self._emit_log(log, f"Whisper transcription completed in {stage_durations['whisper_seconds']:.2f}s")

        self._ensure_not_cancelled(cancel_requested)
        if not source_segments:
            raise RuntimeError(
                "Whisper returned no subtitle segments. Try a larger model or confirm the source language."
            )

        if metadata.stage_durations:
            stage_durations.update(metadata.stage_durations)

        effective_source_language = metadata.detected_language or options.source_language or "unknown"
        if options.source_language == MIXED_LANGUAGE_CODE or options.target_language == OPPOSITE_LANGUAGE_CODE:
            effective_source_language = MIXED_LANGUAGE_CODE
        translation_applied = options.target_language != effective_source_language
        metadata.target_language = options.target_language

        if translation_applied:
            if self._translation_service is None:
                raise RuntimeError("Translation service is not configured.")
            metadata.translation_provider = self._translation_service.provider_name
            metadata.translation_applied = True
            self._emit(progress, 0.5, "Translating subtitle text")
            translation_started_at = perf_counter()
            review_segments = self._translation_service.translate_segments(
                source_segments,
                source_language=effective_source_language,
                target_language=options.target_language,
                log=log,
                cancel_requested=cancel_requested,
            )
            stage_durations["translation_seconds"] = perf_counter() - translation_started_at
            self._emit_log(log, f"Translation completed in {stage_durations['translation_seconds']:.2f}s")
        else:
            metadata.translation_provider = None
            metadata.translation_applied = False
            stage_durations["translation_seconds"] = 0.0
            review_segments = [
                TranslationSegment(
                    start_seconds=segment.start_seconds,
                    end_seconds=segment.end_seconds,
                    source_text=segment.text,
                    translated_text=segment.text,
                )
                for segment in source_segments
            ]
            self._emit_log(log, "Translation skipped because source and target languages match.")

        metadata.stage_durations = stage_durations

        warning_messages = self._build_review_flags(
            review_segments,
            options,
            metadata,
        )
        for warning in warning_messages:
            self._emit_log(log, f"Review flag: {warning}")

        source_srt_text = segments_to_srt_text(
            source_segments,
            max_line_length=options.max_line_length,
        )
        translated_srt_text = translation_segments_to_srt_text(
            review_segments,
            text_field="translated_text",
            max_line_length=options.max_line_length,
        )
        self._emit(progress, 0.76, "Transcription complete - ready for review")

        return TranscriptionResult(
            input_video=input_video,
            source_segments=source_segments,
            review_segments=review_segments,
            metadata=metadata,
            warning_messages=warning_messages,
            source_srt_text=source_srt_text,
            translated_srt_text=translated_srt_text,
        )

    def finalize(
        self,
        transcription: TranscriptionResult,
        srt_text: str,
        options: ProcessingOptions,
        *,
        progress: ProgressReporter | None = None,
        log: LogReporter | None = None,
        cancel_requested: CancelChecker | None = None,
    ) -> PipelineResult:
        started_at = perf_counter()
        stage_durations: dict[str, float] = {}
        if transcription.metadata.stage_durations:
            stage_durations.update(transcription.metadata.stage_durations)

        validation_started_at = perf_counter()
        validation_error = validate_review_srt_text(srt_text, transcription.review_segments)
        stage_durations["validation_seconds"] = perf_counter() - validation_started_at
        self._emit_log(log, f"Subtitle validation completed in {stage_durations['validation_seconds']:.2f}s")
        if validation_error is not None:
            raise RuntimeError(validation_error)

        write_started_at = perf_counter()
        translated_segments = parse_srt_text(srt_text, allow_empty_text=True)
        video_path = transcription.input_video
        output_directory = options.output_directory.expanduser().resolve()
        output_directory.mkdir(parents=True, exist_ok=True)

        subtitle_file = output_directory / f"{video_path.stem}.{options.target_language}.srt"
        burned_video = (
            output_directory / f"{video_path.stem}.subtitled{video_path.suffix}"
            if options.output_mode == OutputMode.BURNED_VIDEO
            else None
        )

        self._emit(progress, 0.76, "Writing subtitle file")
        subtitle_file.write_text(srt_text, encoding="utf-8")
        stage_durations["subtitle_write_seconds"] = perf_counter() - write_started_at
        segment_count = len(translated_segments)
        preview_text = self._build_preview_text(translated_segments)
        self._emit_log(log, f"Wrote {segment_count} subtitle segments to {subtitle_file.name}")
        self._emit_log(log, f"Subtitle file write completed in {stage_durations['subtitle_write_seconds']:.2f}s")

        if burned_video is not None:
            self._ensure_not_cancelled(cancel_requested)
            self._emit(progress, 0.86, "Burning subtitles into video")
            burn_started_at = perf_counter()
            ffmpeg.burn_subtitles(
                video_path,
                subtitle_file,
                burned_video,
                font_size=options.subtitle_font_size,
                outline_style=options.outline_subtitles,
                cancel_requested=cancel_requested,
            )
            stage_durations["subtitle_burn_seconds"] = perf_counter() - burn_started_at
            self._emit_log(log, f"Wrote subtitled video to {burned_video.name}")
            self._emit_log(log, f"Burned subtitle export completed in {stage_durations['subtitle_burn_seconds']:.2f}s")
        else:
            stage_durations["subtitle_burn_seconds"] = 0.0

        elapsed_seconds = perf_counter() - started_at
        stage_durations["finalize_seconds"] = elapsed_seconds
        self._emit(progress, 1.0, "Done")
        self._emit_log(log, f"Finalize completed in {elapsed_seconds:.2f}s")

        return PipelineResult(
            input_video=video_path,
            subtitle_file=subtitle_file,
            burned_video=burned_video,
            detected_language=transcription.metadata.detected_language,
            target_language=options.target_language,
            translation_provider=transcription.metadata.translation_provider,
            device_label=transcription.metadata.device_label,
            segment_count=segment_count,
            elapsed_seconds=elapsed_seconds,
            preview_text=preview_text,
            warning_messages=transcription.warning_messages,
            stage_durations=stage_durations,
        )

    def _prepare_media(
        self,
        input_path: Path,
        options: ProcessingOptions,
        *,
        log: LogReporter | None,
        cancel_requested: CancelChecker | None,
    ) -> Path:
        """Cut the requested ranges (and add a picture to audio) into <name>.edited.mp4.

        The edited file is saved in the output folder and replaces the original for
        the rest of the job, so subtitles line up with the edited timing.
        """
        audio_only = is_audio_file(input_path)
        duration = ffmpeg.probe_duration(input_path)
        try:
            keep = keep_segments(list(options.remove_ranges), duration)
        except ValueError as exc:
            raise RuntimeError(f"{input_path.name}: {exc}") from None

        output_directory = options.output_directory.expanduser().resolve()
        output_path = output_directory / f"{input_path.stem}.edited.mp4"

        if options.remove_ranges:
            removed = duration - total_duration(keep)
            self._emit_log(
                log,
                f"Cutting {len(options.remove_ranges)} range(s) from {input_path.name}: "
                f"{format_clock(duration)} -> {format_clock(duration - removed)} "
                f"({format_clock(removed)} removed)",
            )
        cover_image = options.cover_image
        intro_image = options.intro_image
        show_waveform = options.show_waveform
        canvas = ffmpeg.LANDSCAPE_CANVAS
        if audio_only:
            if cover_image is not None:
                canvas = ffmpeg.canvas_for_picture(ffmpeg.probe_image_size(cover_image))
                self._emit_log(log, f"Using {cover_image.name} as the video picture ({canvas[0]}x{canvas[1]})")
            else:
                self._emit_log(log, "No cover image chosen; the video picture will be black")
            if intro_image is not None:
                self._emit_log(log, f"Adding {intro_image.name} as intro and outro")
            if show_waveform:
                self._emit_log(log, "Adding a moving audio wave")
        else:
            ignored = [
                name
                for name, value in (
                    ("cover image", cover_image),
                    ("intro/outro logo", intro_image),
                    ("audio wave", show_waveform or None),
                )
                if value is not None
            ]
            if ignored:
                self._emit_log(log, f"Ignoring {', '.join(ignored)} because this file already has video")
            cover_image = None
            intro_image = None
            show_waveform = False

        ffmpeg.prepare_media(
            input_path,
            output_path,
            keep=keep,
            audio_only=audio_only,
            cover_image=cover_image,
            canvas=canvas,
            intro_image=intro_image,
            show_waveform=show_waveform,
            cancel_requested=cancel_requested,
        )
        self._emit_log(log, f"Saved edited media to {output_path}")
        return output_path

    @staticmethod
    def _emit(progress: ProgressReporter | None, value: float, message: str) -> None:
        if progress is not None:
            progress(value, message)

    @staticmethod
    def _emit_log(log: LogReporter | None, message: str) -> None:
        if log is not None:
            log(message)

    @staticmethod
    def _ensure_not_cancelled(cancel_requested: CancelChecker | None) -> None:
        if cancel_requested is not None and cancel_requested():
            raise OperationCancelledError("Processing stopped by user.")

    @staticmethod
    def _build_preview_text(segments: list[SubtitleSegment]) -> str:
        preview_lines: list[str] = []
        preview_segments = [segment for segment in segments if segment.text.strip()]
        for index, segment in enumerate(preview_segments[:5], start=1):
            preview_lines.append(f"{index}. {segment.text.replace(chr(10), ' ')}")
        return "\n".join(preview_lines)

    @staticmethod
    def _build_review_flags(
        review_segments: list[TranslationSegment],
        options: ProcessingOptions,
        metadata: TranscriptionMetadata,
    ) -> tuple[str, ...]:
        warnings: list[str] = []
        mixed = options.source_language == MIXED_LANGUAGE_CODE or options.target_language == OPPOSITE_LANGUAGE_CODE
        if mixed:
            warnings.append(
                "Mixed Greek/English speech: check every language switch in the source transcript. "
                "For a subtitle block containing both languages, check each phrase uses the intended subtitle language."
            )
        if options.target_language == OPPOSITE_LANGUAGE_CODE:
            warnings.append("Check Greek speech has English subtitles and English speech has Greek subtitles.")
        detected_language = metadata.detected_language
        effective_source = detected_language or options.source_language

        if (
            options.source_language
            and not mixed
            and options.source_language != "auto"
            and detected_language
            and options.source_language != detected_language
        ):
            warnings.append(
                f"Selected source language is '{language_label(options.source_language)}', "
                f"but Whisper detected '{language_label(detected_language)}'."
            )

        if options.source_language == "auto" and detected_language:
            warnings.append(
                f"Source language was auto-detected as {language_label(detected_language)}. "
                "Review names and idioms before exporting."
            )

        if len(review_segments) < 3:
            warnings.append(
                "Very few subtitle segments were produced. The audio may be too short, quiet, or hard to understand."
            )

        if metadata.translation_applied and effective_source and options.target_language != effective_source:
            identical_count = sum(
                1
                for segment in review_segments
                if _normalize(segment.source_text) == _normalize(segment.translated_text)
            )
            if review_segments and identical_count / len(review_segments) >= 0.5:
                warnings.append(
                    "Many translated subtitle lines still match the source text. Review the translation before export."
                )

        long_line_count = 0
        fast_reading_count = 0
        for segment in review_segments:
            wrapped_lines = segment.translated_text.splitlines() or [segment.translated_text]
            if any(len(line) > options.max_line_length for line in wrapped_lines):
                long_line_count += 1

            duration = max(segment.end_seconds - segment.start_seconds, 0.01)
            reading_speed = len(_normalize(segment.translated_text).replace(" ", "")) / duration
            if reading_speed > 17:
                fast_reading_count += 1

        if long_line_count:
            warnings.append(
                f"{long_line_count} subtitle block(s) exceed the preferred line length. Adjust line breaks in review."
            )
        if fast_reading_count:
            warnings.append(
                f"{fast_reading_count} subtitle block(s) may read too quickly on screen."
            )

        return tuple(warnings)


def _normalize(text: str) -> str:
    return " ".join(text.split()).casefold()
