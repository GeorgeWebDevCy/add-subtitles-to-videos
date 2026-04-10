from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

from ..models import OutputMode, PipelineResult, ProcessingOptions, SubtitleMode, TranscriptionResult
from . import ffmpeg
from .subtitles import segments_to_srt_text, write_srt
from .whisper import WhisperService

ProgressReporter = Callable[[float, str], None]
LogReporter = Callable[[str], None]


class SubtitlePipeline:
    def __init__(self, whisper_service: WhisperService | None = None) -> None:
        self._whisper_service = whisper_service or WhisperService()

    def process_video(
        self,
        video_path: Path,
        options: ProcessingOptions,
        *,
        progress: ProgressReporter | None = None,
        log: LogReporter | None = None,
    ) -> PipelineResult:
        started_at = perf_counter()
        input_video = video_path.expanduser().resolve()
        output_directory = options.output_directory.expanduser().resolve()
        output_directory.mkdir(parents=True, exist_ok=True)

        subtitle_suffix = "en" if options.subtitle_mode == SubtitleMode.ENGLISH else "native"
        subtitle_file = output_directory / f"{input_video.stem}.{subtitle_suffix}.srt"
        burned_video = (
            output_directory / f"{input_video.stem}.subtitled{input_video.suffix}"
            if options.output_mode == OutputMode.BURNED_VIDEO
            else None
        )

        self._emit(progress, 0.03, "Preparing audio")
        self._emit_log(log, f"Using FFmpeg binary at {ffmpeg.ffmpeg_binary()}")

        with TemporaryDirectory(prefix="subtitle-foundry-audio-") as temp_dir:
            audio_path = Path(temp_dir) / "audio.wav"

            self._emit_log(log, f"Extracting mono audio from {input_video.name}")
            ffmpeg.extract_audio(input_video, audio_path)

            self._emit(progress, 0.22, "Running Whisper")
            segments, metadata = self._whisper_service.transcribe(
                audio_path,
                options,
                log=log,
            )

            if not segments:
                raise RuntimeError(
                    "Whisper returned no subtitle segments. Try a larger model or confirm the source language."
                )

            warning_messages = self._build_review_flags(segments, options, metadata)
            for warning in warning_messages:
                self._emit_log(log, f"Review flag: {warning}")

            self._emit(progress, 0.76, "Writing subtitle file")
            segment_count = write_srt(
                segments,
                subtitle_file,
                max_line_length=options.max_line_length,
            )
            self._emit_log(log, f"Wrote {segment_count} subtitle segments to {subtitle_file.name}")

            if burned_video is not None:
                self._emit(progress, 0.86, "Burning subtitles into video")
                ffmpeg.burn_subtitles(
                    input_video,
                    subtitle_file,
                    burned_video,
                    font_size=options.subtitle_font_size,
                )
                self._emit_log(log, f"Wrote subtitled video to {burned_video.name}")

        elapsed_seconds = perf_counter() - started_at
        self._emit(progress, 1.0, "Done")

        return PipelineResult(
            input_video=input_video,
            subtitle_file=subtitle_file,
            burned_video=burned_video,
            detected_language=metadata.detected_language,
            device_label=metadata.device_label,
            segment_count=segment_count,
            elapsed_seconds=elapsed_seconds,
            preview_text=self._build_preview_text(segments),
            warning_messages=warning_messages,
        )

    def transcribe(
        self,
        video_path: Path,
        options: ProcessingOptions,
        *,
        progress: ProgressReporter | None = None,
        log: LogReporter | None = None,
    ) -> TranscriptionResult:
        input_video = video_path.expanduser().resolve()

        self._emit(progress, 0.03, "Preparing audio")
        self._emit_log(log, f"Using FFmpeg binary at {ffmpeg.ffmpeg_binary()}")

        with TemporaryDirectory(prefix="subtitle-foundry-audio-") as temp_dir:
            audio_path = Path(temp_dir) / "audio.wav"
            self._emit_log(log, f"Extracting mono audio from {input_video.name}")
            ffmpeg.extract_audio(input_video, audio_path)

            self._emit(progress, 0.22, "Running Whisper")
            segments, metadata = self._whisper_service.transcribe(
                audio_path,
                options,
                log=log,
            )

        if not segments:
            raise RuntimeError(
                "Whisper returned no subtitle segments. Try a larger model or confirm the source language."
            )

        warning_messages = self._build_review_flags(segments, options, metadata)
        for warning in warning_messages:
            self._emit_log(log, f"Review flag: {warning}")

        srt_text = segments_to_srt_text(segments, max_line_length=options.max_line_length)
        self._emit(progress, 0.76, "Transcription complete — ready for review")

        return TranscriptionResult(
            input_video=input_video,
            segments=segments,
            metadata=metadata,
            warning_messages=warning_messages,
            srt_text=srt_text,
        )

    def finalize(
        self,
        transcription: TranscriptionResult,
        srt_text: str,
        options: ProcessingOptions,
        *,
        progress: ProgressReporter | None = None,
        log: LogReporter | None = None,
    ) -> PipelineResult:
        started_at = perf_counter()
        video_path = transcription.input_video
        output_directory = options.output_directory.expanduser().resolve()
        output_directory.mkdir(parents=True, exist_ok=True)

        subtitle_suffix = "en" if options.subtitle_mode == SubtitleMode.ENGLISH else "native"
        subtitle_file = output_directory / f"{video_path.stem}.{subtitle_suffix}.srt"
        burned_video = (
            output_directory / f"{video_path.stem}.subtitled{video_path.suffix}"
            if options.output_mode == OutputMode.BURNED_VIDEO
            else None
        )

        self._emit(progress, 0.76, "Writing subtitle file")
        subtitle_file.write_text(srt_text, encoding="utf-8")
        segment_count, preview_text = self._parse_srt_summary(srt_text)
        self._emit_log(log, f"Wrote {segment_count} subtitle segments to {subtitle_file.name}")

        if burned_video is not None:
            self._emit(progress, 0.86, "Burning subtitles into video")
            ffmpeg.burn_subtitles(
                video_path,
                subtitle_file,
                burned_video,
                font_size=options.subtitle_font_size,
            )
            self._emit_log(log, f"Wrote subtitled video to {burned_video.name}")

        elapsed_seconds = perf_counter() - started_at
        self._emit(progress, 1.0, "Done")

        return PipelineResult(
            input_video=video_path,
            subtitle_file=subtitle_file,
            burned_video=burned_video,
            detected_language=transcription.metadata.detected_language,
            device_label=transcription.metadata.device_label,
            segment_count=segment_count,
            elapsed_seconds=elapsed_seconds,
            preview_text=preview_text,
            warning_messages=transcription.warning_messages,
        )

    @staticmethod
    def _parse_srt_summary(srt_text: str) -> tuple[int, str]:
        """Return (segment_count, preview_text) parsed from a raw SRT string."""
        blocks = [b.strip() for b in srt_text.strip().split("\n\n") if b.strip()]
        segment_count = len(blocks)
        preview_lines: list[str] = []
        for i, block in enumerate(blocks[:5], start=1):
            lines = block.splitlines()
            text = " ".join(lines[2:]) if len(lines) > 2 else ""
            if text:
                preview_lines.append(f"{i}. {text}")
        return segment_count, "\n".join(preview_lines)

    @staticmethod
    def _emit(progress: ProgressReporter | None, value: float, message: str) -> None:
        if progress is not None:
            progress(value, message)

    @staticmethod
    def _emit_log(log: LogReporter | None, message: str) -> None:
        if log is not None:
            log(message)

    @staticmethod
    def _build_preview_text(segments: list) -> str:
        preview_lines: list[str] = []
        for index, segment in enumerate(segments[:5], start=1):
            preview_lines.append(f"{index}. {segment.text}")
        return "\n".join(preview_lines)

    @staticmethod
    def _build_review_flags(
        segments: list,
        options: ProcessingOptions,
        metadata,
    ) -> tuple[str, ...]:
        warnings: list[str] = []
        detected_language = metadata.detected_language

        if (
            options.source_language
            and options.source_language != "auto"
            and detected_language
            and options.source_language != detected_language
        ):
            warnings.append(
                f"Selected source language is '{options.source_language}', but Whisper detected '{detected_language}'."
            )

        if len(segments) < 3:
            warnings.append(
                "Very few subtitle segments were produced. The audio may be too short, quiet, or hard to understand."
            )

        combined_text = " ".join(segment.text for segment in segments[:12])
        if (
            options.subtitle_mode == SubtitleMode.ENGLISH
            and SubtitlePipeline._looks_mostly_greek(combined_text)
        ):
            warnings.append(
                "The English translation still contains a lot of Greek text. Review the subtitles before burning them into video."
            )

        return tuple(warnings)

    @staticmethod
    def _looks_mostly_greek(text: str) -> bool:
        greek_letters = 0
        latin_letters = 0
        for character in text:
            if "A" <= character <= "Z" or "a" <= character <= "z":
                latin_letters += 1
            elif "\u0370" <= character <= "\u03ff" or "\u1f00" <= character <= "\u1fff":
                greek_letters += 1

        if greek_letters == 0:
            return False
        return greek_letters > max(12, latin_letters)
