from __future__ import annotations

import shutil
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication, QListWidgetItem

from add_subtitles_to_videos.models import (
    SubtitleSegment,
    TranscriptionMetadata,
    TranscriptionResult,
    TranslationSegment,
)
from add_subtitles_to_videos.ui.main_window import MainWindow
from add_subtitles_to_videos.ui.theme import application_stylesheet

ROOT = Path(__file__).resolve().parents[1]
SCREENSHOT_DIR = ROOT / "docs" / "screenshots"


def main() -> int:
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName("Subtitle Foundry")
    app.setOrganizationName("Subtitle Foundry")
    app.setStyle("Fusion")
    app.setStyleSheet(application_stylesheet())

    _render_main_dashboard()
    _render_review_editor()
    drafts_dir = SCREENSHOT_DIR / "_drafts"
    if drafts_dir.exists():
        shutil.rmtree(drafts_dir)
    return 0


def _render_main_dashboard() -> None:
    window = MainWindow()
    window.resize(1520, 940)

    sample_files = [
        ROOT / "exports" / "one.mp4",
        ROOT / "exports" / "two.mp4",
        ROOT / "exports" / "three.mp4",
    ]
    window._selected_files = sample_files
    window.video_list.clear()
    for path in sample_files:
        item = QListWidgetItem(path.name)
        item.setToolTip(str(path))
        window.video_list.addItem(item)

    window.translation_base_url_edit.setText("https://api.openai.com/v1")
    window.translation_model_edit.setText("gpt-5.4")
    window.translation_api_key_edit.setText("sk-live-*******************************")
    window.output_directory_edit.setText(str(ROOT / "exports"))
    window._refresh_translation_status()
    window._refresh_summary(None)

    window.status_label.setText("interview-one.mp4: Translating subtitle text")
    window.progress_bar.setValue(63)
    window.active_file_value.setText("interview-one.mp4")
    window.queue_value.setText("File 1 of 3")
    window.engine_value.setText("CUDA")
    window.gpu_value.setText("NVIDIA GeForce RTX 4070 | 81% util | 64C")
    window.vram_value.setText("6120/12282 MiB used")
    window.elapsed_value.setText("03:42")
    window.summary_label.setText(
        "3 video(s) queued.\n"
        "Target language: English\n"
        f"Output folder: {ROOT / 'exports'}"
    )
    window.review_flags_label.setText(
        "- 2 subtitle block(s) exceed the preferred line length.\n"
        "- Review translation terminology in the final section."
    )
    window.preview_output.setPlainText(
        "1. When I'm stressed, I mostly don't talk...\n"
        "2. Children are like plasticine.\n"
        "3. Bullying must be distinguished from conflict."
    )
    window._append_log("Queued 3 new video(s).")
    window._append_log("Whisper warmup complete for model 'large-v3'.")
    window._append_log("interview-one.mp4: Translation batch 2 finished successfully.")

    _show_and_capture(window, SCREENSHOT_DIR / "main-dashboard.png")


def _render_review_editor() -> None:
    window = MainWindow()
    window.resize(1520, 940)
    window._review_drafts_directory = lambda: SCREENSHOT_DIR / "_drafts"  # type: ignore[method-assign]

    sample_video = ROOT / "exports" / "interview-one.mp4"
    window._selected_files = [sample_video]
    result = _sample_transcription_result(sample_video)
    window._show_review(result, 0)

    window.review_flags_label.setText(
        "- Missing subtitle text was added between existing timed blocks.\n"
        "- Leave music-only sections blank when no caption is needed."
    )
    window.review_warning_label.setVisible(True)
    window.review_warning_label.setText(
        "Keep every original subtitle block and timestamp, but you can insert extra blocks for missed subtitles or leave a block blank when no caption is needed."
    )

    _show_and_capture(window, SCREENSHOT_DIR / "review-editor.png")


def _sample_transcription_result(video_path: Path) -> TranscriptionResult:
    source_segments = [
        SubtitleSegment(0.0, 2.1, "Γιατί τα παιδιά είναι σαν την πλαστελίνη."),
        SubtitleSegment(2.4, 5.3, "Θέλουν χώρο να χαλαρώσουν και να δημιουργήσουν."),
        SubtitleSegment(5.6, 7.1, "Κάποιες φορές η μουσική δεν χρειάζεται υπότιτλο."),
    ]
    review_segments = [
        TranslationSegment(0.0, 2.1, source_segments[0].text, "Because children are like plasticine."),
        TranslationSegment(
            2.4,
            5.3,
            source_segments[1].text,
            "They need space to relax and create.",
        ),
        TranslationSegment(
            5.6,
            7.1,
            source_segments[2].text,
            "",
        ),
    ]

    source_srt_text = (
        "1\n00:00:00,000 --> 00:00:02,100\nΓιατί τα παιδιά είναι σαν την πλαστελίνη.\n\n"
        "2\n00:00:02,400 --> 00:00:05,300\nΘέλουν χώρο να χαλαρώσουν και να δημιουργήσουν.\n\n"
        "3\n00:00:05,300 --> 00:00:05,600\n[μουσική]\n\n"
        "4\n00:00:05,600 --> 00:00:07,100\nΚάποιες φορές η μουσική δεν χρειάζεται υπότιτλο.\n"
    )
    translated_srt_text = (
        "1\n00:00:00,000 --> 00:00:02,100\nBecause children are like plasticine.\n\n"
        "2\n00:00:02,100 --> 00:00:02,400\n[Add missing subtitle text here]\n\n"
        "3\n00:00:02,400 --> 00:00:05,300\nThey need space to relax and create.\n\n"
        "4\n00:00:05,300 --> 00:00:05,600\n\n"
        "5\n00:00:05,600 --> 00:00:07,100\nSometimes music does not need a subtitle.\n"
    )

    return TranscriptionResult(
        input_video=video_path,
        source_segments=source_segments,
        review_segments=review_segments,
        metadata=TranscriptionMetadata(
            detected_language="el",
            detected_language_probability=0.99,
            duration_seconds=421.0,
            device_label="CUDA",
            task_label="transcribe",
            translation_provider="openai_compatible",
            target_language="en",
            translation_applied=True,
            stage_durations={
                "audio_extraction_seconds": 1.7,
                "whisper_seconds": 28.4,
                "translation_seconds": 7.2,
            },
        ),
        warning_messages=(
            "A missing subtitle gap was inserted for review.",
            "One music-only section is intentionally blank.",
        ),
        source_srt_text=source_srt_text,
        translated_srt_text=translated_srt_text,
    )


def _show_and_capture(window: MainWindow, output_path: Path) -> None:
    app = QApplication.instance()
    assert app is not None
    window.show()
    app.processEvents()
    pixmap = window.grab()
    pixmap.save(str(output_path))
    if getattr(window, "_current_transcription", None) is not None:
        window._current_transcription = None
        window._review_started_at = None
        window._clear_active_review_draft_state()
        window._clear_prefetch_state()
    window.hide()
    window.deleteLater()
    app.processEvents()


if __name__ == "__main__":
    raise SystemExit(main())
