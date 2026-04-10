from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QThread, QTimer, Qt, Signal
from PySide6.QtGui import QCloseEvent, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ..config import (
    APP_NAME,
    APP_TAGLINE,
    DEFAULT_MAX_LINE_LENGTH,
    DEFAULT_OUTPUT_DIRECTORY,
    DEFAULT_SUBTITLE_FONT_SIZE,
    DEFAULT_WHISPER_MODEL,
    LANGUAGE_OPTIONS,
    OUTPUT_MODE_OPTIONS,
    SUBTITLE_MODE_OPTIONS,
    VIDEO_FILE_FILTER,
    WHISPER_MODEL_OPTIONS,
)
from ..models import PipelineResult, ProcessingOptions, TranscriptionResult
from ..services.pipeline import SubtitlePipeline



class TranscriptionThread(QThread):
    progress_changed = Signal(int, str)
    log_message = Signal(str)
    completed = Signal(object)  # TranscriptionResult
    failed = Signal(str)

    def __init__(
        self,
        video_path: Path,
        options: ProcessingOptions,
        file_index: int,
        total_files: int,
    ) -> None:
        super().__init__()
        self._video_path = video_path
        self._options = options
        self._file_index = file_index
        self._total_files = total_files

    def run(self) -> None:
        try:
            pipeline = SubtitlePipeline()

            def on_progress(stage_progress: float, message: str) -> None:
                overall = int(((self._file_index + stage_progress) / self._total_files) * 100)
                self.progress_changed.emit(overall, f"{self._video_path.name}: {message}")

            def on_log(message: str) -> None:
                self.log_message.emit(f"{self._video_path.name}: {message}")

            result = pipeline.transcribe(
                self._video_path,
                self._options,
                progress=on_progress,
                log=on_log,
            )
            self.completed.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class FinalizeThread(QThread):
    progress_changed = Signal(int, str)
    log_message = Signal(str)
    completed = Signal(object)  # PipelineResult
    failed = Signal(str)

    def __init__(
        self,
        transcription: TranscriptionResult,
        srt_text: str,
        options: ProcessingOptions,
        file_index: int,
        total_files: int,
    ) -> None:
        super().__init__()
        self._transcription = transcription
        self._srt_text = srt_text
        self._options = options
        self._file_index = file_index
        self._total_files = total_files

    def run(self) -> None:
        try:
            pipeline = SubtitlePipeline()
            video_name = self._transcription.input_video.name

            def on_progress(stage_progress: float, message: str) -> None:
                overall = int(((self._file_index + stage_progress) / self._total_files) * 100)
                self.progress_changed.emit(overall, f"{video_name}: {message}")

            def on_log(message: str) -> None:
                self.log_message.emit(f"{video_name}: {message}")

            result = pipeline.finalize(
                self._transcription,
                self._srt_text,
                self._options,
                progress=on_progress,
                log=on_log,
            )
            self.completed.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self._selected_files: list[Path] = []
        self._current_options: ProcessingOptions | None = None
        self._current_file_index: int = 0
        self._all_results: list[PipelineResult] = []
        self._current_transcription: TranscriptionResult | None = None
        self._transcription_thread: TranscriptionThread | None = None
        self._finalize_thread: FinalizeThread | None = None
        self._job_started_at: datetime | None = None
        self._exit_fullscreen_shortcut: QShortcut | None = None
        self._toggle_fullscreen_shortcut: QShortcut | None = None

        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(1000)
        self._elapsed_timer.timeout.connect(self._update_elapsed_label)

        self.setWindowTitle(APP_NAME)
        self.resize(1320, 820)
        self.setMinimumSize(1120, 720)

        self._build_ui()
        self._install_shortcuts()
        self._refresh_summary(None)

    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("root")
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(22, 22, 22, 22)
        root_layout.setSpacing(18)

        root_layout.addWidget(self._create_hero_card())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._create_left_column())
        splitter.addWidget(self._create_right_column())
        splitter.setSizes([640, 520])

        self._content_stack = QStackedWidget()
        self._content_stack.addWidget(splitter)                      # index 0 — normal view
        self._content_stack.addWidget(self._create_review_panel())   # index 1 — review mode
        root_layout.addWidget(self._content_stack, stretch=1)

        self.setCentralWidget(root)

    def _install_shortcuts(self) -> None:
        self._exit_fullscreen_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
        self._exit_fullscreen_shortcut.activated.connect(self._leave_fullscreen)

        self._toggle_fullscreen_shortcut = QShortcut(QKeySequence("F11"), self)
        self._toggle_fullscreen_shortcut.activated.connect(self._toggle_fullscreen)

    def _create_hero_card(self) -> QWidget:
        card = self._card()
        card.setObjectName("heroCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(26, 24, 26, 24)
        layout.setSpacing(8)

        title = QLabel(APP_NAME)
        title.setObjectName("heroTitle")

        subtitle = QLabel(
            "Built for subtitle-heavy video work, with local Whisper transcription and optional burned-in exports."
        )
        subtitle.setObjectName("heroSubtitle")
        subtitle.setWordWrap(True)

        note = QLabel(
            f"{APP_TAGLINE} Start with Greek -> English, then expand from there."
        )
        note.setObjectName("heroNote")
        note.setWordWrap(True)

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(note)
        return card

    def _create_left_column(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)
        layout.addWidget(self._create_files_card(), stretch=3)
        layout.addWidget(self._create_log_card(), stretch=2)
        return container

    def _create_right_column(self) -> QWidget:
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)
        layout.addWidget(self._create_settings_card())
        layout.addWidget(self._create_output_card())
        layout.addWidget(self._create_run_card())
        layout.addWidget(self._create_summary_card(), stretch=1)
        layout.addStretch(1)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setWidget(content)
        return scroll_area

    def _create_files_card(self) -> QWidget:
        card = self._card()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(22, 22, 22, 22)
        layout.setSpacing(14)
        layout.addWidget(self._section_title("Video Queue"))

        button_row = QHBoxLayout()
        button_row.setSpacing(10)

        self.add_videos_button = QPushButton("Add videos")
        self.add_videos_button.clicked.connect(self._choose_videos)

        self.clear_videos_button = QPushButton("Clear list")
        self.clear_videos_button.setObjectName("secondaryButton")
        self.clear_videos_button.clicked.connect(self._clear_videos)

        button_row.addWidget(self.add_videos_button)
        button_row.addWidget(self.clear_videos_button)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        self.video_list = QListWidget()
        self.video_list.setAlternatingRowColors(True)
        layout.addWidget(self.video_list, stretch=1)
        return card

    def _create_log_card(self) -> QWidget:
        card = self._card()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(22, 22, 22, 22)
        layout.setSpacing(14)
        layout.addWidget(self._section_title("Session Log"))

        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumBlockCount(800)
        self.log_output.setPlaceholderText(
            "Whisper loading, transcription, translation, and review warnings will appear here."
        )
        layout.addWidget(self.log_output, stretch=1)
        return card

    def _create_settings_card(self) -> QWidget:
        card = self._card()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(22, 22, 22, 22)
        layout.setSpacing(14)
        layout.addWidget(self._section_title("Subtitle Workflow"))

        form = QFormLayout()
        form.setHorizontalSpacing(16)
        form.setVerticalSpacing(12)

        self.source_language_combo = QComboBox()
        for value, label in LANGUAGE_OPTIONS:
            self.source_language_combo.addItem(label, value)

        self.subtitle_mode_combo = QComboBox()
        for value, label in SUBTITLE_MODE_OPTIONS:
            self.subtitle_mode_combo.addItem(label, value)

        self.whisper_model_combo = QComboBox()
        for value, label in WHISPER_MODEL_OPTIONS:
            self.whisper_model_combo.addItem(label, value)
        self._select_combo_value(self.whisper_model_combo, DEFAULT_WHISPER_MODEL)

        self.output_mode_combo = QComboBox()
        for value, label in OUTPUT_MODE_OPTIONS:
            self.output_mode_combo.addItem(label, value)

        self.max_line_length_spinbox = QSpinBox()
        self.max_line_length_spinbox.setRange(28, 60)
        self.max_line_length_spinbox.setValue(DEFAULT_MAX_LINE_LENGTH)

        self.subtitle_font_size_spinbox = QSpinBox()
        self.subtitle_font_size_spinbox.setRange(14, 28)
        self.subtitle_font_size_spinbox.setValue(DEFAULT_SUBTITLE_FONT_SIZE)

        form.addRow("Source language", self.source_language_combo)
        form.addRow("Subtitle language", self.subtitle_mode_combo)
        form.addRow("Whisper model", self.whisper_model_combo)
        form.addRow("Output mode", self.output_mode_combo)
        form.addRow("Max line length", self.max_line_length_spinbox)
        form.addRow("Burned subtitle size", self.subtitle_font_size_spinbox)
        layout.addLayout(form)

        hint = QLabel(
            "Tip: for Greek videos, start with source language set to Greek, subtitle language set to English translation, and model set to medium. The app will use a GPU automatically when CUDA or Apple Metal is available."
        )
        hint.setObjectName("supportingText")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        return card

    def _create_output_card(self) -> QWidget:
        card = self._card()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(22, 22, 22, 22)
        layout.setSpacing(14)
        layout.addWidget(self._section_title("Output"))

        self.output_directory_edit = QLineEdit(str(DEFAULT_OUTPUT_DIRECTORY))
        self.output_directory_edit.setPlaceholderText("Choose an export folder")

        row = QHBoxLayout()
        row.setSpacing(10)
        row.addWidget(self.output_directory_edit, stretch=1)

        self.browse_output_button = QPushButton("Browse")
        self.browse_output_button.setObjectName("secondaryButton")
        self.browse_output_button.clicked.connect(self._choose_output_directory)
        row.addWidget(self.browse_output_button)

        layout.addLayout(row)
        return card

    def _create_run_card(self) -> QWidget:
        card = self._card()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(22, 22, 22, 22)
        layout.setSpacing(14)
        layout.addWidget(self._section_title("Run"))

        self.status_label = QLabel("Ready to process.")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p% complete")
        layout.addWidget(self.progress_bar)

        status_grid = QGridLayout()
        status_grid.setHorizontalSpacing(14)
        status_grid.setVerticalSpacing(10)

        self.active_file_value = self._status_value("Waiting for a job")
        self.queue_value = self._status_value("No files running")
        self.engine_value = self._status_value("Automatic")
        self.elapsed_value = self._status_value("00:00")

        status_grid.addWidget(self._status_title("Active file"), 0, 0)
        status_grid.addWidget(self.active_file_value, 0, 1)
        status_grid.addWidget(self._status_title("Queue"), 1, 0)
        status_grid.addWidget(self.queue_value, 1, 1)
        status_grid.addWidget(self._status_title("Engine"), 2, 0)
        status_grid.addWidget(self.engine_value, 2, 1)
        status_grid.addWidget(self._status_title("Elapsed"), 3, 0)
        status_grid.addWidget(self.elapsed_value, 3, 1)
        layout.addLayout(status_grid)

        self.run_button = QPushButton("Generate subtitles")
        self.run_button.setObjectName("runButton")
        self.run_button.clicked.connect(self._start_processing)
        layout.addWidget(self.run_button)
        return card

    def _create_summary_card(self) -> QWidget:
        card = self._card()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(22, 22, 22, 22)
        layout.setSpacing(14)
        layout.addWidget(self._section_title("Last Result"))

        self.summary_label = QLabel()
        self.summary_label.setObjectName("summaryText")
        self.summary_label.setWordWrap(True)
        self.summary_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.summary_label)

        review_title = QLabel("Automatic Review Flags")
        review_title.setObjectName("miniTitle")
        layout.addWidget(review_title)

        self.review_flags_label = QLabel("No review flags yet.")
        self.review_flags_label.setObjectName("summaryText")
        self.review_flags_label.setWordWrap(True)
        self.review_flags_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.review_flags_label)

        preview_title = QLabel("Subtitle Preview")
        preview_title.setObjectName("miniTitle")
        layout.addWidget(preview_title)

        self.preview_output = QPlainTextEdit()
        self.preview_output.setReadOnly(True)
        self.preview_output.setMaximumBlockCount(80)
        self.preview_output.setPlaceholderText("A short subtitle preview will appear here after processing.")
        self.preview_output.setMinimumHeight(150)
        layout.addWidget(self.preview_output)
        layout.addStretch(1)
        return card

    def _create_review_panel(self) -> QWidget:
        card = self._card()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(22, 22, 22, 22)
        layout.setSpacing(14)

        header_row = QHBoxLayout()
        header_row.setSpacing(16)
        header_row.addWidget(self._section_title("Review Transcript"))

        self.review_file_label = QLabel()
        self.review_file_label.setObjectName("statusValue")
        self.review_file_label.setWordWrap(False)
        header_row.addWidget(self.review_file_label, stretch=1)

        self.review_queue_label = QLabel()
        self.review_queue_label.setObjectName("supportingText")
        header_row.addWidget(self.review_queue_label)

        layout.addLayout(header_row)

        self.srt_editor = QPlainTextEdit()
        self.srt_editor.setPlaceholderText(
            "The Whisper SRT transcript will appear here. Edit any mistakes before continuing."
        )
        layout.addWidget(self.srt_editor, stretch=1)

        self.review_warning_label = QLabel("SRT content cannot be empty.")
        self.review_warning_label.setObjectName("warningText")
        self.review_warning_label.setVisible(False)
        layout.addWidget(self.review_warning_label)

        button_row = QHBoxLayout()
        button_row.setSpacing(10)
        button_row.addStretch(1)

        self.use_original_button = QPushButton("Use Original")
        self.use_original_button.setObjectName("secondaryButton")
        self.use_original_button.clicked.connect(self._on_use_original_clicked)

        self.approve_button = QPushButton("Approve & Continue")
        self.approve_button.setObjectName("runButton")
        self.approve_button.clicked.connect(self._on_approve_clicked)

        button_row.addWidget(self.use_original_button)
        button_row.addWidget(self.approve_button)
        layout.addLayout(button_row)

        return card

    def _choose_videos(self) -> None:
        selected_files, _ = QFileDialog.getOpenFileNames(
            self,
            "Choose videos",
            str(Path.cwd()),
            VIDEO_FILE_FILTER,
        )
        if not selected_files:
            return

        existing_paths = {path.resolve() for path in self._selected_files}
        added_count = 0
        for file_name in selected_files:
            path = Path(file_name).expanduser().resolve()
            if path in existing_paths:
                continue
            existing_paths.add(path)
            self._selected_files.append(path)
            item = QListWidgetItem(path.name)
            item.setToolTip(str(path))
            self.video_list.addItem(item)
            added_count += 1

        if added_count:
            self._append_log(f"Queued {added_count} new video(s).")
        self._refresh_summary(None)

    def _clear_videos(self) -> None:
        self._selected_files.clear()
        self.video_list.clear()
        self._append_log("Cleared the current video queue.")
        self._refresh_summary(None)

    def _choose_output_directory(self) -> None:
        chosen_directory = QFileDialog.getExistingDirectory(
            self,
            "Choose an output folder",
            self.output_directory_edit.text() or str(DEFAULT_OUTPUT_DIRECTORY),
        )
        if chosen_directory:
            self.output_directory_edit.setText(chosen_directory)

    def _start_processing(self) -> None:
        transcribing = self._transcription_thread is not None and self._transcription_thread.isRunning()
        finalizing = self._finalize_thread is not None and self._finalize_thread.isRunning()
        if transcribing or finalizing:
            return

        if not self._selected_files:
            QMessageBox.warning(self, APP_NAME, "Add at least one video before starting.")
            return

        output_directory_text = self.output_directory_edit.text().strip()
        if not output_directory_text:
            QMessageBox.warning(self, APP_NAME, "Choose an output directory first.")
            return

        self._current_options = ProcessingOptions(
            source_language=self.source_language_combo.currentData(),
            subtitle_mode=self.subtitle_mode_combo.currentData(),
            whisper_model=self.whisper_model_combo.currentData(),
            output_mode=self.output_mode_combo.currentData(),
            output_directory=Path(output_directory_text),
            max_line_length=self.max_line_length_spinbox.value(),
            subtitle_font_size=self.subtitle_font_size_spinbox.value(),
        )
        self._current_file_index = 0
        self._all_results = []

        self.progress_bar.setValue(0)
        self.status_label.setText("Starting subtitle pipeline...")
        self.active_file_value.setText("Preparing queue")
        self.queue_value.setText(f"0 of {len(self._selected_files)} finished")
        self.engine_value.setText("Detecting best device")
        self.review_flags_label.setText("Automatic review flags will appear after transcription.")
        self.preview_output.setPlainText("")
        self._job_started_at = datetime.now()
        self._elapsed_timer.start()
        self._update_elapsed_label()
        self._set_busy(True)

        self._start_next_file()

    def _start_next_file(self) -> None:
        assert self._current_options is not None
        video_path = self._selected_files[self._current_file_index]
        total = len(self._selected_files)

        self.active_file_value.setText(video_path.name)
        self.queue_value.setText(f"File {self._current_file_index + 1} of {total}")
        self._append_log(f"[{self._current_file_index + 1}/{total}] Starting {video_path.name}")

        thread = TranscriptionThread(
            video_path,
            self._current_options,
            self._current_file_index,
            total,
        )
        thread.progress_changed.connect(self._on_progress)
        thread.log_message.connect(self._append_log)
        thread.completed.connect(self._on_transcription_completed)
        thread.failed.connect(self._on_failed)
        thread.finished.connect(lambda t=thread: self._on_thread_finished(t))
        self._transcription_thread = thread
        thread.start()

    def _on_transcription_completed(self, result: TranscriptionResult) -> None:
        self._current_transcription = result
        total = len(self._selected_files)

        for warning in result.warning_messages:
            self._append_log(f"{result.input_video.name}: Review flag: {warning}")

        self.review_file_label.setText(result.input_video.name)
        self.review_queue_label.setText(f"File {self._current_file_index + 1} of {total}")
        self.srt_editor.setPlainText(result.srt_text)
        self.review_warning_label.setVisible(False)
        self.status_label.setText(f"Review transcript for {result.input_video.name}")
        self._content_stack.setCurrentIndex(1)

    def _on_approve_clicked(self) -> None:
        srt_text = self.srt_editor.toPlainText().strip()
        if not srt_text:
            self.review_warning_label.setVisible(True)
            return
        self._start_finalize(srt_text)

    def _on_use_original_clicked(self) -> None:
        assert self._current_transcription is not None
        self._start_finalize(self._current_transcription.srt_text)

    def _start_finalize(self, srt_text: str) -> None:
        assert self._current_options is not None
        assert self._current_transcription is not None

        self._content_stack.setCurrentIndex(0)
        self.status_label.setText(
            f"Writing subtitles for {self._current_transcription.input_video.name}..."
        )

        total = len(self._selected_files)
        thread = FinalizeThread(
            self._current_transcription,
            srt_text,
            self._current_options,
            self._current_file_index,
            total,
        )
        thread.progress_changed.connect(self._on_progress)
        thread.log_message.connect(self._append_log)
        thread.completed.connect(self._on_finalize_completed)
        thread.failed.connect(self._on_failed)
        thread.finished.connect(lambda t=thread: self._on_thread_finished(t))
        self._finalize_thread = thread
        self._current_transcription = None
        thread.start()

    def _on_finalize_completed(self, result: PipelineResult) -> None:
        self._all_results.append(result)
        self._refresh_summary(result)
        self._current_file_index += 1
        if self._current_file_index < len(self._selected_files):
            self._start_next_file()
        else:
            self._on_all_done()

    def _on_all_done(self) -> None:
        self._set_busy(False)
        self._elapsed_timer.stop()
        self._job_started_at = None
        self.status_label.setText("All subtitle jobs finished.")
        self.progress_bar.setValue(100)
        self._append_log(f"Finished all {len(self._all_results)} queued videos.")
        self.queue_value.setText(f"{len(self._all_results)} of {len(self._selected_files)} finished")

    def _on_progress(self, value: int, message: str) -> None:
        self.progress_bar.setValue(value)
        self.status_label.setText(message)

    def _on_failed(self, error_message: str) -> None:
        self._set_busy(False)
        self._elapsed_timer.stop()
        self._job_started_at = None
        self._content_stack.setCurrentIndex(0)
        self.status_label.setText("The subtitle pipeline hit an error.")
        self._append_log(error_message)
        self.review_flags_label.setText(
            "Processing stopped with an error. Check the session log for the exact FFmpeg or Whisper failure."
        )
        QMessageBox.critical(self, APP_NAME, error_message)

    def _on_thread_finished(self, thread: QThread) -> None:
        if self._transcription_thread is thread:
            self._transcription_thread = None
        elif self._finalize_thread is thread:
            self._finalize_thread = None
        thread.deleteLater()

    def _refresh_summary(self, result: PipelineResult | None) -> None:
        if result is None:
            if not self._selected_files:
                self.summary_label.setText(
                    "Add videos on the left, choose your subtitle workflow, and the app will export SRT files plus optional burned-in videos."
                )
                self.review_flags_label.setText("No review flags yet.")
                self.preview_output.setPlainText("")
            else:
                self.summary_label.setText(
                    f"{len(self._selected_files)} video(s) queued.\n"
                    f"Output folder: {self.output_directory_edit.text() or DEFAULT_OUTPUT_DIRECTORY}"
                )
                self.review_flags_label.setText(
                    "No review flags yet. They will appear here after Whisper finishes."
                )
            return

        burned_video_text = str(result.burned_video) if result.burned_video else "Not generated"
        self.summary_label.setText(
            f"Finished: {result.input_video.name}\n"
            f"Detected language: {result.detected_language or 'unknown'}\n"
            f"Engine: {result.device_label or 'unknown'}\n"
            f"Subtitle file: {result.subtitle_file}\n"
            f"Burned video: {burned_video_text}\n"
            f"Segments: {result.segment_count}\n"
            f"Elapsed: {result.elapsed_seconds:.1f}s"
        )
        self.engine_value.setText(result.device_label or "Unknown")
        if result.warning_messages:
            self.review_flags_label.setText("\n".join(f"- {item}" for item in result.warning_messages))
        else:
            self.review_flags_label.setText(
                "No automatic review flags. You should still skim the subtitle preview before publishing."
            )
        self.preview_output.setPlainText(result.preview_text or "No preview available.")

    def _set_busy(self, busy: bool) -> None:
        controls = [
            self.add_videos_button,
            self.clear_videos_button,
            self.browse_output_button,
            self.run_button,
            self.source_language_combo,
            self.subtitle_mode_combo,
            self.whisper_model_combo,
            self.output_mode_combo,
            self.max_line_length_spinbox,
            self.subtitle_font_size_spinbox,
            self.output_directory_edit,
        ]
        for control in controls:
            control.setEnabled(not busy)

    def _append_log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_output.appendPlainText(f"[{timestamp}] {message}")
        if "Loading Whisper model" in message and " on " in message:
            self.engine_value.setText(message.rsplit(" on ", 1)[1])

    def _update_elapsed_label(self) -> None:
        if self._job_started_at is None:
            self.elapsed_value.setText("00:00")
            return

        elapsed = datetime.now() - self._job_started_at
        total_seconds = int(elapsed.total_seconds())
        minutes, seconds = divmod(total_seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            self.elapsed_value.setText(f"{hours:02d}:{minutes:02d}:{seconds:02d}")
        else:
            self.elapsed_value.setText(f"{minutes:02d}:{seconds:02d}")

    def _leave_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showNormal()

    def _toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def closeEvent(self, event: QCloseEvent) -> None:
        running = any(
            t is not None and t.isRunning()
            for t in (self._transcription_thread, self._finalize_thread)
        )
        if running:
            QMessageBox.information(
                self,
                APP_NAME,
                "Subtitle generation is still running. Wait for the current job to finish before closing the app.",
            )
            event.ignore()
            return

        for thread in (self._transcription_thread, self._finalize_thread):
            if thread is not None:
                thread.wait(1000)
        super().closeEvent(event)

    @staticmethod
    def _card() -> QFrame:
        card = QFrame()
        card.setObjectName("card")
        return card

    @staticmethod
    def _section_title(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("sectionTitle")
        return label

    @staticmethod
    def _status_title(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("statusTitle")
        return label

    @staticmethod
    def _status_value(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("statusValue")
        label.setWordWrap(True)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        return label

    @staticmethod
    def _select_combo_value(combo_box: QComboBox, value: object) -> None:
        for index in range(combo_box.count()):
            if combo_box.itemData(index) == value:
                combo_box.setCurrentIndex(index)
                return
