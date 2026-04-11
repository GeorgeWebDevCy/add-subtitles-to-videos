from __future__ import annotations

from datetime import datetime
from pathlib import Path
from time import monotonic

from PySide6.QtCore import QSettings, QThread, QTimer, Qt, Signal
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
    DEFAULT_OUTPUT_MODE,
    DEFAULT_OUTPUT_DIRECTORY,
    DEFAULT_SOURCE_LANGUAGE,
    DEFAULT_SUBTITLE_FONT_SIZE,
    DEFAULT_TARGET_LANGUAGE,
    DEFAULT_TRANSLATION_BASE_URL,
    DEFAULT_TRANSLATION_MODEL,
    DEFAULT_TRANSLATION_PROVIDER,
    DEFAULT_WHISPER_MODEL,
    OUTPUT_MODE_OPTIONS,
    SOURCE_LANGUAGE_OPTIONS,
    TARGET_LANGUAGE_OPTIONS,
    TRANSLATION_PROVIDER_OPTIONS,
    VIDEO_FILE_FILTER,
    WHISPER_MODEL_OPTIONS,
)
from ..languages import language_label
from ..models import OutputMode, PipelineResult, ProcessingOptions, SubtitleSegment, TranscriptionResult
from ..services import OperationCancelledError
from ..services.pipeline import SubtitlePipeline
from ..services.subtitles import validate_review_srt_text
from ..services.translation import (
    OpenAICompatibleTranslationService,
    TranslationProviderConfig,
    TranslationServiceError,
)
from ..services.whisper_worker import WhisperWorkerClient

TRANSCRIPTION_WORKER = WhisperWorkerClient()


class TranscriptionThread(QThread):
    progress_changed = Signal(int, str)
    log_message = Signal(str)
    completed = Signal(object)
    cancelled = Signal(str)
    failed = Signal(str)

    def __init__(
        self,
        video_path: Path,
        options: ProcessingOptions,
        translation_service: OpenAICompatibleTranslationService | None,
        file_index: int,
        total_files: int,
    ) -> None:
        super().__init__()
        self._video_path = video_path
        self._options = options
        self._translation_service = translation_service
        self._file_index = file_index
        self._total_files = total_files

    def run(self) -> None:
        try:
            pipeline = SubtitlePipeline(
                whisper_service=TRANSCRIPTION_WORKER,
                translation_service=self._translation_service,
            )

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
                cancel_requested=self.isInterruptionRequested,
            )
            self.completed.emit(result)
        except OperationCancelledError as exc:
            self.cancelled.emit(str(exc))
        except Exception as exc:
            self.failed.emit(str(exc))

    def request_stop(self) -> None:
        self.requestInterruption()


class FinalizeThread(QThread):
    progress_changed = Signal(int, str)
    log_message = Signal(str)
    completed = Signal(object)
    cancelled = Signal(str)
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
                cancel_requested=self.isInterruptionRequested,
            )
            self.completed.emit(result)
        except OperationCancelledError as exc:
            self.cancelled.emit(str(exc))
        except Exception as exc:
            self.failed.emit(str(exc))

    def request_stop(self) -> None:
        self.requestInterruption()


class ModelPreloadThread(QThread):
    log_message = Signal(str)
    completed = Signal(str)
    failed = Signal(str)

    def __init__(self, model_name: str) -> None:
        super().__init__()
        self._model_name = model_name

    def run(self) -> None:
        try:
            TRANSCRIPTION_WORKER.preload_model(self._model_name, log=self.log_message.emit)
            self.completed.emit(self._model_name)
        except Exception as exc:
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self._settings = QSettings("Subtitle Foundry", APP_NAME)
        self._selected_files: list[Path] = []
        self._current_options: ProcessingOptions | None = None
        self._current_translation_service: OpenAICompatibleTranslationService | None = None
        self._current_file_index = 0
        self._all_results: list[PipelineResult] = []
        self._current_transcription: TranscriptionResult | None = None
        self._transcription_thread: TranscriptionThread | None = None
        self._finalize_thread: FinalizeThread | None = None
        self._preload_thread: ModelPreloadThread | None = None
        self._preloaded_model_name: str | None = None
        self._prefetched_transcription: TranscriptionResult | None = None
        self._prefetched_file_index: int | None = None
        self._waiting_for_prefetched_review = False
        self._review_started_at: float | None = None
        self._active_transcription_index: int | None = None
        self._active_transcription_is_prefetch = False
        self._job_started_at: datetime | None = None
        self._stop_requested = False
        self._exit_fullscreen_shortcut: QShortcut | None = None
        self._toggle_fullscreen_shortcut: QShortcut | None = None

        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(1000)
        self._elapsed_timer.timeout.connect(self._update_elapsed_label)

        self.setWindowTitle(APP_NAME)
        self.resize(1420, 860)
        self.setMinimumSize(1180, 760)

        self._build_ui()
        self._install_shortcuts()
        self._load_provider_settings()
        self._refresh_translation_status()
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
        splitter.setSizes([620, 560])

        self._content_stack = QStackedWidget()
        self._content_stack.addWidget(splitter)
        self._content_stack.addWidget(self._create_review_panel())
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
            "Local Whisper handles speech recognition, then an online translation layer converts subtitle text into the target language before mandatory review."
        )
        subtitle.setObjectName("heroSubtitle")
        subtitle.setWordWrap(True)

        note = QLabel(
            f"{APP_TAGLINE} Europe-focused launch set, mandatory review, and optional burned-in exports."
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
        self.log_output.setObjectName("consolePanel")
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumBlockCount(1000)
        self.log_output.setPlaceholderText(
            "Whisper, translation, validation, and export messages will appear here."
        )
        layout.addWidget(self.log_output, stretch=1)
        return card

    def _create_settings_card(self) -> QWidget:
        card = self._card()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(22, 22, 22, 22)
        layout.setSpacing(14)
        layout.addWidget(self._section_title("Workflow"))

        form = QFormLayout()
        form.setHorizontalSpacing(16)
        form.setVerticalSpacing(12)

        self.source_language_combo = QComboBox()
        for value, label in SOURCE_LANGUAGE_OPTIONS:
            self.source_language_combo.addItem(label, value)
        self._select_combo_value(self.source_language_combo, DEFAULT_SOURCE_LANGUAGE)
        self.source_language_combo.currentIndexChanged.connect(self._refresh_translation_status)

        self.target_language_combo = QComboBox()
        for value, label in TARGET_LANGUAGE_OPTIONS:
            self.target_language_combo.addItem(label, value)
        self._select_combo_value(self.target_language_combo, DEFAULT_TARGET_LANGUAGE)
        self.target_language_combo.currentIndexChanged.connect(self._refresh_translation_status)

        self.translation_provider_combo = QComboBox()
        for value, label in TRANSLATION_PROVIDER_OPTIONS:
            self.translation_provider_combo.addItem(label, value)
        self._select_combo_value(self.translation_provider_combo, DEFAULT_TRANSLATION_PROVIDER)
        self.translation_provider_combo.currentIndexChanged.connect(self._refresh_translation_status)

        self.whisper_model_combo = QComboBox()
        for value, label in WHISPER_MODEL_OPTIONS:
            self.whisper_model_combo.addItem(label, value)
        self._select_combo_value(self.whisper_model_combo, DEFAULT_WHISPER_MODEL)
        self.whisper_model_combo.currentIndexChanged.connect(self._on_whisper_model_changed)

        self.output_mode_combo = QComboBox()
        for value, label in OUTPUT_MODE_OPTIONS:
            self.output_mode_combo.addItem(label, value)
        self._select_combo_value(self.output_mode_combo, DEFAULT_OUTPUT_MODE)

        self.max_line_length_spinbox = QSpinBox()
        self.max_line_length_spinbox.setRange(28, 60)
        self.max_line_length_spinbox.setValue(DEFAULT_MAX_LINE_LENGTH)

        self.subtitle_font_size_spinbox = QSpinBox()
        self.subtitle_font_size_spinbox.setRange(14, 28)
        self.subtitle_font_size_spinbox.setValue(DEFAULT_SUBTITLE_FONT_SIZE)

        form.addRow("Source language", self.source_language_combo)
        form.addRow("Target language", self.target_language_combo)
        form.addRow("Translation provider", self.translation_provider_combo)
        form.addRow("Whisper model", self.whisper_model_combo)
        form.addRow("Output mode", self.output_mode_combo)
        form.addRow("Max line length", self.max_line_length_spinbox)
        form.addRow("Burned subtitle size", self.subtitle_font_size_spinbox)
        layout.addLayout(form)

        provider_title = QLabel("Online Translation Settings")
        provider_title.setObjectName("miniTitle")
        layout.addWidget(provider_title)

        provider_form = QFormLayout()
        provider_form.setHorizontalSpacing(16)
        provider_form.setVerticalSpacing(12)

        self.translation_base_url_edit = QLineEdit()
        self.translation_base_url_edit.setPlaceholderText("https://api.openai.com/v1")
        self.translation_base_url_edit.editingFinished.connect(self._persist_provider_settings)
        self.translation_base_url_edit.textChanged.connect(self._refresh_translation_status)

        self.translation_model_edit = QLineEdit()
        self.translation_model_edit.setPlaceholderText("Model name")
        self.translation_model_edit.editingFinished.connect(self._persist_provider_settings)
        self.translation_model_edit.textChanged.connect(self._refresh_translation_status)

        self.translation_api_key_edit = QLineEdit()
        self.translation_api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.translation_api_key_edit.setPlaceholderText("API key")
        self.translation_api_key_edit.editingFinished.connect(self._persist_provider_settings)
        self.translation_api_key_edit.textChanged.connect(self._refresh_translation_status)

        provider_form.addRow("Base URL", self.translation_base_url_edit)
        provider_form.addRow("Model", self.translation_model_edit)
        provider_form.addRow("API key", self.translation_api_key_edit)
        layout.addLayout(provider_form)

        provider_actions = QHBoxLayout()
        provider_actions.setSpacing(10)

        self.save_translation_settings_button = QPushButton("Save API Settings")
        self.save_translation_settings_button.setObjectName("secondaryButton")
        self.save_translation_settings_button.clicked.connect(self._save_translation_settings)

        self.test_translation_button = QPushButton("Test Connection")
        self.test_translation_button.setObjectName("secondaryButton")
        self.test_translation_button.clicked.connect(self._test_translation_connection)

        provider_actions.addWidget(self.save_translation_settings_button)
        provider_actions.addWidget(self.test_translation_button)
        provider_actions.addStretch(1)
        layout.addLayout(provider_actions)

        self.translation_status_label = QLabel()
        self.translation_status_label.setObjectName("summaryText")
        self.translation_status_label.setWordWrap(True)
        layout.addWidget(self.translation_status_label)

        hint = QLabel(
            "Launch set: English, Greek, Turkish, German, French, Italian, Spanish, Portuguese, Dutch, Romanian, Polish, and Czech. Review is always required before export."
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

        self.run_button = QPushButton("Transcribe and review")
        self.run_button.setObjectName("runButton")
        self.run_button.clicked.connect(self._start_processing)

        self.stop_button = QPushButton("Stop")
        self.stop_button.setObjectName("secondaryButton")
        self.stop_button.clicked.connect(self._request_stop)
        self.stop_button.setEnabled(False)

        action_row = QHBoxLayout()
        action_row.addWidget(self.run_button, stretch=1)
        action_row.addWidget(self.stop_button)
        layout.addLayout(action_row)
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

        review_title = QLabel("Review Flags")
        review_title.setObjectName("miniTitle")
        layout.addWidget(review_title)

        self.review_flags_label = QLabel("No review flags yet.")
        self.review_flags_label.setObjectName("summaryText")
        self.review_flags_label.setWordWrap(True)
        self.review_flags_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.review_flags_label)

        preview_title = QLabel("Translated Preview")
        preview_title.setObjectName("miniTitle")
        layout.addWidget(preview_title)

        self.preview_output = QPlainTextEdit()
        self.preview_output.setObjectName("previewPanel")
        self.preview_output.setReadOnly(True)
        self.preview_output.setMaximumBlockCount(80)
        self.preview_output.setPlaceholderText("A translated subtitle preview will appear here after processing.")
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
        header_row.addWidget(self._section_title("Review Translation"))

        self.review_file_label = QLabel()
        self.review_file_label.setObjectName("statusValue")
        header_row.addWidget(self.review_file_label, stretch=1)

        self.review_queue_label = QLabel()
        self.review_queue_label.setObjectName("supportingText")
        header_row.addWidget(self.review_queue_label)
        layout.addLayout(header_row)

        self.review_summary_label = QLabel(
            "Detected source language, target language, provider, and warnings will appear here once processing finishes."
        )
        self.review_summary_label.setObjectName("reviewSummary")
        self.review_summary_label.setWordWrap(True)
        self.review_summary_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.review_summary_label)

        transcript_splitter = QSplitter(Qt.Orientation.Horizontal)
        transcript_splitter.setChildrenCollapsible(False)

        source_card = self._card()
        source_layout = QVBoxLayout(source_card)
        source_layout.setContentsMargins(16, 16, 16, 16)
        source_layout.setSpacing(10)
        source_title = QLabel("Source Transcript")
        source_title.setObjectName("miniTitle")
        source_layout.addWidget(source_title)

        self.source_srt_view = QPlainTextEdit()
        self.source_srt_view.setObjectName("sourceTranscriptPanel")
        self.source_srt_view.setReadOnly(True)
        self.source_srt_view.setPlaceholderText("Source-language SRT will appear here.")
        source_layout.addWidget(self.source_srt_view, stretch=1)
        transcript_splitter.addWidget(source_card)

        translated_card = self._card()
        translated_layout = QVBoxLayout(translated_card)
        translated_layout.setContentsMargins(16, 16, 16, 16)
        translated_layout.setSpacing(10)
        translated_title = QLabel("Translated Subtitle Draft")
        translated_title.setObjectName("miniTitle")
        translated_layout.addWidget(translated_title)

        self.translated_srt_editor = QPlainTextEdit()
        self.translated_srt_editor.setObjectName("translatedTranscriptPanel")
        self.translated_srt_editor.setPlaceholderText(
            "Translated SRT text will appear here. You can edit wording, but timing must stay unchanged."
        )
        translated_layout.addWidget(self.translated_srt_editor, stretch=1)
        transcript_splitter.addWidget(translated_card)
        transcript_splitter.setSizes([640, 700])

        layout.addWidget(transcript_splitter, stretch=1)

        self.review_warning_label = QLabel(
            "Translated subtitles must keep the same number of subtitle blocks and timestamps as the source transcript."
        )
        self.review_warning_label.setObjectName("warningText")
        self.review_warning_label.setVisible(False)
        layout.addWidget(self.review_warning_label)

        button_row = QHBoxLayout()
        button_row.addStretch(1)

        self.use_original_button = QPushButton("Use Original Draft")
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

        translation_service = self._build_translation_service()
        source_language = self.source_language_combo.currentData()
        if (
            self._translation_required_from_choices()
            and source_language != "auto"
            and not translation_service.is_configured()
        ):
            QMessageBox.warning(self, APP_NAME, translation_service.configuration_status())
            return

        self._current_translation_service = translation_service if translation_service.is_configured() else None
        self._current_options = ProcessingOptions(
            source_language=self.source_language_combo.currentData(),
            target_language=self.target_language_combo.currentData(),
            translation_provider=self.translation_provider_combo.currentData(),
            whisper_model=self.whisper_model_combo.currentData(),
            output_mode=OutputMode(str(self.output_mode_combo.currentData())),
            output_directory=Path(output_directory_text),
            max_line_length=self.max_line_length_spinbox.value(),
            subtitle_font_size=self.subtitle_font_size_spinbox.value(),
        )
        self._current_file_index = 0
        self._all_results = []
        self._stop_requested = False
        self._current_transcription = None
        self._review_started_at = None
        self._clear_prefetch_state()

        self.progress_bar.setValue(0)
        self.status_label.setText("Starting subtitle pipeline...")
        self.active_file_value.setText("Preparing queue")
        self.queue_value.setText(f"0 of {len(self._selected_files)} finished")
        self.engine_value.setText("Detecting best device")
        self.review_flags_label.setText("Review flags will appear after processing.")
        self.preview_output.setPlainText("")
        self._job_started_at = datetime.now()
        self._elapsed_timer.start()
        self._update_elapsed_label()
        self._set_busy(True)
        self._schedule_model_preload()
        self._start_transcription(self._current_file_index, as_prefetch=False)

    def _start_transcription(self, file_index: int, *, as_prefetch: bool) -> None:
        assert self._current_options is not None
        video_path = self._selected_files[file_index]
        total = len(self._selected_files)

        if as_prefetch:
            self._append_log(
                f"[{file_index + 1}/{total}] Preparing {video_path.name} in the background during review."
            )
        else:
            self.active_file_value.setText(video_path.name)
            self.queue_value.setText(f"File {file_index + 1} of {total}")
            self._append_log(f"[{file_index + 1}/{total}] Starting {video_path.name}")
            self.stop_button.setEnabled(True)

        thread = TranscriptionThread(
            video_path,
            self._current_options,
            self._current_translation_service,
            file_index,
            total,
        )
        if not as_prefetch:
            thread.progress_changed.connect(self._on_progress)
        thread.log_message.connect(self._append_log)
        if as_prefetch:
            thread.completed.connect(
                lambda result, index=file_index: self._on_prefetch_completed(result, index)
            )
        else:
            thread.completed.connect(
                lambda result, index=file_index: self._on_transcription_completed(result, index)
            )
        thread.cancelled.connect(self._on_cancelled)
        thread.failed.connect(self._on_failed)
        thread.finished.connect(lambda t=thread: self._on_thread_finished(t))
        self._transcription_thread = thread
        self._active_transcription_index = file_index
        self._active_transcription_is_prefetch = as_prefetch
        thread.start()

    def _on_transcription_completed(self, result: TranscriptionResult, file_index: int) -> None:
        self._waiting_for_prefetched_review = False
        self._show_review(result, file_index)

    def _on_prefetch_completed(self, result: TranscriptionResult, file_index: int) -> None:
        self._prefetched_transcription = result
        self._prefetched_file_index = file_index
        if self._current_transcription is None and file_index == self._current_file_index:
            self._waiting_for_prefetched_review = False
            self._show_review(result, file_index)
            self._clear_prefetch_state()
            return

        self._append_log(
            f"[{file_index + 1}/{len(self._selected_files)}] {result.input_video.name} is ready for review."
        )
        self._refresh_review_queue_label()

    def _show_review(self, result: TranscriptionResult, file_index: int) -> None:
        self._current_transcription = result
        self._current_file_index = file_index
        self._review_started_at = monotonic()
        total = len(self._selected_files)

        for warning in result.warning_messages:
            self._append_log(f"{result.input_video.name}: Review flag: {warning}")

        self.review_file_label.setText(result.input_video.name)
        self.review_summary_label.setText(self._review_summary_text(result))
        self.source_srt_view.setPlainText(result.source_srt_text)
        self.translated_srt_editor.setPlainText(result.translated_srt_text)
        self.review_warning_label.setVisible(False)
        self.status_label.setText(f"Review translated subtitles for {result.input_video.name}")
        self.stop_button.setEnabled(True)
        self._content_stack.setCurrentIndex(1)
        self._refresh_review_queue_label()
        self._maybe_start_prefetch()

    def _on_approve_clicked(self) -> None:
        assert self._current_transcription is not None
        srt_text = self.translated_srt_editor.toPlainText().strip()
        if not srt_text:
            self.review_warning_label.setText("Translated subtitles cannot be empty.")
            self.review_warning_label.setVisible(True)
            return

        error_message = validate_review_srt_text(srt_text, self._current_transcription.review_segments)
        if error_message is not None:
            self.review_warning_label.setText(error_message)
            self.review_warning_label.setVisible(True)
            return

        self._start_finalize(srt_text + "\n")

    def _on_use_original_clicked(self) -> None:
        assert self._current_transcription is not None
        self._start_finalize(self._current_transcription.translated_srt_text)

    def _start_finalize(self, srt_text: str) -> None:
        assert self._current_options is not None
        assert self._current_transcription is not None
        review_wait_seconds = 0.0
        if self._review_started_at is not None:
            review_wait_seconds = max(0.0, monotonic() - self._review_started_at)
            self._append_log(
                f"{self._current_transcription.input_video.name}: Review completed in {review_wait_seconds:.2f}s"
            )
        self._review_started_at = None

        self._content_stack.setCurrentIndex(0)
        self.status_label.setText(
            f"Writing translated subtitles for {self._current_transcription.input_video.name}..."
        )
        self.stop_button.setEnabled(True)

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
        thread.completed.connect(
            lambda result, review_seconds=review_wait_seconds: self._on_finalize_completed(
                result,
                review_seconds,
            )
        )
        thread.cancelled.connect(self._on_cancelled)
        thread.failed.connect(self._on_failed)
        thread.finished.connect(lambda t=thread: self._on_thread_finished(t))
        self._finalize_thread = thread
        self._current_transcription = None
        thread.start()

    def _on_finalize_completed(self, result: PipelineResult, review_wait_seconds: float) -> None:
        result.stage_durations["review_wait_seconds"] = review_wait_seconds
        self._all_results.append(result)
        self._refresh_summary(result)
        self._current_file_index += 1
        if self._current_file_index >= len(self._selected_files):
            self._on_all_done()
            return

        self._advance_after_finalize()

    def _on_all_done(self) -> None:
        self._set_busy(False)
        self._elapsed_timer.stop()
        self._job_started_at = None
        self._stop_requested = False
        self.stop_button.setEnabled(False)
        self._review_started_at = None
        self._clear_prefetch_state()
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
        self._stop_requested = False
        self.stop_button.setEnabled(False)
        self._review_started_at = None
        self._current_transcription = None
        self._clear_prefetch_state()
        self._content_stack.setCurrentIndex(0)
        self.status_label.setText("The subtitle pipeline hit an error.")
        self._append_log(error_message)
        self.review_flags_label.setText(
            "Processing stopped with an error. Check the session log for the exact FFmpeg, Whisper, or translation failure."
        )
        QMessageBox.critical(self, APP_NAME, error_message)

    def _on_cancelled(self, message: str) -> None:
        self._set_busy(False)
        self._elapsed_timer.stop()
        self._job_started_at = None
        self._stop_requested = False
        self.stop_button.setEnabled(False)
        self._review_started_at = None
        self._current_transcription = None
        self._clear_prefetch_state()
        self._content_stack.setCurrentIndex(0)
        self.status_label.setText("Processing stopped.")
        self.active_file_value.setText("Stopped")
        self.queue_value.setText(
            f"Stopped after {len(self._all_results)} of {len(self._selected_files)} finished"
        )
        self.review_flags_label.setText(message)
        self._append_log(message)

    def _request_stop(self) -> None:
        active_threads = [
            thread
            for thread in (self._transcription_thread, self._finalize_thread)
            if thread is not None and thread.isRunning()
        ]
        if (not active_threads and self._current_transcription is None) or self._stop_requested:
            return

        self._stop_requested = True
        self.stop_button.setEnabled(False)
        self._append_log("Stop requested by user.")

        for thread in active_threads:
            thread.request_stop()

        if self._finalize_thread is not None and self._finalize_thread.isRunning():
            self.status_label.setText("Stop requested. Cancelling the current export...")
            return

        if self._transcription_thread is not None and self._transcription_thread.isRunning():
            if self._active_transcription_is_prefetch and self._current_transcription is not None:
                self.status_label.setText("Stop requested. Cancelling background preparation...")
            else:
                self.status_label.setText(
                    "Stop requested. Waiting briefly for Whisper to finish before cancelling."
                )
            return

        self._on_cancelled("Processing stopped by user.")

    def _advance_after_finalize(self) -> None:
        next_index = self._current_file_index
        next_video = self._selected_files[next_index]
        self.active_file_value.setText(next_video.name)
        self.queue_value.setText(f"File {next_index + 1} of {len(self._selected_files)}")
        self.stop_button.setEnabled(True)

        if (
            self._prefetched_transcription is not None
            and self._prefetched_file_index == next_index
        ):
            prefetched_result = self._prefetched_transcription
            self._clear_prefetch_state()
            self._show_review(prefetched_result, next_index)
            return

        if (
            self._transcription_thread is not None
            and self._transcription_thread.isRunning()
            and self._active_transcription_is_prefetch
            and self._active_transcription_index == next_index
        ):
            self._waiting_for_prefetched_review = True
            self.status_label.setText(f"Waiting for background preparation of {next_video.name}...")
            self._append_log(
                f"[{next_index + 1}/{len(self._selected_files)}] Waiting for background preparation of {next_video.name} to finish."
            )
            return

        self._start_transcription(next_index, as_prefetch=False)

    def _maybe_start_prefetch(self) -> None:
        if self._stop_requested or self._current_transcription is None:
            return

        next_index = self._current_file_index + 1
        if next_index >= len(self._selected_files):
            self._refresh_review_queue_label()
            return

        if (
            self._prefetched_transcription is not None
            and self._prefetched_file_index == next_index
        ):
            self._refresh_review_queue_label()
            return

        if self._transcription_thread is not None and self._transcription_thread.isRunning():
            if self._active_transcription_is_prefetch and self._active_transcription_index == next_index:
                self._refresh_review_queue_label()
            return

        self._start_transcription(next_index, as_prefetch=True)
        self._refresh_review_queue_label()

    def _clear_prefetch_state(self) -> None:
        self._prefetched_transcription = None
        self._prefetched_file_index = None
        self._waiting_for_prefetched_review = False

    def _refresh_review_queue_label(self) -> None:
        if self._current_transcription is None:
            return

        total = len(self._selected_files)
        base = f"File {self._current_file_index + 1} of {total}"
        next_index = self._current_file_index + 1
        if next_index >= total:
            self.review_queue_label.setText(base)
            return

        next_name = self._selected_files[next_index].name
        if (
            self._prefetched_transcription is not None
            and self._prefetched_file_index == next_index
        ):
            self.review_queue_label.setText(f"{base} | {next_name} ready next")
            return

        if (
            self._transcription_thread is not None
            and self._transcription_thread.isRunning()
            and self._active_transcription_is_prefetch
            and self._active_transcription_index == next_index
        ):
            self.review_queue_label.setText(f"{base} | Preparing {next_name} in background")
            return

        if self._waiting_for_prefetched_review:
            self.review_queue_label.setText(f"{base} | Waiting for background preparation")
            return

        self.review_queue_label.setText(base)

    def _on_thread_finished(self, thread: QThread) -> None:
        was_transcription_thread = self._transcription_thread is thread
        if self._transcription_thread is thread:
            self._transcription_thread = None
            self._active_transcription_index = None
            self._active_transcription_is_prefetch = False
        elif self._finalize_thread is thread:
            self._finalize_thread = None
        elif self._preload_thread is thread:
            self._preload_thread = None

        if (
            was_transcription_thread
            and self._current_transcription is not None
            and not self._stop_requested
        ):
            self._maybe_start_prefetch()

        thread.deleteLater()

    def _refresh_summary(self, result: PipelineResult | None) -> None:
        if result is None:
            target_label = language_label(self.target_language_combo.currentData())
            if not self._selected_files:
                self.summary_label.setText(
                    "Add videos, choose a source and target language, and the app will transcribe locally, translate subtitle text, and pause for review before export."
                )
                self.review_flags_label.setText("No review flags yet.")
                self.preview_output.setPlainText("")
            else:
                self.summary_label.setText(
                    f"{len(self._selected_files)} video(s) queued.\n"
                    f"Target language: {target_label}\n"
                    f"Output folder: {self.output_directory_edit.text() or DEFAULT_OUTPUT_DIRECTORY}"
                )
                self.review_flags_label.setText(
                    "No review flags yet. They will appear here after transcription and translation."
                )
            return

        burned_video_text = str(result.burned_video) if result.burned_video else "Not generated"
        self.summary_label.setText(
            f"Finished: {result.input_video.name}\n"
            f"Detected language: {language_label(result.detected_language)}\n"
            f"Target language: {language_label(result.target_language)}\n"
            f"Translation provider: {result.translation_provider or 'Not used'}\n"
            f"Engine: {result.device_label or 'unknown'}\n"
            f"Subtitle file: {result.subtitle_file}\n"
            f"Burned video: {burned_video_text}\n"
            f"Segments: {result.segment_count}\n"
            f"Elapsed: {result.elapsed_seconds:.1f}s"
        )
        if result.stage_durations:
            ordered_keys = (
                "audio_extraction_seconds",
                "whisper_seconds",
                "translation_seconds",
                "review_wait_seconds",
                "finalize_seconds",
            )
            timing_summary = ", ".join(
                f"{key.removesuffix('_seconds')}: {result.stage_durations[key]:.1f}s"
                for key in ordered_keys
                if key in result.stage_durations
            )
            if timing_summary:
                self.summary_label.setText(f"{self.summary_label.text()}\nStage timings: {timing_summary}")
        self.engine_value.setText(result.device_label or "Unknown")
        if result.warning_messages:
            self.review_flags_label.setText("\n".join(f"- {item}" for item in result.warning_messages))
        else:
            self.review_flags_label.setText(
                "No automatic review flags. You should still skim the translated preview before publishing."
            )
        self.preview_output.setPlainText(result.preview_text or "No preview available.")

    def _review_summary_text(self, result: TranscriptionResult) -> str:
        metadata = result.metadata
        lines = [
            f"Detected source language: {language_label(metadata.detected_language)}",
            f"Target language: {language_label(metadata.target_language)}",
            f"Translation provider: {metadata.translation_provider or 'Not used'}",
            f"Segments: {len(result.review_segments)}",
            f"Model / engine: {self.whisper_model_combo.currentData()} on {metadata.device_label or 'unknown'}",
        ]
        if metadata.stage_durations:
            stage_order = ("audio_extraction_seconds", "whisper_seconds", "translation_seconds")
            stage_summary = ", ".join(
                f"{key.removesuffix('_seconds')}: {metadata.stage_durations[key]:.1f}s"
                for key in stage_order
                if key in metadata.stage_durations
            )
            if stage_summary:
                lines.append(f"Stage timings: {stage_summary}")
        if result.warning_messages:
            lines.append("Check first:")
            lines.extend(f"- {warning}" for warning in result.warning_messages)
        else:
            lines.append("No automatic review flags. Still skim terminology, line breaks, and reading speed.")
        return "\n".join(lines)

    def _set_busy(self, busy: bool) -> None:
        controls = [
            self.add_videos_button,
            self.clear_videos_button,
            self.browse_output_button,
            self.run_button,
            self.source_language_combo,
            self.target_language_combo,
            self.translation_provider_combo,
            self.whisper_model_combo,
            self.output_mode_combo,
            self.max_line_length_spinbox,
            self.subtitle_font_size_spinbox,
            self.output_directory_edit,
            self.translation_base_url_edit,
            self.translation_model_edit,
            self.translation_api_key_edit,
            self.save_translation_settings_button,
            self.test_translation_button,
        ]
        for control in controls:
            control.setEnabled(not busy)
        if not busy:
            self.stop_button.setEnabled(False)

    def _append_log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_output.appendPlainText(f"[{timestamp}] {message}")
        if "Whisper model" in message and " on " in message:
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

    def _translation_required_from_choices(self) -> bool:
        source_language = self.source_language_combo.currentData()
        target_language = self.target_language_combo.currentData()
        return source_language == "auto" or source_language != target_language

    def _build_translation_service(self) -> OpenAICompatibleTranslationService:
        config = TranslationProviderConfig.from_values(
            base_url=self.translation_base_url_edit.text(),
            api_key=self.translation_api_key_edit.text(),
            model=self.translation_model_edit.text(),
        )
        return OpenAICompatibleTranslationService(config)

    def _on_whisper_model_changed(self, _index: int | None = None) -> None:
        if self._can_preload_selected_model():
            self._schedule_model_preload()

    def _can_preload_selected_model(self) -> bool:
        return not any(
            thread is not None and thread.isRunning()
            for thread in (self._transcription_thread, self._finalize_thread)
        ) and self._current_transcription is None and self._prefetched_transcription is None

    def _schedule_model_preload(self) -> None:
        if not self._can_preload_selected_model():
            return

        model_name = str(self.whisper_model_combo.currentData())
        if not model_name or self._preloaded_model_name == model_name:
            return
        if self._preload_thread is not None and self._preload_thread.isRunning():
            return

        thread = ModelPreloadThread(model_name)
        thread.log_message.connect(self._append_log)
        thread.completed.connect(self._on_model_preload_completed)
        thread.failed.connect(self._on_model_preload_failed)
        thread.finished.connect(lambda t=thread: self._on_thread_finished(t))
        self._preload_thread = thread
        thread.start()

    def _on_model_preload_completed(self, model_name: str) -> None:
        self._preloaded_model_name = model_name
        self._append_log(f"Whisper warmup complete for model '{model_name}'.")

    def _on_model_preload_failed(self, error_message: str) -> None:
        self._append_log(f"Whisper warmup failed: {error_message}")

    def _refresh_translation_status(self) -> None:
        service = self._build_translation_service()
        if not self._translation_required_from_choices():
            self.translation_status_label.setText(
                "Translation provider optional: source and target languages match, so the app will transcribe only."
            )
            return

        if self.source_language_combo.currentData() == "auto" and not service.is_configured():
            self.translation_status_label.setText(
                "Translation unavailable: missing API key. Auto-detect jobs can still start, but non-matching languages will stop when translation is required."
            )
            return

        self.translation_status_label.setText(service.configuration_status())

    def _load_provider_settings(self) -> None:
        base_url = self._settings.value("translation/base_url", DEFAULT_TRANSLATION_BASE_URL, str)
        model = self._settings.value("translation/model", DEFAULT_TRANSLATION_MODEL, str)
        api_key = self._settings.value("translation/api_key", "", str)
        self.translation_base_url_edit.setText(base_url)
        self.translation_model_edit.setText(model)
        self.translation_api_key_edit.setText(api_key)

    def _persist_provider_settings(self) -> None:
        self._settings.setValue("translation/base_url", self.translation_base_url_edit.text().strip())
        self._settings.setValue("translation/model", self.translation_model_edit.text().strip())
        self._settings.setValue("translation/api_key", self.translation_api_key_edit.text())
        sync = getattr(self._settings, "sync", None)
        if callable(sync):
            sync()

    def _save_translation_settings(self) -> None:
        self._persist_provider_settings()
        self._refresh_translation_status()
        self._append_log("Saved translation settings locally.")
        QMessageBox.information(
            self,
            APP_NAME,
            "Translation settings saved locally on this machine.",
        )

    def _test_translation_connection(self) -> None:
        self._persist_provider_settings()
        service = self._build_translation_service()
        if not service.is_configured():
            message = service.configuration_status()
            self._append_log(message)
            QMessageBox.warning(self, APP_NAME, message)
            return

        target_language = self.target_language_combo.currentData()
        if target_language == "en":
            target_language = "de"

        self._append_log("Testing translation provider connection...")
        try:
            translated_text = service.test_connection(
                source_language="en",
                target_language=target_language,
                log=self._append_log,
            )
        except TranslationServiceError as exc:
            self._append_log(f"Translation connection test failed: {exc}")
            QMessageBox.critical(self, APP_NAME, str(exc))
            return
        except Exception as exc:
            self._append_log(f"Unexpected translation test failure: {exc}")
            QMessageBox.critical(self, APP_NAME, str(exc))
            return

        self._refresh_translation_status()
        self._append_log(f"Translation test succeeded: {translated_text}")
        QMessageBox.information(
            self,
            APP_NAME,
            f"Translation test succeeded.\n\nProvider output:\n{translated_text}",
        )

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
            thread is not None and thread.isRunning()
            for thread in (self._transcription_thread, self._finalize_thread)
        )
        review_active = self._current_transcription is not None or self._prefetched_transcription is not None
        preload_running = self._preload_thread is not None and self._preload_thread.isRunning()
        if running or review_active:
            QMessageBox.information(
                self,
                APP_NAME,
                "Subtitle generation or review is still active. Use Stop to cancel the current queue before closing the app.",
            )
            event.ignore()
            return
        if preload_running:
            QMessageBox.information(
                self,
                APP_NAME,
                "Whisper warmup is still running. Wait for it to finish before closing the app.",
            )
            event.ignore()
            return

        for thread in (self._transcription_thread, self._finalize_thread, self._preload_thread):
            if thread is not None:
                thread.wait(1000)
        TRANSCRIPTION_WORKER.close()
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
