# Post-Approval SRT Editing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow users to open any `.srt` file in the existing in-app review panel, edit it, and re-burn a video — accessible via a new "Edit Existing SRT" button in the Standalone Subtitle Burn section.

**Architecture:** The "Edit Existing SRT" button (already wired in the UI to `_start_existing_srt_edit`) reads the chosen `.srt` into the existing review panel under a new `"standalone_edit"` mode. The panel's labels, button texts, and left-pane visibility adapt for this mode. On confirm, the edited SRT is written back to disk and `ExistingSubtitleBurnThread` is started. A `_leave_standalone_edit_mode` helper restores the panel on save, cancel, or error.

**Tech Stack:** Python, PySide6, existing `parse_srt_text` / `ExistingSubtitleBurnThread` from the same codebase.

---

## File Map

- **Modify:** `src/add_subtitles_to_videos/ui/main_window.py` — all logic
- **Test:** `tests/test_subtitles.py` — append new tests at the end

---

## Task 1: Panel infrastructure — store widget references, add Cancel button, add `_standalone_edit_original_text`

**Files:**
- Modify: `src/add_subtitles_to_videos/ui/main_window.py`
- Test: `tests/test_subtitles.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_subtitles.py`:

```python
def test_standalone_edit_panel_infrastructure(monkeypatch) -> None:
    _application()
    _patch_settings(monkeypatch)
    window = main_window_module.MainWindow()

    # New widget attributes exist
    assert hasattr(window, "_review_source_pane")
    assert hasattr(window, "_review_panel_title")
    assert hasattr(window, "_review_translated_title")
    assert hasattr(window, "cancel_edit_button")
    assert hasattr(window, "_standalone_edit_original_text")

    # Cancel button starts hidden
    assert window.cancel_edit_button.isVisible() is False

    # Source pane starts visible
    assert window._review_source_pane.isVisible() is True

    window.close()
```

- [ ] **Step 2: Run test to verify it fails**

```
cd "d:/GitHub Projects/add-subtitles-to-videos"
python -m pytest tests/test_subtitles.py::test_standalone_edit_panel_infrastructure -v
```

Expected: FAIL with `AttributeError`

- [ ] **Step 3: In `__init__`, add `_standalone_edit_original_text`**

In `src/add_subtitles_to_videos/ui/main_window.py`, find the block of `self._review_*` fields around line 260. Add after `self._review_mode: str | None = None`:

```python
        self._standalone_edit_original_text: str | None = None
```

- [ ] **Step 4: In `_create_review_panel`, store the section title label**

Find (around line 705):
```python
        header_row.addWidget(self._section_title("Review Translation"))
```
Replace with:
```python
        self._review_panel_title = self._section_title("Review Translation")
        header_row.addWidget(self._review_panel_title)
```

- [ ] **Step 5: In `_create_review_panel`, store the source card**

Find (around line 742):
```python
        transcript_splitter.addWidget(source_card)
```
Replace with:
```python
        self._review_source_pane = source_card
        transcript_splitter.addWidget(source_card)
```

- [ ] **Step 6: In `_create_review_panel`, store the translated pane title**

Find (around line 750):
```python
        translated_title = QLabel("Translated Subtitle Draft")
        translated_title.setObjectName("miniTitle")
        translated_layout.addWidget(translated_title)
```
Replace with:
```python
        self._review_translated_title = QLabel("Translated Subtitle Draft")
        self._review_translated_title.setObjectName("miniTitle")
        translated_layout.addWidget(self._review_translated_title)
```

- [ ] **Step 7: In `_create_review_panel`, add the Cancel button to the button row**

Find (around line 789):
```python
        button_row.addWidget(self.insert_missing_segment_button)
        button_row.addWidget(self.use_original_button)
        button_row.addWidget(self.approve_button)
```
Replace with:
```python
        self.cancel_edit_button = QPushButton("Cancel")
        self.cancel_edit_button.setObjectName("secondaryButton")
        self.cancel_edit_button.clicked.connect(self._on_cancel_edit_clicked)
        self.cancel_edit_button.setVisible(False)

        button_row.addWidget(self.insert_missing_segment_button)
        button_row.addWidget(self.cancel_edit_button)
        button_row.addWidget(self.use_original_button)
        button_row.addWidget(self.approve_button)
```

- [ ] **Step 8: Run test to verify it passes**

```
python -m pytest tests/test_subtitles.py::test_standalone_edit_panel_infrastructure -v
```

Expected: PASS

- [ ] **Step 9: Run the full test suite to check for regressions**

```
python -m pytest tests/test_subtitles.py -x -q
```

Expected: all tests pass

- [ ] **Step 10: Commit**

```bash
git add src/add_subtitles_to_videos/ui/main_window.py tests/test_subtitles.py
git commit -m "feat: add standalone_edit panel infrastructure — widget refs and cancel button"
```

---

## Task 2: Implement `_start_existing_srt_edit` (entry point)

**Files:**
- Modify: `src/add_subtitles_to_videos/ui/main_window.py`
- Test: `tests/test_subtitles.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_subtitles.py`:

```python
def test_standalone_edit_mode_entry(monkeypatch, tmp_path) -> None:
    _application()
    _patch_settings(monkeypatch)

    srt_path = tmp_path / "demo.srt"
    srt_content = "1\n00:00:00,000 --> 00:00:01,000\nhello\n"
    srt_path.write_text(srt_content, encoding="utf-8")
    video_path = tmp_path / "demo.mp4"
    video_path.write_bytes(b"video")

    window = main_window_module.MainWindow()
    window.existing_burn_video_edit.setText(str(video_path))
    window.existing_burn_subtitle_edit.setText(str(srt_path))

    window._start_existing_srt_edit()

    assert window._content_stack.currentIndex() == 1
    assert window._review_mode == "standalone_edit"
    assert window._review_source_pane.isVisible() is False
    assert window.review_queue_label.isVisible() is False
    assert window.cancel_edit_button.isVisible() is True
    assert window.approve_button.text() == "Save & Re-burn"
    assert window.use_original_button.text() == "Reset to File"
    assert window._review_panel_title.text() == "Edit SRT"
    assert window.review_file_label.text() == "demo.srt"
    assert "demo.srt" in window.review_summary_label.text()
    assert "overwrite" in window.review_summary_label.text()
    assert window.translated_srt_editor.toPlainText() == srt_content.strip()
    assert window._standalone_edit_original_text == srt_content

    window.close()


def test_standalone_edit_entry_rejects_missing_fields(monkeypatch, tmp_path) -> None:
    _application()
    _patch_settings(monkeypatch)
    warning_calls: list[str] = []
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "warning",
        lambda *a, **kw: warning_calls.append(a[2]),
    )

    srt_path = tmp_path / "demo.srt"
    srt_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n", encoding="utf-8")

    window = main_window_module.MainWindow()

    # No video, no SRT
    window._start_existing_srt_edit()
    assert window._content_stack.currentIndex() == 0
    assert len(warning_calls) == 1

    # Video set, no SRT
    warning_calls.clear()
    window.existing_burn_video_edit.setText(str(tmp_path / "demo.mp4"))
    window._start_existing_srt_edit()
    assert window._content_stack.currentIndex() == 0
    assert len(warning_calls) == 1

    # Both set but SRT doesn't exist
    warning_calls.clear()
    window.existing_burn_subtitle_edit.setText(str(tmp_path / "missing.srt"))
    window._start_existing_srt_edit()
    assert window._content_stack.currentIndex() == 0
    assert len(warning_calls) == 1

    window.close()
```

- [ ] **Step 2: Run tests to verify they fail**

```
python -m pytest tests/test_subtitles.py::test_standalone_edit_mode_entry tests/test_subtitles.py::test_standalone_edit_entry_rejects_missing_fields -v
```

Expected: FAIL with `AttributeError: '_start_existing_srt_edit'`

- [ ] **Step 3: Implement `_start_existing_srt_edit`**

Add this method to `main_window.py` after `_start_existing_burn` (around line 1030). The method reads and validates the fields, loads the SRT into the review panel, and adapts its appearance for standalone-edit mode:

```python
    def _start_existing_srt_edit(self) -> None:
        video_text = self.existing_burn_video_edit.text().strip()
        subtitle_text = self.existing_burn_subtitle_edit.text().strip()
        if not video_text:
            QMessageBox.warning(self, APP_NAME, "Choose a source video first.")
            return
        if not subtitle_text:
            QMessageBox.warning(self, APP_NAME, "Choose an SRT file to edit.")
            return
        srt_path = Path(subtitle_text).expanduser().resolve()
        if not srt_path.exists():
            QMessageBox.warning(self, APP_NAME, "The selected SRT file does not exist.")
            return
        if srt_path.suffix.casefold() != ".srt":
            QMessageBox.warning(self, APP_NAME, "Choose an SRT file with the .srt extension.")
            return

        srt_content = srt_path.read_text(encoding="utf-8")

        self._review_mode = "standalone_edit"
        self._standalone_edit_original_text = srt_content

        self._review_panel_title.setText("Edit SRT")
        self.review_file_label.setText(srt_path.name)
        self.review_queue_label.setVisible(False)
        self.review_summary_label.setText(
            f"Editing {srt_path.name} — saving will overwrite the original file."
        )
        self._review_source_pane.setVisible(False)
        self._review_translated_title.setText("SRT Contents")
        self.translated_srt_editor.setPlaceholderText("SRT content will appear here.")
        self.translated_srt_editor.setPlainText(srt_content.strip())
        self.use_original_button.setText("Reset to File")
        self.approve_button.setText("Save & Re-burn")
        self.cancel_edit_button.setVisible(True)
        self.review_warning_label.setVisible(False)

        self._content_stack.setCurrentIndex(1)
```

- [ ] **Step 4: Run tests to verify they pass**

```
python -m pytest tests/test_subtitles.py::test_standalone_edit_mode_entry tests/test_subtitles.py::test_standalone_edit_entry_rejects_missing_fields -v
```

Expected: PASS

- [ ] **Step 5: Run the full test suite**

```
python -m pytest tests/test_subtitles.py -x -q
```

Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
git add src/add_subtitles_to_videos/ui/main_window.py tests/test_subtitles.py
git commit -m "feat: implement _start_existing_srt_edit entry point"
```

---

## Task 3: Implement `_leave_standalone_edit_mode` (shared cleanup)

**Files:**
- Modify: `src/add_subtitles_to_videos/ui/main_window.py`
- Test: `tests/test_subtitles.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_subtitles.py`:

```python
def test_standalone_edit_leave_restores_panel(monkeypatch, tmp_path) -> None:
    _application()
    _patch_settings(monkeypatch)

    srt_path = tmp_path / "demo.srt"
    srt_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n", encoding="utf-8")
    video_path = tmp_path / "demo.mp4"
    video_path.write_bytes(b"video")

    window = main_window_module.MainWindow()
    window.existing_burn_video_edit.setText(str(video_path))
    window.existing_burn_subtitle_edit.setText(str(srt_path))
    window._start_existing_srt_edit()

    # Confirm we are in standalone_edit mode
    assert window._review_mode == "standalone_edit"

    window._leave_standalone_edit_mode()

    assert window._review_mode is None
    assert window._standalone_edit_original_text is None
    assert window._content_stack.currentIndex() == 0
    assert window._review_source_pane.isVisible() is True
    assert window.review_queue_label.isVisible() is True
    assert window.cancel_edit_button.isVisible() is False
    assert window.approve_button.text() == "Approve & Continue"
    assert window.use_original_button.text() == "Use Original Draft"
    assert window._review_panel_title.text() == "Review Translation"
    assert window._review_translated_title.text() == "Translated Subtitle Draft"

    window.close()
```

- [ ] **Step 2: Run test to verify it fails**

```
python -m pytest tests/test_subtitles.py::test_standalone_edit_leave_restores_panel -v
```

Expected: FAIL with `AttributeError: '_leave_standalone_edit_mode'`

- [ ] **Step 3: Implement `_leave_standalone_edit_mode`**

Add this method to `main_window.py` after `_start_existing_srt_edit`:

```python
    def _leave_standalone_edit_mode(self) -> None:
        self._review_mode = None
        self._standalone_edit_original_text = None

        self._review_panel_title.setText("Review Translation")
        self.review_queue_label.setVisible(True)
        self.review_summary_label.setText(
            "Detected source language, target language, provider, and warnings will appear here once processing finishes."
        )
        self._review_source_pane.setVisible(True)
        self._review_translated_title.setText("Translated Subtitle Draft")
        self.translated_srt_editor.setPlaceholderText(
            "Translated SRT text will appear here. Keep the original timed blocks, but you can add extra SRT blocks for missed subtitles."
        )
        self.use_original_button.setText("Use Original Draft")
        self.approve_button.setText("Approve & Continue")
        self.cancel_edit_button.setVisible(False)
        self.review_warning_label.setVisible(False)

        self._content_stack.setCurrentIndex(0)
```

- [ ] **Step 4: Run test to verify it passes**

```
python -m pytest tests/test_subtitles.py::test_standalone_edit_leave_restores_panel -v
```

Expected: PASS

- [ ] **Step 5: Run the full test suite**

```
python -m pytest tests/test_subtitles.py -x -q
```

Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
git add src/add_subtitles_to_videos/ui/main_window.py tests/test_subtitles.py
git commit -m "feat: implement _leave_standalone_edit_mode cleanup helper"
```

---

## Task 4: Implement "Save & Re-burn" — modify `_on_approve_clicked`

**Files:**
- Modify: `src/add_subtitles_to_videos/ui/main_window.py`
- Test: `tests/test_subtitles.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_subtitles.py`:

```python
def test_standalone_edit_save_overwrites_srt(monkeypatch, tmp_path) -> None:
    _application()
    _patch_settings(monkeypatch)
    monkeypatch.setattr(main_window_module, "ExistingSubtitleBurnThread", ImmediateExistingBurnThread)

    srt_path = tmp_path / "demo.srt"
    srt_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n", encoding="utf-8")
    video_path = tmp_path / "demo.mp4"
    video_path.write_bytes(b"video")

    window = main_window_module.MainWindow()
    window.output_directory_edit.setText(str(tmp_path))
    window.existing_burn_video_edit.setText(str(video_path))
    window.existing_burn_subtitle_edit.setText(str(srt_path))
    window._start_existing_srt_edit()

    window.translated_srt_editor.setPlainText("1\n00:00:00,000 --> 00:00:01,500\nhello world")
    window._on_approve_clicked()

    assert srt_path.read_text(encoding="utf-8") == "1\n00:00:00,000 --> 00:00:01,500\nhello world\n"
    assert window._review_mode is None
    assert window._content_stack.currentIndex() == 0
    assert _pump_events_until(lambda: window.status_label.text() == "Standalone subtitle burn finished.")

    window.close()


def test_standalone_edit_validation_rejects_empty(monkeypatch, tmp_path) -> None:
    _application()
    _patch_settings(monkeypatch)

    srt_path = tmp_path / "demo.srt"
    srt_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n", encoding="utf-8")
    video_path = tmp_path / "demo.mp4"
    video_path.write_bytes(b"video")

    window = main_window_module.MainWindow()
    window.existing_burn_video_edit.setText(str(video_path))
    window.existing_burn_subtitle_edit.setText(str(srt_path))
    window._start_existing_srt_edit()

    window.translated_srt_editor.setPlainText("")
    window._on_approve_clicked()

    # Stays in review panel, file untouched
    assert window._content_stack.currentIndex() == 1
    assert window._review_mode == "standalone_edit"
    assert window.review_warning_label.isVisible() is True
    assert srt_path.read_text(encoding="utf-8") == "1\n00:00:00,000 --> 00:00:01,000\nhello\n"

    window.close()


def test_standalone_edit_validation_rejects_invalid_srt(monkeypatch, tmp_path) -> None:
    _application()
    _patch_settings(monkeypatch)

    srt_path = tmp_path / "demo.srt"
    srt_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n", encoding="utf-8")
    video_path = tmp_path / "demo.mp4"
    video_path.write_bytes(b"video")

    window = main_window_module.MainWindow()
    window.existing_burn_video_edit.setText(str(video_path))
    window.existing_burn_subtitle_edit.setText(str(srt_path))
    window._start_existing_srt_edit()

    window.translated_srt_editor.setPlainText("this is not valid srt content at all")
    window._on_approve_clicked()

    assert window._content_stack.currentIndex() == 1
    assert window._review_mode == "standalone_edit"
    assert window.review_warning_label.isVisible() is True
    assert srt_path.read_text(encoding="utf-8") == "1\n00:00:00,000 --> 00:00:01,000\nhello\n"

    window.close()
```

- [ ] **Step 2: Run tests to verify they fail**

```
python -m pytest tests/test_subtitles.py::test_standalone_edit_save_overwrites_srt tests/test_subtitles.py::test_standalone_edit_validation_rejects_empty tests/test_subtitles.py::test_standalone_edit_validation_rejects_invalid_srt -v
```

Expected: FAIL (approve in standalone_edit mode calls the normal flow which asserts `_current_transcription is not None`)

- [ ] **Step 3: Modify `_on_approve_clicked` to branch on `_review_mode`**

Find `_on_approve_clicked` (around line 1008). Add a branch at the top of the method:

```python
    def _on_approve_clicked(self) -> None:
        if self._review_mode == "standalone_edit":
            self._on_standalone_edit_save()
            return
        assert self._current_transcription is not None
        # ... rest of the existing method unchanged ...
```

Then add `_on_standalone_edit_save` after `_on_approve_clicked`:

```python
    def _on_standalone_edit_save(self) -> None:
        srt_text = self.translated_srt_editor.toPlainText().strip()
        if not srt_text:
            self.review_warning_label.setText("SRT content cannot be empty.")
            self.review_warning_label.setVisible(True)
            return
        try:
            parse_srt_text(srt_text)
        except ValueError:
            self.review_warning_label.setText(
                "The SRT text is not valid. Check that each block has an index, a timestamp line, and text."
            )
            self.review_warning_label.setVisible(True)
            return

        srt_path = Path(self.existing_burn_subtitle_edit.text()).expanduser().resolve()
        srt_path.write_text(srt_text + "\n", encoding="utf-8")

        self._leave_standalone_edit_mode()
        self._start_existing_burn()
```

- [ ] **Step 4: Run tests to verify they pass**

```
python -m pytest tests/test_subtitles.py::test_standalone_edit_save_overwrites_srt tests/test_subtitles.py::test_standalone_edit_validation_rejects_empty tests/test_subtitles.py::test_standalone_edit_validation_rejects_invalid_srt -v
```

Expected: PASS

- [ ] **Step 5: Run the full test suite**

```
python -m pytest tests/test_subtitles.py -x -q
```

Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
git add src/add_subtitles_to_videos/ui/main_window.py tests/test_subtitles.py
git commit -m "feat: implement save and re-burn for standalone SRT edit mode"
```

---

## Task 5: Implement "Reset to File" — modify `_on_use_original_clicked`

**Files:**
- Modify: `src/add_subtitles_to_videos/ui/main_window.py`
- Test: `tests/test_subtitles.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_subtitles.py`:

```python
def test_standalone_edit_reset_reloads_from_disk(monkeypatch, tmp_path) -> None:
    _application()
    _patch_settings(monkeypatch)

    original = "1\n00:00:00,000 --> 00:00:01,000\nhello\n"
    srt_path = tmp_path / "demo.srt"
    srt_path.write_text(original, encoding="utf-8")
    video_path = tmp_path / "demo.mp4"
    video_path.write_bytes(b"video")

    window = main_window_module.MainWindow()
    window.existing_burn_video_edit.setText(str(video_path))
    window.existing_burn_subtitle_edit.setText(str(srt_path))
    window._start_existing_srt_edit()

    # Simulate user edits
    window.translated_srt_editor.setPlainText("1\n00:00:00,000 --> 00:00:01,000\nchanged text")

    window._on_use_original_clicked()

    assert window.translated_srt_editor.toPlainText() == original.strip()
    assert window._standalone_edit_original_text == original
    # Still in edit mode
    assert window._review_mode == "standalone_edit"

    window.close()
```

- [ ] **Step 2: Run test to verify it fails**

```
python -m pytest tests/test_subtitles.py::test_standalone_edit_reset_reloads_from_disk -v
```

Expected: FAIL (assertion error because `assert self._current_transcription is not None` fires)

- [ ] **Step 3: Modify `_on_use_original_clicked` to branch on `_review_mode`**

Find `_on_use_original_clicked` (around line 1024):

```python
    def _on_use_original_clicked(self) -> None:
        assert self._current_transcription is not None
        self._start_finalize(self._current_transcription.translated_srt_text)
```

Replace with:

```python
    def _on_use_original_clicked(self) -> None:
        if self._review_mode == "standalone_edit":
            srt_path = Path(self.existing_burn_subtitle_edit.text()).expanduser().resolve()
            content = srt_path.read_text(encoding="utf-8")
            self._standalone_edit_original_text = content
            self.translated_srt_editor.setPlainText(content.strip())
            return
        assert self._current_transcription is not None
        self._start_finalize(self._current_transcription.translated_srt_text)
```

- [ ] **Step 4: Run test to verify it passes**

```
python -m pytest tests/test_subtitles.py::test_standalone_edit_reset_reloads_from_disk -v
```

Expected: PASS

- [ ] **Step 5: Run the full test suite**

```
python -m pytest tests/test_subtitles.py -x -q
```

Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
git add src/add_subtitles_to_videos/ui/main_window.py tests/test_subtitles.py
git commit -m "feat: implement reset-to-file for standalone SRT edit mode"
```

---

## Task 6: Implement Cancel — add `_on_cancel_edit_clicked`

**Files:**
- Modify: `src/add_subtitles_to_videos/ui/main_window.py`
- Test: `tests/test_subtitles.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_subtitles.py`:

```python
def test_standalone_edit_cancel_clean_no_prompt(monkeypatch, tmp_path) -> None:
    _application()
    _patch_settings(monkeypatch)
    question_calls: list = []
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "question",
        lambda *a, **kw: question_calls.append(a) or main_window_module.QMessageBox.StandardButton.Discard,
    )

    srt_path = tmp_path / "demo.srt"
    srt_content = "1\n00:00:00,000 --> 00:00:01,000\nhello\n"
    srt_path.write_text(srt_content, encoding="utf-8")
    video_path = tmp_path / "demo.mp4"
    video_path.write_bytes(b"video")

    window = main_window_module.MainWindow()
    window.existing_burn_video_edit.setText(str(video_path))
    window.existing_burn_subtitle_edit.setText(str(srt_path))
    window._start_existing_srt_edit()
    # No edits — content matches what was loaded from disk

    window._on_cancel_edit_clicked()

    assert len(question_calls) == 0  # No dialog shown
    assert window._content_stack.currentIndex() == 0
    assert window._review_mode is None

    window.close()


def test_standalone_edit_cancel_dirty_prompts(monkeypatch, tmp_path) -> None:
    _application()
    _patch_settings(monkeypatch)

    srt_path = tmp_path / "demo.srt"
    srt_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n", encoding="utf-8")
    video_path = tmp_path / "demo.mp4"
    video_path.write_bytes(b"video")

    window = main_window_module.MainWindow()
    window.existing_burn_video_edit.setText(str(video_path))
    window.existing_burn_subtitle_edit.setText(str(srt_path))
    window._start_existing_srt_edit()

    # Make a change
    window.translated_srt_editor.setPlainText("1\n00:00:00,000 --> 00:00:01,000\nchanged")

    # User clicks Cancel but then clicks Cancel on the confirmation dialog
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "question",
        lambda *a, **kw: main_window_module.QMessageBox.StandardButton.Cancel,
    )
    window._on_cancel_edit_clicked()
    assert window._content_stack.currentIndex() == 1  # Still in edit mode

    # User clicks Cancel and confirms Discard
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "question",
        lambda *a, **kw: main_window_module.QMessageBox.StandardButton.Discard,
    )
    window._on_cancel_edit_clicked()
    assert window._content_stack.currentIndex() == 0
    assert window._review_mode is None

    window.close()
```

- [ ] **Step 2: Run tests to verify they fail**

```
python -m pytest tests/test_subtitles.py::test_standalone_edit_cancel_clean_no_prompt tests/test_subtitles.py::test_standalone_edit_cancel_dirty_prompts -v
```

Expected: FAIL with `AttributeError: '_on_cancel_edit_clicked'`

- [ ] **Step 3: Implement `_on_cancel_edit_clicked`**

Add this method to `main_window.py` after `_leave_standalone_edit_mode`. The dirty check compares current editor text to what was originally loaded from disk (`_standalone_edit_original_text`). Note that `_start_existing_srt_edit` stores the raw file content in `_standalone_edit_original_text` and loads `srt_content.strip()` into the editor, so the comparison is against the stripped version of the original:

```python
    def _on_cancel_edit_clicked(self) -> None:
        current_text = self.translated_srt_editor.toPlainText()
        original_stripped = (self._standalone_edit_original_text or "").strip()
        if current_text != original_stripped:
            response = QMessageBox.question(
                self,
                APP_NAME,
                "You have unsaved changes. Discard them and return to the main panel?",
                QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if response != QMessageBox.StandardButton.Discard:
                return
        self._leave_standalone_edit_mode()
```

- [ ] **Step 4: Run tests to verify they pass**

```
python -m pytest tests/test_subtitles.py::test_standalone_edit_cancel_clean_no_prompt tests/test_subtitles.py::test_standalone_edit_cancel_dirty_prompts -v
```

Expected: PASS

- [ ] **Step 5: Run the full test suite**

```
python -m pytest tests/test_subtitles.py -x -q
```

Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
git add src/add_subtitles_to_videos/ui/main_window.py tests/test_subtitles.py
git commit -m "feat: implement cancel with dirty-check for standalone SRT edit mode"
```

---

## Done

All 6 tasks complete. The "Edit Existing SRT" button (already present in the UI) is now fully wired. Users can open any `.srt` file, edit it in the review panel, and re-burn — from a fresh app launch, independent of any transcription session.
