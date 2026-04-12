# Post-Approval SRT Editing Design

**Date:** 2026-04-13
**Status:** Approved
**Scope:** Allow users to open any `.srt` file, edit it in the existing in-app review panel, and re-burn a video — reachable at any time, not just during a transcription session.

---

## Problem

Once a user clicks "Approve & Continue", the review panel is dismissed and the SRT is written. There is no in-app way to re-open a completed `.srt` file for editing. Users who spot errors after the fact must use an external editor and then manually re-run the standalone burn — a disjointed workflow.

---

## Approach

Extend the existing **Standalone Subtitle Burn** section with an **"Edit SRT"** button. Clicking it loads the chosen `.srt` file into the review panel in a new `"standalone_edit"` mode, where the user can correct the content and then re-burn in one click.

This approach was chosen over alternatives (new dedicated tab, or a new menu entry point) because the standalone burn section already collects both the video and SRT paths needed for re-burning, and the review panel is already the right widget for SRT editing.

---

## Section 1 — Entry Point

- Add an **"Edit SRT"** button next to `existing_burn_subtitle_edit` in the standalone burn section
- Button is enabled only when:
  - The SRT field contains a path ending in `.srt`
  - The video field is also filled
- On click: read the `.srt` file from disk and open the review panel in `"standalone_edit"` mode
- The video + SRT paths remain in their existing fields and are used unchanged for the re-burn

---

## Section 2 — Review Panel Standalone-Edit Mode

`_review_mode` (already `str | None`) takes the new value `"standalone_edit"`.

**Visual changes on entry:**

| Element | Normal review | Standalone-edit mode |
|---|---|---|
| Section title | "Review Translation" | "Edit SRT" |
| `review_file_label` | video filename | SRT filename |
| `review_queue_label` | queue position | hidden |
| `review_summary_label` | transcription metadata | "Editing {filename} — saving will overwrite the original file." |
| Left pane (`source_srt_view` card) | visible | hidden (editor takes full width) |
| Right pane title | "Translated Subtitle Draft" | "SRT Contents" |
| `use_original_button` text | "Use Original Draft" | "Reset to File" |
| `approve_button` text | "Approve & Continue" | "Save & Re-burn" |
| Cancel button | hidden | visible |

**Implementation notes:**
- `source_card` (the left pane widget) must be stored as `self._review_source_pane` during `_create_review_panel` so it can be toggled via `setVisible`.
- A new field `_standalone_edit_original_text: str | None` stores the text loaded from disk on entry, used for the dirty check on cancel.

**Draft autosave:** Unchanged — saves to a side-car draft file. The actual `.srt` on disk is never touched until the user confirms. On cancel, the draft is left in place so edits survive accidental dismissal.

---

## Section 3 — Validation and Save & Re-burn Flow

### Validation

Simpler than normal review — no `review_segments` to check against:
- Must be non-empty
- Must be parseable by `parse_srt_text`
- No segment-count or timestamp-ordering constraints (user may be fixing these)

### Save & Re-burn (on button click)

1. Validate (above); show `review_warning_label` and abort on failure
2. Write edited text to the original `.srt` file path (overwrite in place)
3. Switch content stack to index 0 (main panel)
4. Restore all panel labels/buttons to their normal state
5. Start `ExistingSubtitleBurnThread` with the video + SRT paths from the burn fields
6. Completion and error handling follow the existing standalone burn path — no new code needed

### Reset to File

- Re-reads the `.srt` from disk and replaces editor content
- No confirmation prompt (immediately reversible by typing)

### Cancel

- If editor content matches what was originally loaded (clean): return to main panel immediately, no dialog
- If editor content differs (dirty): show a one-line `QMessageBox` confirmation before discarding

---

## Section 4 — Tests

All tests go in the existing `main_window` test file, following existing mock patterns.

| Test | Assert |
|---|---|
| `test_standalone_edit_mode_entry` | Panel switches to index 1, `_review_mode == "standalone_edit"`, left pane hidden, button labels updated |
| `test_standalone_edit_save_overwrites_srt` | `Path.write_text` called with edited content at original SRT path before thread starts |
| `test_standalone_edit_validation_rejects_empty` | Warning shown, no file write, no thread started |
| `test_standalone_edit_validation_rejects_invalid_srt` | Same as above for malformed SRT |
| `test_standalone_edit_reset_reloads_from_disk` | `Path.read_text` called, editor content replaced |
| `test_standalone_edit_cancel_dirty_prompts` | `QMessageBox` shown before returning to main panel |
| `test_standalone_edit_cancel_clean_no_prompt` | Immediate return, no dialog |

---

## Files Affected

- `src/add_subtitles_to_videos/ui/main_window.py` — all UI and flow changes
- Existing test file covering `main_window` behaviour — new tests only

No new files. No changes to service layer, models, or config.
