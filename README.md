# Subtitle Foundry

Cross-platform desktop app for transcribing spoken video, translating subtitle text, and exporting subtitle files with optional burned-in video.

The current release is a Europe-focused multilingual workflow: local Whisper handles speech recognition, an online text-translation provider handles target-language translation, and the app always pauses for review before export.

## What It Does

- Select one or more videos from anywhere on disk.
- Choose the source language or let Whisper detect it.
- Choose a target subtitle language from the launch set.
- Transcribe speech locally with faster-whisper by default, then translate subtitle text through an OpenAI-compatible API.
- Review the source transcript and translated SRT side by side before export.
- Export an `.srt` file.
- Optionally burn the subtitles directly into a new video file.
- Run local transcription with `faster-whisper` by default and a bundled FFmpeg binary while keeping translation text-only.

## Launch Languages

- English
- Greek
- Turkish
- German
- French
- Italian
- Spanish
- Portuguese
- Dutch
- Romanian
- Polish
- Czech

## Why Local Whisper + Online Translation

Speech recognition stays local so large video audio does not have to be uploaded, and Whisper can keep using your CPU or GPU directly. Translation is handled separately through an OpenAI-compatible text API so the app can support many target languages instead of Whisper's built-in translate-to-English-only path.

## Quick Start

1. Install `uv` if you do not already have it: https://docs.astral.sh/uv/
2. Sync the project:

```bash
uv sync
```

3. Launch the desktop app:

```bash
powershell -ExecutionPolicy Bypass -File .\scripts\launch_windows_outside_venv.ps1
```

On Windows, the recommended launcher is `launch-subtitle-foundry.vbs`. It starts the app from a user-level Python 3.12 install outside the repo `.venv`, installs the external runtime dependencies automatically if they are missing, and keeps normal day-to-day app runs out of the project virtual environment.

For local development tasks such as running tests, packaging, or editing dependencies, keep using `uv` commands from the repo as usual.

Before running translated workflows, configure these values in the app:

- Translation base URL
- Translation model
- Translation API key

If the source and target languages match, the app can run transcription-only without translation credentials.

## Notes

- The first run downloads the selected Whisper model, so it can take a while.
- If you want higher Hugging Face download rate limits for that first model fetch, optionally set `HF_TOKEN` before launching the app.
- `large-v3` is the default because accuracy matters more than speed for subtitle review, and it now runs through `faster-whisper` by default on CPU and CUDA systems.
- Whisper is used only for transcription and language detection. Many-to-many translation is handled by the configured text provider.
- On Windows x64, the project is pinned to the official CUDA 12.4 PyTorch wheel so NVIDIA GPUs can be used automatically when available.
- Burned-in subtitle exports are written with a new `.subtitled` suffix so your original files stay untouched.
- The app surfaces active file, queue position, elapsed time, active engine, translation-provider status, subtitle preview, and automatic review flags so you can tell when the workflow needs human attention.

## Packaging Direction

- The app is built with `PySide6`, so the next packaging step is `pyside6-deploy` or a platform-specific packager such as PyInstaller.
- Plan to build native installers separately on Windows, macOS, and Linux.
- `openai-whisper` remains available as a compatibility fallback for environments where `faster-whisper` is not a fit.
- If you later want an even more specialized GPU-focused backend on specific machines, `whisper.cpp` is the next option to evaluate.

## Windows Packaging

Build a branded Windows app bundle:

```bash
powershell -ExecutionPolicy Bypass -File .\scripts\build_windows_bundle.ps1
```

Build a branded Windows installer:

```bash
powershell -ExecutionPolicy Bypass -File .\scripts\build_windows_installer.ps1
```

Outputs:

- App bundle: `dist\SubtitleFoundry\SubtitleFoundry.exe`
- Installer: `build\installer\subtitle-foundry-setup.exe`

## Testing

```bash
uv run pytest
```
