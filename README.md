# Subtitle Foundry

Cross-platform desktop app for turning spoken video into subtitle files and optional burned-in exports.

The first workflow is focused on Greek videos that need English subtitles, but the app also supports source-language subtitles and other source-language choices.

## What It Does

- Select one or more videos from anywhere on disk.
- Choose the source language or let Whisper detect it.
- Generate subtitles in English or in the source language.
- Export an `.srt` file.
- Optionally burn the subtitles directly into a new video file.
- Run fully locally with `openai-whisper` and a bundled FFmpeg binary.

## Why Local Whisper By Default

The hosted OpenAI Whisper API is usable, but it is not the best default for this repo because it adds cost and requires an API key. The app in this repository uses local `openai-whisper`, which is free to run on your own machine and works offline once the model has been downloaded.

## Quick Start

1. Install `uv` if you do not already have it: https://docs.astral.sh/uv/
2. Sync the project:

```bash
uv sync
```

3. Launch the desktop app:

```bash
uv run add-subtitles-to-videos
```

On Windows, you can also double-click `launch-subtitle-foundry.vbs` to start the app without leaving a terminal window open.

## Notes

- The first run downloads the selected Whisper model, so it can take a while.
- `medium` is a good starting point for Greek-to-English subtitle work, with `large-v3` as the quality upgrade.
- Avoid `turbo` for this specific job. Whisper translation is more reliable with the multilingual models above.
- On Windows x64, the project is pinned to the official CUDA 12.4 PyTorch wheel so NVIDIA GPUs can be used automatically when available.
- Burned-in subtitle exports are written with a new `.subtitled` suffix so your original files stay untouched.
- The app now surfaces active file, queue position, elapsed time, active engine, subtitle preview, and automatic review flags so you can tell when Whisper is loading or if the translation looks suspicious.

## Packaging Direction

- The app is built with `PySide6`, so the next packaging step is `pyside6-deploy` or a platform-specific packager such as PyInstaller.
- Plan to build native installers separately on Windows, macOS, and Linux.
- If you later want a faster GPU-focused backend on specific machines, `whisper.cpp` is the best follow-up option to evaluate.

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
