# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules


ROOT = Path.cwd()
SRC = ROOT / "src"
ASSET_DIR = ROOT / "assets" / "branding"

datas = []
datas += collect_data_files("whisper")
datas += collect_data_files("imageio_ffmpeg")

binaries = []
binaries += collect_dynamic_libs("torch")

hiddenimports = []
hiddenimports += collect_submodules("tiktoken_ext")
hiddenimports += collect_submodules("whisper")


a = Analysis(
    [str(SRC / "add_subtitles_to_videos" / "__main__.py")],
    pathex=[str(ROOT), str(SRC)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="SubtitleFoundry",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    icon=str(ASSET_DIR / "subtitle-foundry-icon.ico"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="SubtitleFoundry",
)
