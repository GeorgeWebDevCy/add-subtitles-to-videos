try:
    from .main import main
except ImportError:  # pragma: no cover - supports PyInstaller script execution
    from add_subtitles_to_videos.main import main


if __name__ == "__main__":
    main()
