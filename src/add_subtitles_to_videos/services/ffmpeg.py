from __future__ import annotations

from collections import deque
from collections.abc import Callable
import re
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
from time import sleep

import imageio_ffmpeg

from . import OperationCancelledError

CancelChecker = Callable[[], bool]


class MediaToolError(RuntimeError):
    pass


def ffmpeg_binary() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def extract_audio(
    video_path: Path,
    audio_path: Path,
    *,
    cancel_requested: CancelChecker | None = None,
) -> None:
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg(
        [
            ffmpeg_binary(),
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(audio_path),
        ],
        cancel_requested=cancel_requested,
    )


_DURATION_PATTERN = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_VIDEO_SIZE_PATTERN = re.compile(r"Stream #.*?Video:.*?\b(\d{2,5})x(\d{2,5})\b")

# Audio files become a video at this frame rate. Landscape/no picture uses 1920x1080;
# a portrait picture gets a portrait canvas 1080 wide matching its shape.
COVER_VIDEO_FPS = 25
LANDSCAPE_CANVAS = (1920, 1080)
PORTRAIT_CANVAS_WIDTH = 1080
PORTRAIT_CANVAS_MAX_HEIGHT = 1920

# Intro/outro timing in seconds: logo held, blurred logo, crossfade length, and the
# gap before speech starts / after it ends inside the photo part.
INTRO_HOLD = 2.6
INTRO_BLUR = 1.2
INTRO_FADE = 0.6
SPEECH_LEAD = 0.4
SPEECH_TAIL = 0.6

# Waveform: dark dotted bars, centered horizontally, 68% down the frame.
WAVE_COLUMNS = 40
WAVE_ROWS = 18
WAVE_COLOR = "0x3a3a3a"
WAVE_Y = 0.683
WAVE_GAIN = 2.5


def _ffmpeg_summary(media_path: Path) -> str:
    completed = subprocess.run(
        [ffmpeg_binary(), "-hide_banner", "-i", str(media_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
    )
    return completed.stderr or ""


def probe_duration(media_path: Path) -> float:
    """Read the media duration in seconds from FFmpeg's input summary."""
    match = _DURATION_PATTERN.search(_ffmpeg_summary(media_path))
    if match is None:
        raise MediaToolError(f"Couldn't read the duration of {media_path.name}. Is it a valid audio or video file?")
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def probe_image_size(image_path: Path) -> tuple[int, int]:
    match = _VIDEO_SIZE_PATTERN.search(_ffmpeg_summary(image_path))
    if match is None:
        raise MediaToolError(f"Couldn't read the size of {image_path.name}. Is it a valid image?")
    return int(match.group(1)), int(match.group(2))


def canvas_for_picture(picture_size: tuple[int, int] | None) -> tuple[int, int]:
    """Portrait pictures get a portrait canvas of the same shape; everything else is 1920x1080."""
    if picture_size is None:
        return LANDSCAPE_CANVAS
    width, height = picture_size
    if height <= width:
        return LANDSCAPE_CANVAS
    canvas_height = round(PORTRAIT_CANVAS_WIDTH * height / width / 2) * 2
    return PORTRAIT_CANVAS_WIDTH, min(canvas_height, PORTRAIT_CANVAS_MAX_HEIGHT)


def _normalize_video(fps: int) -> str:
    # xfade needs every clip in the same size, rate, timebase and pixel format.
    return f"fps={fps},settb=AVTB,setsar=1,format=yuv420p"


def _waveform_filters(audio_label: str, canvas: tuple[int, int], fps: int, delay: float) -> list[str]:
    """Moving dotted waveform as an RGBA stream labelled [wave].

    A low-res showwaves image is scaled up in blocks, then cut into columns by a
    mask that also tapers the bars down to dots toward both edges.
    """
    width, _height = canvas
    pitch = max(round(width / 90), 4)
    unit = pitch // 2
    dot = round(pitch * 2 / 3)
    wave_w, wave_h = WAVE_COLUMNS * pitch, WAVE_ROWS * unit
    mask = (
        f"if(lt(mod(X,{pitch}),{dot})*lte(abs(Y-{wave_h}/2),"
        f"max({unit}/2+1,{wave_h}/2*(1-pow((X-{wave_w}/2)/({wave_w}/2),2)))),255,0)"
    )
    return [
        f"[{audio_label}]volume={WAVE_GAIN},showwaves=s={WAVE_COLUMNS}x{WAVE_ROWS}:mode=cline:rate={fps}:"
        f"scale=lin:draw=full:colors=white,format=gray,scale={wave_w}:{wave_h}:flags=neighbor[wshape]",
        f"color=c=black:s={wave_w}x{wave_h}:r={fps}:d={1 / fps:.4f},format=gray,"
        f"geq=lum='{mask}',loop=loop=-1:size=1[wmask]",
        "[wshape][wmask]blend=all_mode=multiply:shortest=1,gblur=sigma=0.8[walpha]",
        f"color=c={WAVE_COLOR}:s={wave_w}x{wave_h}:r={fps}[wcolor]",
        f"[wcolor][walpha]alphamerge,setpts=PTS+{delay:.3f}/TB[wave]",
    ]


def _intro_outro_filters(
    logo_input: int,
    main_label: str,
    main_duration: float,
    canvas: tuple[int, int],
    fps: int,
) -> list[str]:
    """Logo on white -> blurred, faded logo -> main -> blurred logo -> logo, labelled [v]."""
    width, height = canvas
    norm = _normalize_video(fps)
    fade_to_white = "lutrgb=r='val+(255-val)*0.55':g='val+(255-val)*0.55':b='val+(255-val)*0.55'"
    offset1 = INTRO_HOLD - INTRO_FADE
    offset2 = offset1 + INTRO_BLUR - INTRO_FADE
    offset3 = offset2 + main_duration - INTRO_FADE
    offset4 = offset3 + INTRO_BLUR - INTRO_FADE
    return [
        f"[{logo_input}:v]scale={round(width * 0.82)}:{round(height * 0.7)}:force_original_aspect_ratio=decrease,"
        "setsar=1[logo]",
        f"color=c=white:s={width}x{height}:r={fps}[card_bg]",
        f"[card_bg][logo]overlay=x=(W-w)/2:y=(H-h)/2:shortest=1,trim=duration={INTRO_HOLD},"
        "split=4[card1][card2][card3][card4]",
        f"[card1]{norm}[intro_logo]",
        f"[card2]{norm}[outro_logo]",
        f"[card3]trim=duration={INTRO_BLUR},gblur=sigma=28,{fade_to_white},{norm}[intro_blur]",
        f"[card4]trim=duration={INTRO_BLUR},gblur=sigma=28,{fade_to_white},{norm}[outro_blur]",
        f"[intro_logo][intro_blur]xfade=transition=fade:duration={INTRO_FADE}:offset={offset1:.3f}[x1]",
        f"[x1][{main_label}]xfade=transition=fade:duration={INTRO_FADE}:offset={offset2:.3f}[x2]",
        f"[x2][outro_blur]xfade=transition=fade:duration={INTRO_FADE}:offset={offset3:.3f}[x3]",
        f"[x3][outro_logo]xfade=transition=fade:duration={INTRO_FADE}:offset={offset4:.3f}[v]",
    ]


def intro_main_offset() -> float:
    """Seconds from the start of the video to where the main (photo) part begins."""
    return INTRO_HOLD + INTRO_BLUR - 2 * INTRO_FADE


def intro_outro_extra_seconds() -> float:
    return 2 * (INTRO_HOLD + INTRO_BLUR) - 4 * INTRO_FADE


def build_prepare_command(
    input_path: Path,
    output_path: Path,
    *,
    keep: list[tuple[float, float]],
    audio_only: bool,
    cover_image: Path | None,
    canvas: tuple[int, int] = LANDSCAPE_CANVAS,
    intro_image: Path | None = None,
    show_waveform: bool = False,
) -> list[str]:
    """Build an FFmpeg command that joins the kept parts of the input into one MP4.

    Each kept part is opened as its own input with -ss/-t so FFmpeg seeks straight
    to it, then the parts are joined with the concat filter. For audio input the
    picture is the cover image (or black), optionally with a moving waveform and a
    logo intro/outro. The intro and outro are silent, so Whisper's timestamps on
    the finished file still line up with the speech.
    """
    command = [ffmpeg_binary(), "-y", "-hide_banner"]
    for start, end in keep:
        command += ["-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", "-i", str(input_path)]

    count = len(keep)
    audio_inputs = "".join(f"[{index}:a:0]" for index in range(count))
    filters: list[str] = []
    if audio_only:
        fps = COVER_VIDEO_FPS
        width, height = canvas
        if cover_image is not None:
            command += ["-loop", "1", "-framerate", str(fps), "-i", str(cover_image)]
        else:
            command += ["-f", "lavfi", "-i", f"color=c=black:s={width}x{height}:r={fps}"]
        if intro_image is not None:
            command += ["-loop", "1", "-framerate", str(fps), "-i", str(intro_image)]

        speech_seconds = sum(end - start for start, end in keep)
        lead = SPEECH_LEAD if intro_image is not None else 0.0
        main_seconds = lead + speech_seconds + (SPEECH_TAIL if intro_image is not None else 0.0)

        if show_waveform:
            filters.append(f"{audio_inputs}concat=n={count}:v=0:a=1,asplit[speech][wave_audio]")
        else:
            filters.append(f"{audio_inputs}concat=n={count}:v=0:a=1[speech]")

        filters.append(
            # out_range=tv: JPEG photos are full-range; keep output in the standard range players expect.
            f"[{count}:v]scale={width}:{height}:force_original_aspect_ratio=decrease:out_range=tv,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,"
            f"trim=duration={main_seconds:.3f}[picture]"
        )
        main_label = "picture"
        if show_waveform:
            filters += _waveform_filters("wave_audio", canvas, fps, lead)
            filters.append(f"[picture][wave]overlay=x=(W-w)/2:y=H*{WAVE_Y}-h/2:eof_action=pass[main_raw]")
            main_label = "main_raw"

        if intro_image is not None:
            filters.append(f"[{main_label}]{_normalize_video(fps)}[main]")
            filters += _intro_outro_filters(count + 1, "main", main_seconds, canvas, fps)
            total_seconds = main_seconds + intro_outro_extra_seconds()
            delay_ms = round((intro_main_offset() + lead) * 1000)
            filters.append(f"[speech]adelay=delays={delay_ms}:all=1,apad=whole_dur={total_seconds:.3f}[a]")
        else:
            filters.append(f"[{main_label}]format=yuv420p[v]")
            filters.append("[speech]anull[a]")
            total_seconds = main_seconds

        tune = [] if show_waveform else ["-tune", "stillimage"]
        # The picture input never ends, so stop at the finished length.
        length = ["-t", f"{total_seconds:.3f}"]
    else:
        pairs = "".join(f"[{index}:v:0][{index}:a:0]" for index in range(count))
        filters.append(f"{pairs}concat=n={count}:v=1:a=1[vc][a]")
        filters.append("[vc]format=yuv420p[v]")
        tune = []
        length = []

    command += [
        "-filter_complex",
        ";".join(filters),
        "-map",
        "[v]",
        "-map",
        "[a]",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        *tune,
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        *length,
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    return command


def prepare_media(
    input_path: Path,
    output_path: Path,
    *,
    keep: list[tuple[float, float]],
    audio_only: bool,
    cover_image: Path | None = None,
    canvas: tuple[int, int] = LANDSCAPE_CANVAS,
    intro_image: Path | None = None,
    show_waveform: bool = False,
    cancel_requested: CancelChecker | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg(
        build_prepare_command(
            input_path,
            output_path,
            keep=keep,
            audio_only=audio_only,
            cover_image=cover_image,
            canvas=canvas,
            intro_image=intro_image,
            show_waveform=show_waveform,
        ),
        cancel_requested=cancel_requested,
    )


# Burned subtitle looks. Sizes are in libass units (relative to a 288-line frame).
BOXED_STYLE = "FontSize={size},OutlineColour=&H4A000000,BorderStyle=3,Outline=1,Shadow=0,MarginV=28"
OUTLINE_STYLE = (
    "FontName=Arial,FontSize={size},PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
    "BackColour=&H80000000,BorderStyle=1,Outline=1,Shadow=1,MarginV=34"
)


def burn_subtitles(
    video_path: Path,
    subtitle_path: Path,
    output_path: Path,
    *,
    font_size: int,
    outline_style: bool = False,
    cancel_requested: CancelChecker | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    style = (OUTLINE_STYLE if outline_style else BOXED_STYLE).format(size=font_size)

    with TemporaryDirectory(prefix="subtitle-foundry-burn-") as temp_dir:
        temp_path = Path(temp_dir)
        local_subtitle_path = temp_path / "captions.srt"
        shutil.copy2(subtitle_path, local_subtitle_path)
        filter_expression = f"subtitles={local_subtitle_path.name}:force_style='{style}'"
        _run_ffmpeg(
            [
                ffmpeg_binary(),
                "-y",
                "-i",
                str(video_path),
                "-vf",
                filter_expression,
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "copy",
                "-movflags",
                "+faststart",
                str(output_path),
            ],
            cwd=temp_path,
            cancel_requested=cancel_requested,
        )


def _run_ffmpeg(
    command: list[str],
    cwd: Path | None = None,
    *,
    cancel_requested: CancelChecker | None = None,
) -> None:
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
    )
    stderr_tail: deque[str] = deque(maxlen=24)
    stderr_thread = _start_stderr_reader(process, stderr_tail)

    try:
        while True:
            if cancel_requested is not None and cancel_requested():
                _terminate_process(process)
                raise OperationCancelledError("Processing stopped by user.")

            result_code = process.poll()
            if result_code is None:
                sleep(0.1)
                continue
            break
    finally:
        stderr_thread.join(timeout=2)

    if result_code == 0:
        return

    detail = "\n".join(stderr_tail).strip()
    raise MediaToolError(
        "FFmpeg failed while processing media.\n"
        f"Command: {' '.join(command)}\n"
        f"Details:\n{detail}"
    )


def _start_stderr_reader(
    process: subprocess.Popen[str],
    stderr_tail: deque[str],
) -> Thread:
    def drain() -> None:
        stream = process.stderr
        if stream is None:
            return

        try:
            for line in iter(stream.readline, ""):
                if not line:
                    break
                stripped = line.rstrip()
                if stripped:
                    stderr_tail.append(stripped)
        finally:
            stream.close()

    thread = Thread(target=drain, daemon=True)
    thread.start()
    return thread


def _terminate_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return

    process.terminate()
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2)
