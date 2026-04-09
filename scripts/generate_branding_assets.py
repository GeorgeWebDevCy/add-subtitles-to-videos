from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parent.parent
ASSET_DIR = REPO_ROOT / "assets" / "branding"


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    create_app_icon()
    create_installer_sidebar()
    create_installer_small_image()


def create_app_icon() -> None:
    size = 1024
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    draw.rounded_rectangle(
        (80, 80, size - 80, size - 80),
        radius=220,
        fill=(20, 54, 66, 255),
    )
    draw.rounded_rectangle(
        (140, 140, size - 140, size - 140),
        radius=180,
        fill=(31, 94, 104, 255),
    )
    draw.polygon(
        [
            (220, 250),
            (804, 250),
            (730, 730),
            (294, 730),
        ],
        fill=(255, 248, 239, 255),
    )
    draw.rectangle((316, 342, 708, 386), fill=(227, 100, 20, 255))
    draw.rectangle((316, 438, 672, 482), fill=(227, 100, 20, 230))
    draw.rectangle((316, 534, 624, 578), fill=(227, 100, 20, 210))
    draw.rectangle((316, 630, 580, 674), fill=(227, 100, 20, 190))

    font = ImageFont.load_default(size=170)
    draw.text((355, 112), "A", fill=(255, 248, 239, 215), font=font)

    png_path = ASSET_DIR / "subtitle-foundry-icon.png"
    ico_path = ASSET_DIR / "subtitle-foundry-icon.ico"
    image.save(png_path)
    image.save(ico_path, sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)])


def create_installer_sidebar() -> None:
    width, height = 164, 314
    image = Image.new("RGB", (width, height), (20, 54, 66))
    draw = ImageDraw.Draw(image)

    for offset in range(height):
        blend = offset / height
        color = (
            int(20 + (40 - 20) * blend),
            int(54 + (114 - 54) * blend),
            int(66 + (113 - 66) * blend),
        )
        draw.line((0, offset, width, offset), fill=color)

    draw.rounded_rectangle((18, 20, 146, 148), radius=28, fill=(255, 248, 239))
    draw.polygon([(42, 42), (122, 42), (112, 110), (52, 110)], fill=(20, 54, 66))
    draw.rectangle((54, 56, 108, 64), fill=(227, 100, 20))
    draw.rectangle((54, 74, 102, 82), fill=(227, 100, 20))
    draw.rectangle((54, 92, 96, 100), fill=(227, 100, 20))

    title_font = ImageFont.load_default(size=20)
    body_font = ImageFont.load_default(size=14)
    draw.text((20, 172), "Subtitle", fill=(255, 248, 239), font=title_font)
    draw.text((20, 194), "Foundry", fill=(255, 248, 239), font=title_font)
    draw.text((20, 232), "Greek video", fill=(255, 240, 221), font=body_font)
    draw.text((20, 250), "to English subtitles", fill=(255, 240, 221), font=body_font)
    draw.text((20, 268), "and burned-in exports", fill=(255, 240, 221), font=body_font)

    image.save(ASSET_DIR / "installer-sidebar.bmp")


def create_installer_small_image() -> None:
    width, height = 55, 55
    image = Image.new("RGB", (width, height), (255, 248, 239))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((4, 4, 51, 51), radius=12, fill=(20, 54, 66))
    draw.polygon([(14, 14), (41, 14), (37, 40), (18, 40)], fill=(255, 248, 239))
    draw.rectangle((18, 20, 36, 23), fill=(227, 100, 20))
    draw.rectangle((18, 27, 33, 30), fill=(227, 100, 20))
    draw.rectangle((18, 34, 30, 37), fill=(227, 100, 20))
    image.save(ASSET_DIR / "installer-small.bmp")


if __name__ == "__main__":
    main()
