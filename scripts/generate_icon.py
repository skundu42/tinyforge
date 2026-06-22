"""Generate the TinyForge logo: an actual forge — an anvil with a glowing ember
being struck — on the deep-indigo squircle.

Renders at 4x supersample, then emits:
  * App/Sources/Assets.xcassets/AppIcon.appiconset  (the macOS app icon)
  * App/Sources/Assets.xcassets/AppLogo.imageset     (the same art, for in-app use)

so the icon and every in-app logo are literally the same image.

Run: uv run --with pillow --with numpy python scripts/generate_icon.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ASSETS = Path(__file__).resolve().parent.parent / "App" / "Sources" / "Assets.xcassets"
ICON_OUT = ASSETS / "AppIcon.appiconset"
LOGO_OUT = ASSETS / "AppLogo.imageset"
SS = 4
BASE = 1024


def _lerp(a, b, t):
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


def vertical_gradient(size, top, bottom):
    grad = np.zeros((size, size, 3), dtype=np.uint8)
    for y in range(size):
        grad[y, :, :] = _lerp(top, bottom, y / (size - 1))
    return Image.fromarray(grad, "RGB").convert("RGBA")


def radial_glow(size, center, radius, color, blur=0.45):
    img = Image.new("L", (size, size), 0)
    cx, cy = center
    ImageDraw.Draw(img).ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=255)
    img = img.filter(ImageFilter.GaussianBlur(radius * blur))
    glow = Image.new("RGBA", (size, size), color + (0,))
    glow.putalpha(img)
    return glow


def star(center, r_outer, r_inner, points=4, rotation=-math.pi / 2):
    cx, cy = center
    return [
        (cx + (r_outer if i % 2 == 0 else r_inner) * math.cos(rotation + i * math.pi / points),
         cy + (r_outer if i % 2 == 0 else r_inner) * math.sin(rotation + i * math.pi / points))
        for i in range(points * 2)
    ]


def anvil_points(size):
    """Classic anvil silhouette (horn left, waisted body, flared base)."""
    aw, ah = 0.54 * size, 0.40 * size
    left, top = (size - aw) / 2, 0.46 * size
    norm = [
        (0.05, 0.14),  # horn tip
        (0.30, 0.00),  # face top-left
        (0.97, 0.00),  # face top-right
        (0.97, 0.23),  # face right edge
        (0.67, 0.29),  # right overhang under the face
        (0.61, 0.43),  # pedestal upper-right
        (0.71, 1.00),  # base lower-right
        (0.29, 1.00),  # base lower-left
        (0.39, 0.43),  # pedestal upper-left
        (0.33, 0.29),  # left overhang under the face
        (0.30, 0.25),  # face bottom-left (into the horn)
    ]
    return [(left + nx * aw, top + ny * ah) for nx, ny in norm]


def render(size):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    margin = int(size * 0.085)
    box = (margin, margin, size - margin, size - margin)
    radius = int((box[2] - box[0]) * 0.235)

    # Drop shadow.
    shadow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle(
        (box[0], box[1] + int(size * 0.018), box[2], box[3] + int(size * 0.018)),
        radius=radius, fill=(0, 0, 0, 130))
    img = Image.alpha_composite(img, shadow.filter(ImageFilter.GaussianBlur(size * 0.02)))

    # Squircle body.
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle(box, radius=radius, fill=255)
    body = vertical_gradient(size, (74, 60, 138), (26, 20, 51))  # #4A3C8A -> #1A1433
    img.paste(body, (0, 0), mask)

    # Top highlight.
    hi = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    ImageDraw.Draw(hi).rounded_rectangle(
        (box[0], box[1], box[2], box[1] + int((box[3] - box[1]) * 0.5)),
        radius=radius, fill=(255, 255, 255, 24))
    hi_masked = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    hi_masked.paste(hi, (0, 0), mask)
    img = Image.alpha_composite(img, hi_masked.filter(ImageFilter.GaussianBlur(size * 0.01)))

    # Anvil (steel gradient masked by the silhouette).
    anvil = anvil_points(size)
    anvil_mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(anvil_mask).polygon(anvil, fill=255)
    steel = vertical_gradient(size, (236, 233, 251), (150, 142, 196))  # lavender steel
    steel_layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    steel_layer.paste(steel, (0, 0), anvil_mask)
    # soft contact shadow under the anvil
    img = Image.alpha_composite(
        img, radial_glow(size, (size // 2, int(size * 0.86)), int(size * 0.22), (0, 0, 0), blur=0.5))
    img = Image.alpha_composite(img, steel_layer)

    # The ember being forged: glow + spark + white-hot core, on the anvil face.
    spark_center = (int(size * 0.50), int(size * 0.345))
    img = Image.alpha_composite(img, radial_glow(size, spark_center, int(size * 0.21), (255, 150, 60)))
    big = star(spark_center, int(size * 0.155), int(size * 0.046))
    grad_mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(grad_mask).polygon(big, fill=255)
    warm = vertical_gradient(size, (255, 216, 130), (255, 120, 61))
    spark = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    spark.paste(warm, (0, 0), grad_mask)
    img = Image.alpha_composite(img, spark)
    img = Image.alpha_composite(img, radial_glow(size, spark_center, int(size * 0.05), (255, 246, 232)))
    cr = int(size * 0.028)
    ImageDraw.Draw(img).ellipse(
        (spark_center[0] - cr, spark_center[1] - cr, spark_center[0] + cr, spark_center[1] + cr),
        fill=(255, 252, 245, 255))

    # A couple of flying spark particles.
    drw = ImageDraw.Draw(img)
    for dx, dy, pr in [(0.10, -0.07, 0.012), (0.13, 0.02, 0.008), (-0.11, -0.04, 0.009)]:
        px, py = int(size * (0.50 + dx)), int(size * (0.345 + dy))
        rr = int(size * pr)
        drw.ellipse((px - rr, py - rr, px + rr, py + rr), fill=(255, 214, 150, 235))

    return img


def write_imageset(master, out, idiom, specs):
    out.mkdir(parents=True, exist_ok=True)
    images, seen = [], {}
    for pt, scale in specs:
        px = pt * scale
        if px not in seen:
            seen[px] = f"img_{px}.png"
            master.resize((px, px), Image.LANCZOS).save(out / seen[px])
        images.append({"idiom": idiom, "size": f"{pt}x{pt}", "scale": f"{scale}x", "filename": seen[px]})
    (out / "Contents.json").write_text(
        json.dumps({"images": images, "info": {"version": 1, "author": "tinyforge"}}, indent=2))


def main():
    master = render(BASE * SS).resize((BASE, BASE), Image.LANCZOS)

    write_imageset(master, ICON_OUT, "mac", [
        (16, 1), (16, 2), (32, 1), (32, 2), (128, 1), (128, 2),
        (256, 1), (256, 2), (512, 1), (512, 2)])

    # The same art, as a universal in-app logo (no size key for universal images).
    LOGO_OUT.mkdir(parents=True, exist_ok=True)
    images = []
    for scale, px in [(1, 128), (2, 256), (3, 384)]:
        name = f"logo_{px}.png"
        master.resize((px, px), Image.LANCZOS).save(LOGO_OUT / name)
        images.append({"idiom": "universal", "scale": f"{scale}x", "filename": name})
    (LOGO_OUT / "Contents.json").write_text(
        json.dumps({"images": images, "info": {"version": 1, "author": "tinyforge"}}, indent=2))

    print("Wrote AppIcon.appiconset + AppLogo.imageset (anvil-and-ember forge).")


if __name__ == "__main__":
    main()
