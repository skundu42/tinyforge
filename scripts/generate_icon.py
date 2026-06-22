"""Generate the TinyForge macOS app icon into an .appiconset.

Renders at 4x supersample for crisp edges: a rounded "squircle" with a deep
indigo gradient, a soft drop shadow, a top highlight, and a warm forge-spark
(4-point sparkle) with a white-hot core and glow — "forging a spark of
intelligence".

Run: uv run --with pillow --with numpy python scripts/generate_icon.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

OUT = Path(__file__).resolve().parent.parent / "App" / "Sources" / "Assets.xcassets" / "AppIcon.appiconset"
SS = 4  # supersample factor
BASE = 1024


def _lerp(a, b, t):
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


def vertical_gradient(size, top, bottom):
    grad = np.zeros((size, size, 3), dtype=np.uint8)
    for y in range(size):
        grad[y, :, :] = _lerp(top, bottom, y / (size - 1))
    return Image.fromarray(grad, "RGBA".replace("A", "")).convert("RGBA")


def radial_glow(size, center, radius, color):
    img = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(img)
    cx, cy = center
    d.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=255)
    img = img.filter(ImageFilter.GaussianBlur(radius * 0.45))
    glow = Image.new("RGBA", (size, size), color + (0,))
    glow.putalpha(img)
    return glow


def star(center, r_outer, r_inner, points=4, rotation=-math.pi / 2):
    cx, cy = center
    verts = []
    for i in range(points * 2):
        angle = rotation + i * math.pi / points
        radius = r_outer if i % 2 == 0 else r_inner
        verts.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    return verts


def render(size):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))

    margin = int(size * 0.085)
    box = (margin, margin, size - margin, size - margin)
    radius = int((box[2] - box[0]) * 0.235)

    # Drop shadow.
    shadow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    sd.rounded_rectangle(
        (box[0], box[1] + int(size * 0.018), box[2], box[3] + int(size * 0.018)),
        radius=radius, fill=(0, 0, 0, 130),
    )
    img = Image.alpha_composite(img, shadow.filter(ImageFilter.GaussianBlur(size * 0.02)))

    # Squircle body with indigo gradient.
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle(box, radius=radius, fill=255)
    body = vertical_gradient(size, (74, 60, 138), (26, 20, 51))  # #4A3C8A -> #1A1433
    img.paste(body, (0, 0), mask)

    # Subtle top highlight.
    hi = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    hd = ImageDraw.Draw(hi)
    hd.rounded_rectangle(
        (box[0], box[1], box[2], box[1] + int((box[3] - box[1]) * 0.5)),
        radius=radius, fill=(255, 255, 255, 26),
    )
    hi_masked = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    hi_masked.paste(hi, (0, 0), mask)
    img = Image.alpha_composite(img, hi_masked.filter(ImageFilter.GaussianBlur(size * 0.01)))

    center = (size // 2, int(size * 0.52))

    # Warm glow behind the spark.
    img = Image.alpha_composite(
        img, radial_glow(size, center, int(size * 0.30), (255, 150, 60))
    )

    # The forge-spark: a large 4-point star + a small twinkle.
    spark_layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    sl = ImageDraw.Draw(spark_layer)
    r_out = int(size * 0.255)
    big = star(center, r_out, int(r_out * 0.30))
    sl.polygon(big, fill=(255, 196, 84, 255))  # gold
    # warm lower half overlay for gradient feel
    grad_mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(grad_mask).polygon(big, fill=255)
    warm = vertical_gradient(size, (255, 214, 130), (255, 120, 61))
    spark = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    spark.paste(warm, (0, 0), grad_mask)
    img = Image.alpha_composite(img, spark)

    # White-hot core.
    core = radial_glow(size, center, int(size * 0.085), (255, 245, 230))
    img = Image.alpha_composite(img, core)
    cd = ImageDraw.Draw(img)
    cr = int(size * 0.045)
    cd.ellipse((center[0] - cr, center[1] - cr, center[0] + cr, center[1] + cr),
               fill=(255, 252, 245, 255))

    # Small twinkle upper-right.
    tw_center = (int(size * 0.66), int(size * 0.34))
    tr = int(size * 0.07)
    sl2 = ImageDraw.Draw(img)
    sl2.polygon(star(tw_center, tr, int(tr * 0.28)), fill=(255, 235, 200, 235))

    return img


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    master = render(BASE * SS).resize((BASE, BASE), Image.LANCZOS)

    specs = [
        (16, 1), (16, 2), (32, 1), (32, 2),
        (128, 1), (128, 2), (256, 1), (256, 2), (512, 1), (512, 2),
    ]
    images = []
    seen = {}
    for pt, scale in specs:
        px = pt * scale
        if px not in seen:
            seen[px] = f"icon_{px}.png"
            master.resize((px, px), Image.LANCZOS).save(OUT / seen[px])
        images.append({
            "idiom": "mac", "size": f"{pt}x{pt}", "scale": f"{scale}x",
            "filename": seen[px],
        })

    contents = {"images": images, "info": {"version": 1, "author": "tinyforge"}}
    (OUT / "Contents.json").write_text(json.dumps(contents, indent=2))
    print(f"Wrote {len(seen)} PNGs + Contents.json to {OUT}")


if __name__ == "__main__":
    main()
