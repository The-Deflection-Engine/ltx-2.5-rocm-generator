#!/usr/bin/env python3
"""Regenerate the app icon set.

    python icons/make_icon.py

Draws at 1024 and downsamples, so the small sizes stay clean rather than being
drawn with sub-pixel geometry. Colours match the GUI's own accents (the cyan of
the telemetry strip, the green of the output label).
"""
import os
from PIL import Image, ImageDraw

OUT = os.path.dirname(os.path.abspath(__file__))
BG      = (17, 17, 17, 255)      # telemetry panel background
ACCENT  = (0, 255, 204, 255)     # #00ffcc, the telemetry cyan
ACCENT2 = (0, 170, 119, 255)     # #0a7, the output-label green

def draw(size=1024):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    r = int(size * 0.18)
    d.rounded_rectangle([0, 0, size - 1, size - 1], radius=r, fill=BG)

    # Film-strip sprocket columns down each side.
    hole_w, hole_h = int(size * 0.075), int(size * 0.055)
    gap = int(size * 0.036)
    x_l, x_r = int(size * 0.075), size - int(size * 0.075) - hole_w
    y = int(size * 0.13)
    while y + hole_h < size - int(size * 0.13):
        for x in (x_l, x_r):
            d.rounded_rectangle([x, y, x + hole_w, y + hole_h],
                                radius=int(hole_h * 0.28), fill=ACCENT2)
        y += hole_h + gap

    # Play triangle, centred in the frame between the sprocket columns.
    cx, cy = size / 2, size / 2
    h = size * 0.34
    w = h * 0.88
    d.polygon([(cx - w * 0.45, cy - h / 2),
               (cx - w * 0.45, cy + h / 2),
               (cx + w * 0.62, cy)], fill=ACCENT)
    return img

def main():
    master = draw(1024)
    master.save(os.path.join(OUT, "ltx25.png"))
    for s in (256, 128, 64, 48, 32, 24, 16):
        master.resize((s, s), Image.LANCZOS).save(os.path.join(OUT, f"ltx25-{s}.png"))
    print("wrote", OUT)

if __name__ == "__main__":
    main()
