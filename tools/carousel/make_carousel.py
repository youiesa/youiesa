"""Render a TikTok carousel in Alice's style.

Usage: python3 make_carousel.py carousel.json photos_dir out_dir

carousel.json: {"slides": [{"photo": "IMG_1.JPG", "text": ["headline", "body"],
                            "focus": [0.5, 0.4]}, ...]}
Each entry in "text" becomes its own white box; long entries wrap to stepped lines.
"focus" is the point (fractions of width/height) the 3:4 crop keeps in frame.
"""
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

W, H = 1080, 1440                 # 3:4, matches her posted slides
FONT = Path(__file__).parent / "fonts" / "tts-600.ttf"
SIZE = 36                         # calibrated against her posted slides
PAD_X, PAD_Y = 28, 12             # inside each line's box
BASELINE = 44                     # box top to first baseline
PITCH = 47                        # baseline-to-baseline
BOX_GAP = 21                      # between headline and body boxes
RADIUS = 12
MAX_TEXT_W = 830
CENTRE_Y = 0.49                   # vertical centre of the text block
TEXT_RANGE = (0.28, 0.76)         # how far the block may move to clear a face
FACE_MODEL = Path(__file__).parent / "yunet.onnx"


def find_faces(img):
    bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    det = cv2.FaceDetectorYN.create(str(FACE_MODEL), "", (W, H), 0.6)
    _, faces = det.detect(bgr)
    if faces is None:
        return []
    # pad each face so text doesn't sit right on the hairline or chin
    return [(x - w * 0.15, y - h * 0.2, x + w * 1.15, y + h * 1.15)
            for x, y, w, h in faces[:, :4]]


def overlap(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def place_text(faces, width, height, preferred):
    """Centre y for the text block: nearest the preferred spot that covers no face."""
    def cost(cy):
        rect = ((W - width) / 2, cy - height / 2, (W + width) / 2, cy + height / 2)
        return (sum(overlap(rect, f) for f in faces), abs(cy - preferred))
    lo, hi = (int(H * t) for t in TEXT_RANGE)
    return min(range(lo, hi + 1, 10), key=cost)


def crop_3x4(img, focus):
    img = ImageOps.exif_transpose(img).convert("RGB")
    w, h = img.size
    if w / h > W / H:
        cw, ch = round(h * W / H), h
    else:
        cw, ch = w, round(w * H / W)
    fx, fy = focus
    x = min(max(round(fx * w - cw / 2), 0), w - cw)
    y = min(max(round(fy * h - ch / 2), 0), h - ch)
    return img.crop((x, y, x + cw, y + ch)).resize((W, H), Image.LANCZOS)


def wrap(text, font):
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if font.getlength(trial) <= MAX_TEXT_W or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    lines.append(cur)
    # balance two-line wraps so the last line isn't a lonely word
    if len(lines) == 2:
        words = text.split()
        best = min(range(1, len(words)), key=lambda i: abs(
            font.getlength(" ".join(words[:i])) - font.getlength(" ".join(words[i:]))))
        a, b = " ".join(words[:best]), " ".join(words[best:])
        if max(font.getlength(a), font.getlength(b)) <= MAX_TEXT_W:
            lines = [a, b]
    return lines


def block_height(n):
    # one line = 71px, as measured on her slides; each extra line adds PITCH
    return n * PITCH + 2 * PAD_Y


def draw_box(draw, font, lines, top):
    # one rounded rectangle per line, overlapping, so lines of different
    # length step in like TikTok's own text boxes
    for i, line in enumerate(lines):
        tw = font.getlength(line)
        x0 = (W - tw) / 2 - PAD_X
        y0 = top + i * PITCH
        draw.rounded_rectangle((x0, y0, x0 + tw + 2 * PAD_X, y0 + PITCH + 2 * PAD_Y),
                               RADIUS, fill="white")
    for i, line in enumerate(lines):
        draw.text((W / 2, top + BASELINE + i * PITCH), line, font=font,
                  fill="black", anchor="ms")


def render(slide, photos):
    if re.search(r"\[[^\]]*\]", " ".join(slide["text"])):
        raise SystemExit(f"unfilled [bracket] in: {slide['text']}")
    img = crop_3x4(Image.open(photos / slide["photo"]), slide.get("focus", (0.5, 0.5)))
    font = ImageFont.truetype(str(FONT), SIZE)
    # curly apostrophes, as TikTok's text tool sets them
    boxes = [wrap(t.replace("'", "\u2019"), font) for t in slide["text"]]
    total = sum(block_height(len(b)) for b in boxes) + BOX_GAP * (len(boxes) - 1)
    width = max(font.getlength(line) for b in boxes for line in b) + 2 * PAD_X
    if "text_y" in slide:
        centre = H * slide["text_y"]
    else:
        centre = place_text(find_faces(img), width, total, H * CENTRE_Y)
    top = centre - total / 2
    draw = ImageDraw.Draw(img)
    for lines in boxes:
        draw_box(draw, font, lines, top)
        top += block_height(len(lines)) + BOX_GAP
    return img


def main(spec, photos, out):
    spec, photos, out = json.loads(Path(spec).read_text()), Path(photos), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    used = [s["photo"] for s in spec["slides"]]
    if len(set(used)) != len(used):
        raise SystemExit("a photo is used twice in this carousel")
    for i, slide in enumerate(spec["slides"], 1):
        # saved without EXIF, so no GPS/location data leaves with the slide
        render(slide, photos).save(out / f"slide-{i}.jpg", quality=95)
        print(f"slide-{i}.jpg  <- {slide['photo']}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
