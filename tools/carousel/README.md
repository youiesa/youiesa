# Carousel maker

Turns a carousel script plus photos into TikTok slides in Alice's style:
1080×1440 (3:4), white rounded text boxes, black TikTok Sans SemiBold, curly
apostrophes, boxes nudged up or down so they never sit on a face. Slides are
saved without EXIF, so no GPS location leaves with them.

    pip install -r requirements.txt
    python3 make_carousel.py carousels/2026-10-26.json /path/to/photos out/2026-10-26

Each slide in the JSON names a photo, its text (one entry per white box) and
optionally `focus` (where to keep the 3:4 crop centred) or `text_y` (fixed
text height, overriding face avoidance). It refuses to render text that still
contains a `[placeholder]`, or a carousel that uses one photo twice.

Photos are not kept in this repo.

- `fonts/tts-600.ttf`: TikTok Sans, SIL Open Font License (Google Fonts).
- `yunet.onnx`: YuNet face detector from opencv_zoo (MIT licence).
