"""
drawing.py  --  STAGE 5 of the pipeline:  draw bbox + label onto the frame.

============================ THE BIG PICTURE ================================
Detection gives us NUMBERS (box coordinates + confidence). Numbers are hard
to eyeball. So we draw those boxes back ONTO the image, plus a little text
label, and show it in a window. This is purely for US, the humans, to confirm
"yes, it's finding the people correctly". The neural network doesn't need
this; our eyes do. This is called VISUALIZATION or an "annotated frame".

Everything here is done with OpenCV's simple drawing functions, which paint
directly onto the pixel array (they modify the image in place).

Reminder: OpenCV colours are (Blue, Green, Red), each 0-255. So:
  (0, 255, 0)   = pure green
  (0, 0, 255)   = pure red
  (255, 0, 0)   = pure blue
============================================================================
"""

import cv2

# The colour we draw person boxes in. Green stands out against most scenes.
BOX_COLOR = (0, 255, 0)          # BGR = green
TEXT_COLOR = (0, 0, 0)           # BGR = black (drawn on a green label patch)
BOX_THICKNESS = 2                # line thickness in pixels
FONT = cv2.FONT_HERSHEY_SIMPLEX  # a built-in OpenCV font


def draw_detections(frame, detections):
    """
    Draw every detection onto `frame` and return the annotated frame.

    NOTE: OpenCV draws IN PLACE (it edits the array you pass in). We return it
    too, just so the calling code reads naturally.
    """
    for det in detections:
        # 1) The rectangle. cv2.rectangle needs the two opposite corners:
        #    top-left (x1, y1) and bottom-right (x2, y2).
        cv2.rectangle(
            frame,
            (det.x1, det.y1),   # top-left corner
            (det.x2, det.y2),   # bottom-right corner
            BOX_COLOR,
            BOX_THICKNESS,
        )

        # 2) A text label, e.g. "person 0.87", so we can read the confidence.
        label = f"person {det.confidence:.2f}"

        # Measure how big that text will be, so we can draw a filled
        # background patch behind it -- otherwise white-on-white text vanishes.
        (text_w, text_h), baseline = cv2.getTextSize(label, FONT, 0.5, 1)

        # Draw the filled label background just ABOVE the box's top-left corner.
        cv2.rectangle(
            frame,
            (det.x1, det.y1 - text_h - baseline - 4),  # top-left of patch
            (det.x1 + text_w, det.y1),                 # bottom-right of patch
            BOX_COLOR,
            thickness=-1,   # -1 means "fill the rectangle solid"
        )

        # Draw the text on top of that patch.
        cv2.putText(
            frame,
            label,
            (det.x1, det.y1 - baseline - 2),  # bottom-left anchor of the text
            FONT,
            0.5,             # font scale (size)
            TEXT_COLOR,
            1,               # text thickness
            cv2.LINE_AA,     # anti-aliased = smoother-looking text
        )

    return frame


def draw_hud(frame, person_count, fps=None):
    """
    Draw a small "heads-up display" in the top-left: how many people are in
    view, and (optionally) how fast we're processing, in frames per second.
    Handy for confirming the pipeline is keeping up in real time.
    """
    text = f"People: {person_count}"
    if fps is not None:
        text += f"   FPS: {fps:.1f}"

    cv2.putText(frame, text, (10, 30), FONT, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    return frame
