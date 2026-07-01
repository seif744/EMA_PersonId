"""
detector.py  --  STAGE 3 of the pipeline:  frame  ->  person model  ->  bbox, classid.

============================ THE BIG PICTURE ================================
We now have a single frame (a grid of pixels). We want to answer:
"WHERE are the people in this image?"

That job is called OBJECT DETECTION. We use a pre-trained neural network
called YOLO ("You Only Look Once"). It was trained on a giant dataset called
COCO, which contains 80 kinds of objects (person, car, dog, chair, ...).
Each object type has a number, its CLASS ID. For COCO, person == 0.

For each object it finds, YOLO returns three things:
  1. bbox  (BOUNDING BOX): four numbers (x1, y1, x2, y2) = the pixel
            coordinates of the top-left and bottom-right corners of a
            rectangle drawn tightly around the object.
  2. classid: which of the 80 categories it thinks this is (we keep only 0).
  3. confidence: how sure the model is, from 0.0 to 1.0.

We wrap all this so the rest of the program gets back a simple, clean list of
"here are the people", and never has to know YOLO's internal details.

WHY do we even need a neural net? Because "find every person, at any size,
pose, lighting, or clothing" is impossible to write by hand with if-statements.
YOLO learned it from millions of labelled example images.
============================================================================
"""

from dataclasses import dataclass
from ultralytics import YOLO  # the library that loads & runs YOLO models.


@dataclass
class Detection:
    """
    One detected person. A dataclass is just a lightweight container with
    named fields -- clearer than passing around a raw tuple or dict.

    Coordinates are in PIXELS, measured from the top-left of the image:
      x grows to the right, y grows DOWNWARD (standard for images).
    """
    x1: int          # left edge of the box
    y1: int          # top edge of the box
    x2: int          # right edge of the box
    y2: int          # bottom edge of the box
    confidence: float  # how sure YOLO is (0.0 - 1.0)
    class_id: int      # which COCO class (will always be 0 = person here)


class PersonDetector:
    """
    Loads the YOLO model once, then detects people in each frame you give it.

    We load the model ONCE (in __init__) because loading is slow, but running
    it on a frame is comparatively fast. You never want to reload per frame.
    """

    def __init__(self, model_path="yolo11n.pt",
                 confidence_threshold=0.4, person_class_id=0):
        # Loading the weights. The first time this runs, Ultralytics will
        # DOWNLOAD the yolo11n.pt file (~5 MB) automatically, then reuse it.
        print(f"[detector] Loading model '{model_path}' (first run may download it)...")
        self.model = YOLO(model_path)

        self.confidence_threshold = confidence_threshold
        self.person_class_id = person_class_id
        print("[detector] Model ready.")

    def detect(self, frame):
        """
        Run the model on ONE frame and return a list[Detection] of people.

        Steps:
          1. Feed the frame to YOLO.
          2. YOLO returns ALL objects it found (people, cars, etc.).
          3. We keep only class 0 (person) above our confidence threshold.
        """
        # We pass our thresholds straight into YOLO so it does the filtering
        # efficiently on its side:
        #   classes=[person_class_id] -> only return people, ignore the other 79 classes.
        #   conf=...                  -> drop anything below our confidence bar.
        #   verbose=False             -> don't spam the console every frame.
        results = self.model(
            frame,
            classes=[self.person_class_id],
            conf=self.confidence_threshold,
            verbose=False,
        )

        detections = []

        # `results` is a list (one entry per image; we passed one image, so
        # one entry). Each result has a `.boxes` holding every detection.
        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                # box.xyxy is a tensor shaped (1, 4): [[x1, y1, x2, y2]].
                # .tolist() turns the tensor into a plain Python list so we can
                # read the numbers out normally.
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])

                detections.append(
                    Detection(
                        # We round to whole pixels -- you can't draw at pixel 12.7.
                        x1=int(round(x1)),
                        y1=int(round(y1)),
                        x2=int(round(x2)),
                        y2=int(round(y2)),
                        confidence=confidence,
                        class_id=class_id,
                    )
                )

        return detections
