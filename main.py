"""
main.py  --  the ENTRY POINT. Run this file to start the pipeline.

    python main.py

This is the "conductor": it doesn't do any heavy lifting itself. It just wires
together the pieces we built and runs them in a loop, in the exact order of
your drawing:

    STAGE 1-2   VideoSource      RTSP/file  ->  frame
    STAGE 3     PersonDetector   frame      ->  [Detection, Detection, ...]
    STAGE 5     drawing          detections ->  annotated frame  ->  screen

That's the whole current scope. Tracking (stage 4), embeddings (stage 6),
cross-camera matching (7) and clustering (8) are deliberately NOT here yet --
they're the next increments.
"""

import time
import yaml   # reads our config.yaml (installed alongside ultralytics)
import cv2

from src.video_source import VideoSource
from src.detector import PersonDetector
from src.drawing import draw_detections, draw_hud


def load_config(path="config.yaml"):
    """Read config.yaml into a plain Python dictionary."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    # ---- Load settings ------------------------------------------------------
    cfg = load_config()
    src_cfg = cfg["source"]
    det_cfg = cfg["detector"]
    disp_cfg = cfg["display"]

    # ---- STAGE 3 setup: build the detector ONCE, before the loop ------------
    # (Loading the model is slow; we do it a single time and reuse it.)
    detector = PersonDetector(
        model_path=det_cfg["model"],
        confidence_threshold=det_cfg["confidence_threshold"],
        person_class_id=det_cfg["person_class_id"],
    )

    # Optional: a video writer, only if the user set display.save_path.
    writer = None

    # ---- STAGE 1-2: open the video file, then loop over frames --------------
    # The `with` block guarantees the file handle is released when we're done.
    with VideoSource(path=src_cfg["url"]) as cam:

        print("[main] Pipeline running. Press 'q' in the window to quit.")

        # These two are just for measuring FPS (frames processed per second).
        prev_time = time.time()
        fps = 0.0

        for frame in cam.frames():          # <-- STAGE 1-2 yields each frame

            # ---- STAGE 3: detect people in this frame -----------------------
            detections = detector.detect(frame)

            # ---- STAGE 5: draw the results ----------------------------------
            frame = draw_detections(frame, detections)

            # Compute a smoothed FPS number just for the on-screen display.
            now = time.time()
            instant_fps = 1.0 / max(now - prev_time, 1e-6)  # avoid divide-by-0
            fps = 0.9 * fps + 0.1 * instant_fps  # smooth it so it doesn't jitter
            prev_time = now

            frame = draw_hud(frame, person_count=len(detections), fps=fps)

            # ---- Save to file, if requested ---------------------------------
            if disp_cfg["save_path"]:
                if writer is None:
                    # Create the writer lazily, once we know the frame size.
                    h, w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(disp_cfg["save_path"], fourcc, 20.0, (w, h))
                writer.write(frame)

            # ---- Show the live window ---------------------------------------
            if disp_cfg["show_window"]:
                cv2.imshow("Person ID - stages 1-3", frame)
                # waitKey(1) = wait 1ms for a keypress; needed for the window to
                # actually refresh. If the user presses 'q', we break out.
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[main] 'q' pressed -> stopping.")
                    break

    # ---- Cleanup ------------------------------------------------------------
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print("[main] Done.")


if __name__ == "__main__":
    # This guard means main() only runs when you execute `python main.py`,
    # not when something merely imports this file.
    main()
