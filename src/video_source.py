"""
video_source.py  --  STAGE 1 -> 2 of the pipeline:  RTSP / file  ->  frames.

============================ THE BIG PICTURE ================================
A camera does not hand you "images". It broadcasts a continuous, compressed
video *stream* over the network using a protocol called RTSP (Real-Time
Streaming Protocol). Think of RTSP like a radio station for video: the camera
is always transmitting, and anyone who "tunes in" to its URL gets the feed.

But a neural network can't look at a "stream" -- it looks at ONE still image
at a time. A single still image pulled out of a video is called a FRAME.
Video is just many frames shown quickly (e.g. 25 frames per second = 25 FPS).

So this file's ONE job is:  connect to the stream, and keep pulling out
frames one by one, forever, handing each frame to whoever asks for it.

We use OpenCV (the `cv2` library) for this. OpenCV knows how to:
  1. Speak RTSP and connect to the camera.
  2. DECODE the compressed video (undo the compression) into raw pixels.
Each frame OpenCV gives us is just a grid of pixels: a NumPy array with shape
(height, width, 3) -- the 3 being the Blue, Green, Red colour channels.
(Yes, OpenCV uses B-G-R order, not R-G-B. A famous historical quirk.)
============================================================================
"""

import time
import cv2  # OpenCV: the library that does all the video heavy-lifting.


class VideoSource:
    """
    Wraps an OpenCV video capture so the rest of the program never has to
    think about RTSP, reconnecting, or decoding. It just asks for frames.

    Usage:
        with VideoSource(cfg) as cam:
            for frame in cam.frames():
                ...do something with `frame`...
    """

    def __init__(self, url, max_reconnect_attempts=5, reconnect_delay_seconds=2):
        # `url` is whatever was in config.yaml under source.url.
        # It might be an rtsp:// URL, a file path, or "0" for a webcam.
        self.url = self._normalize_source(url)
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_delay_seconds = reconnect_delay_seconds

        # `capture` is OpenCV's object that actually holds the open connection.
        # None means "not connected yet".
        self.capture = None

        # Is this a LIVE stream (RTSP/webcam) or a FIXED file that has an end?
        # We treat a read-failure differently for each:
        #   - live stream -> a blip; try to reconnect.
        #   - file        -> the video simply ended; stop cleanly.
        self.is_stream = self._looks_like_stream(self.url)

    @staticmethod
    def _looks_like_stream(url):
        """True for live sources (webcam index or rtsp:// URL), False for files."""
        if isinstance(url, int):
            return True                      # a webcam index like 0 is live
        return str(url).lower().startswith(("rtsp://", "http://", "https://"))

    @staticmethod
    def _normalize_source(url):
        """
        OpenCV accepts a webcam by an integer index (0 = first camera), but a
        file/RTSP by a string. If the user wrote "0" in the config, turn it
        into the integer 0. Otherwise leave it as the string it is.
        """
        if isinstance(url, str) and url.strip().isdigit():
            return int(url.strip())
        return url

    def open(self):
        """
        Actually establish the connection to the camera/file.

        cv2.VideoCapture(...) is the line that "tunes in" to the stream. It
        returns immediately even if the connection is still warming up, so we
        then check .isOpened() to confirm it really connected.
        """
        # cv2.CAP_FFMPEG tells OpenCV to use the FFMPEG backend, which is the
        # one that handles RTSP and most video files reliably.
        self.capture = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)

        if not self.capture.isOpened():
            raise ConnectionError(
                f"Could not open video source: {self.url!r}. "
                f"If it's an RTSP camera, check the URL, username/password, "
                f"and that you're on the same network."
            )

        # A tiny buffer means we favour the *latest* frame over old buffered
        # ones -- important for live cameras so we don't lag behind reality.
        # (Not all backends honour this; it's a best-effort hint.)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return self

    def _reconnect(self):
        """
        RTSP over a network is not perfectly reliable -- packets drop, WiFi
        blips. If reading a frame fails, we try to close and re-open the
        stream a few times before giving up. This is what makes the pipeline
        survive a real-world flaky camera.
        """
        for attempt in range(1, self.max_reconnect_attempts + 1):
            print(f"[video_source] Reconnecting... attempt "
                  f"{attempt}/{self.max_reconnect_attempts}")
            if self.capture is not None:
                self.capture.release()  # close the old, broken connection
            time.sleep(self.reconnect_delay_seconds)
            try:
                self.open()
                print("[video_source] Reconnected.")
                return True
            except ConnectionError:
                continue  # try again
        return False  # all attempts exhausted

    def frames(self):
        """
        A GENERATOR that yields frames one at a time.

        A "generator" is a function that produces a stream of values lazily:
        each time the caller's for-loop asks for the next item, execution
        resumes here, grabs ONE frame, and hands it back. This means we never
        load the whole video into memory -- we process it frame-by-frame.

        Each yielded `frame` is a NumPy array of shape (height, width, 3).
        """
        if self.capture is None:
            self.open()

        while True:
            # .read() does two things at once:
            #   ok    -> a boolean: did we successfully get a frame?
            #   frame -> the actual image (or None if ok is False).
            ok, frame = self.capture.read()

            if not ok:
                if not self.is_stream:
                    # A FILE returned no frame -> we reached the end. Done.
                    print("[video_source] End of video file.")
                    break
                # A live stream returned no frame -> probably a network blip.
                # Try to reconnect; if that also fails, stop cleanly.
                print("[video_source] Frame read failed (live stream).")
                if self._reconnect():
                    continue          # reconnected -> keep going
                else:
                    break             # give up -> end the generator

            # Success: hand this single frame to the caller.
            yield frame

    def release(self):
        """Politely close the connection and free the camera/file handle."""
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    # ---- These two make `with VideoSource(...) as cam:` work. ----
    # A "context manager" guarantees release() is called even if the code
    # inside the `with` block crashes -- so we never leave a camera handle open.
    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()
