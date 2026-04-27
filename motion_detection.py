"""
Smart Camera System with Motion Detection for Raspberry Pi
---------------------------------------------------------
Dependencies:
    pip install picamera2 opencv-python-headless requests numpy

Usage:
    python smart_camera.py
"""

import cv2
import time
import requests
import logging
import numpy as np
from io import BytesIO
from datetime import datetime
from picamera2 import Picamera2

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WEBHOOK_URL = "http://localhost:3000/motion"   # Your server endpoint
CAMERA_RESOLUTION = (1280, 720)                # Width x Height
FRAME_RATE = 10                                # Frames per second to analyse
MOTION_THRESHOLD = 5000                        # Min changed pixels to trigger
BLUR_KERNEL = (21, 21)                         # Gaussian blur kernel size
MIN_CONTOUR_AREA = 1500                        # Ignore tiny contours (noise)
COOLDOWN_SECONDS = 5                           # Seconds between webhook sends
JPEG_QUALITY = 85                              # Image quality sent to server (1-100)
LOG_LEVEL = logging.INFO

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Camera initialisation
# ---------------------------------------------------------------------------

def init_camera() -> Picamera2:
    """Initialise and start the Raspberry Pi camera."""
    log.info("Initialising camera...")
    cam = Picamera2()

    config = cam.create_video_configuration(
        main={"size": CAMERA_RESOLUTION, "format": "RGB888"},
    )
    cam.configure(config)
    cam.start()

    # Allow the sensor to warm up
    time.sleep(2)
    log.info("Camera ready at %dx%d", *CAMERA_RESOLUTION)
    return cam


# ---------------------------------------------------------------------------
# Motion detection
# ---------------------------------------------------------------------------

def preprocess(frame: np.ndarray) -> np.ndarray:
    """Convert a frame to a blurred greyscale image for diffing."""
    grey = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return cv2.GaussianBlur(grey, BLUR_KERNEL, 0)


def detect_motion(prev_frame: np.ndarray, curr_frame: np.ndarray):
    """
    Compare two preprocessed frames.

    Returns:
        (motion_detected: bool, annotated_frame: np.ndarray | None)
        annotated_frame has bounding boxes drawn around motion regions.
    """
    delta = cv2.absdiff(prev_frame, curr_frame)
    _, thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)

    # Dilate to fill small holes in the foreground mask
    thresh = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    significant = [c for c in contours if cv2.contourArea(c) >= MIN_CONTOUR_AREA]
    total_area = sum(cv2.contourArea(c) for c in significant)

    if total_area < MOTION_THRESHOLD or not significant:
        return False, None

    return True, significant


# ---------------------------------------------------------------------------
# Webhook
# ---------------------------------------------------------------------------

def send_webhook(frame: np.ndarray, contours) -> bool:
    """
    Encode the frame as JPEG and POST it to the configured webhook URL.

    The server receives:
        - files["image"]  : JPEG-encoded snapshot
        - data["timestamp"]: ISO-8601 timestamp
        - data["motion_regions"]: number of detected motion contours
    """
    # Draw bounding boxes on a copy of the frame
    annotated = frame.copy()
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Encode to JPEG in memory
    ok, buf = cv2.imencode(
        ".jpg", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
    )
    if not ok:
        log.error("Failed to encode frame as JPEG")
        return False

    timestamp = datetime.utcnow().isoformat() + "Z"

    try:
        response = requests.post(
            WEBHOOK_URL,
            files={"image": ("motion.jpg", BytesIO(buf.tobytes()), "image/jpeg")},
            data={"timestamp": timestamp, "motion_regions": len(contours)},
            timeout=5,
        )
        response.raise_for_status()
        log.info("Webhook sent ✓  [HTTP %d] — %d region(s) at %s",
                 response.status_code, len(contours), timestamp)
        return True

    except requests.exceptions.ConnectionError:
        log.error("Webhook failed — cannot reach %s", WEBHOOK_URL)
    except requests.exceptions.Timeout:
        log.error("Webhook timed out after 5 s")
    except requests.exceptions.HTTPError as exc:
        log.error("Server returned an error: %s", exc)

    return False


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run():
    cam = init_camera()
    last_sent = 0.0
    prev_processed = None

    log.info("Motion detection started (threshold=%d px²)", MOTION_THRESHOLD)

    try:
        while True:
            frame = cam.capture_array()            # RGB numpy array
            curr_processed = preprocess(frame)

            if prev_processed is None:
                prev_processed = curr_processed
                continue

            motion, contours = detect_motion(prev_processed, curr_processed)

            if motion:
                now = time.time()
                if now - last_sent >= COOLDOWN_SECONDS:
                    log.info("Motion detected — sending webhook...")
                    if send_webhook(frame, contours):
                        last_sent = now
                else:
                    remaining = COOLDOWN_SECONDS - (now - last_sent)
                    log.debug("Motion detected but cooling down (%.1f s left)", remaining)

            prev_processed = curr_processed
            time.sleep(1.0 / FRAME_RATE)

    except KeyboardInterrupt:
        log.info("Shutting down — Ctrl+C received.")
    finally:
        cam.stop()
        log.info("Camera stopped.")


if __name__ == "__main__":
    run()