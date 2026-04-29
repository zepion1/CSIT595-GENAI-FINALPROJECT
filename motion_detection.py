"""
Smart Camera System with Motion Detection for Raspberry Pi
---------------------------------------------------------
Image-only version: sends a single annotated JPEG snapshot to the
Ollama vision server when sustained motion is detected.

Dependencies:
    pip install picamera2 opencv-python requests numpy

Usage:
    python motion_detection.py

Preview window controls:
    q  — quit
    p  — pause / resume detection
    s  — save current frame as JPEG snapshot
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

# Your Ollama FastAPI server endpoint.
# - For Colab + ngrok:  "https://xxxx-xx-xx-xx-xx.ngrok-free.app/analyze"
# - For HPC direct:     "http://<HPC_IP>:8000/analyze"
# - For SSH tunnel:     "http://localhost:8000/analyze"
WEBHOOK_URL = "https://september-festivity-legislate.ngrok-free.dev/analyze"

# Must match the API_KEY set in your FastAPI server.py
API_KEY = "change-me-to-something-secret"

CAMERA_RESOLUTION = (1280, 720)   # Width x Height
FRAME_RATE        = 10            # Frames per second to analyse
MOTION_THRESHOLD  = 16000         # Min changed pixels to trigger
BLUR_KERNEL       = (21, 21)      # Gaussian blur kernel size
MIN_CONTOUR_AREA  = 6000          # Ignore tiny contours (noise)
COOLDOWN_SECONDS  = 32            # Seconds between webhook sends
JPEG_QUALITY      = 89            # Image quality sent to server (1-100)
SUSTAINED_FRAMES  = 16            # Consecutive frames with motion required
                                  # before a webhook fires.
WEBHOOK_TIMEOUT   = 30            # Seconds — Colab inference is slower than HPC

LOG_LEVEL = logging.INFO

WINDOW_NAME   = "Smart Camera — Motion Detection"
PREVIEW_SCALE = 1.0   # Scale the preview window (e.g. 0.5 for half size)

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
        (motion_detected: bool, contours: list)
    """
    delta  = cv2.absdiff(prev_frame, curr_frame)
    _, thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    significant = [c for c in contours if cv2.contourArea(c) >= MIN_CONTOUR_AREA]
    total_area  = sum(cv2.contourArea(c) for c in significant)

    if total_area < MOTION_THRESHOLD or not significant:
        return False, []

    return True, significant


# ---------------------------------------------------------------------------
# Canvas / preview
# ---------------------------------------------------------------------------

def build_preview(
    frame: np.ndarray,
    contours: list,
    motion: bool,
    paused: bool,
    cooldown_remaining: float,
    sustained_count: int,
) -> np.ndarray:
    """
    Draw the live preview canvas with bounding boxes and a status bar.
    """
    canvas = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # --- Motion bounding boxes --------------------------------------------
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), 2)
        area = int(cv2.contourArea(contour))
        cv2.putText(
            canvas, f"{area} px²",
            (x + 4, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA,
        )

    # --- Status bar (top strip) -------------------------------------------
    bar_h = 32
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], bar_h), (20, 20, 20), -1)

    timestamp = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    cv2.putText(canvas, timestamp, (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    if paused:
        status_text   = "PAUSED"
        status_colour = (0, 200, 255)
    elif motion:
        status_text   = f"MOTION  [{sustained_count}/{SUSTAINED_FRAMES}]"
        status_colour = (0, 60, 255)
    else:
        status_text   = "Monitoring..."
        status_colour = (0, 200, 80)

    (tw, _), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cx = (canvas.shape[1] - tw) // 2
    cv2.putText(canvas, status_text, (cx, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_colour, 2, cv2.LINE_AA)

    # Cooldown badge (top-right)
    if cooldown_remaining > 0:
        badge = f"Next send in {cooldown_remaining:.1f}s"
        (bw, _), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(canvas, badge,
                    (canvas.shape[1] - bw - 8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 180, 255), 1, cv2.LINE_AA)

    # --- Legend (bottom-right) --------------------------------------------
    legend_lines = ["q — quit", "p — pause", "s — snapshot"]
    for i, line in enumerate(reversed(legend_lines)):
        cv2.putText(
            canvas, line,
            (canvas.shape[1] - 130, canvas.shape[0] - 10 - i * 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1, cv2.LINE_AA,
        )

    # --- Region count (bottom-left) ---------------------------------------
    cv2.putText(canvas, f"Regions: {len(contours)}",
                (8, canvas.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    # --- Optional downscale -----------------------------------------------
    if PREVIEW_SCALE != 1.0:
        new_w = int(canvas.shape[1] * PREVIEW_SCALE)
        new_h = int(canvas.shape[0] * PREVIEW_SCALE)
        canvas = cv2.resize(canvas, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return canvas


# ---------------------------------------------------------------------------
# Webhook
# ---------------------------------------------------------------------------

def send_webhook(frame: np.ndarray, contours: list) -> bool:
    """
    Encode the frame as an annotated JPEG and POST it to the configured
    webhook URL with the API key header.

    Returns True on success, False on any error.
    """
    # --- Build annotated snapshot JPEG -----------------------------------
    annotated = frame.copy()
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

    ok, buf = cv2.imencode(
        ".jpg", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
    )
    if not ok:
        log.error("Failed to encode frame as JPEG")
        return False

    timestamp = datetime.utcnow().isoformat() + "Z"

    # --- POST to server --------------------------------------------------
    try:
        response = requests.post(
            WEBHOOK_URL,
            files={"image": ("motion.jpg", BytesIO(buf.tobytes()), "image/jpeg")},
            data={
                "timestamp":      timestamp,
                "motion_regions": len(contours),
            },
            headers={"X-API-Key": API_KEY},
            timeout=WEBHOOK_TIMEOUT,
        )
        response.raise_for_status()
        log.info(
            "Webhook sent ✓  [HTTP %d] — %d region(s) at %s",
            response.status_code, len(contours), timestamp,
        )
        # Log description returned by the vision model if present
        try:
            description = response.json().get("description", "")
            if description:
                log.info("AI description: %s", description)
        except Exception:
            pass

        return True

    except requests.exceptions.ConnectionError:
        log.error("Webhook failed — cannot reach %s", WEBHOOK_URL)
    except requests.exceptions.Timeout:
        log.error("Webhook timed out after %ds — consider increasing WEBHOOK_TIMEOUT", WEBHOOK_TIMEOUT)
    except requests.exceptions.HTTPError as exc:
        log.error("Server returned an error: %s", exc)

    return False


# ---------------------------------------------------------------------------
# Snapshot helper
# ---------------------------------------------------------------------------

def save_snapshot(frame: np.ndarray) -> None:
    """Save the current frame as a timestamped JPEG to the working directory."""
    fname = datetime.now().strftime("snapshot_%Y%m%d_%H%M%S.jpg")
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if ok:
        with open(fname, "wb") as f:
            f.write(buf.tobytes())
        log.info("Snapshot saved: %s", fname)
    else:
        log.error("Failed to save snapshot")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run():
    cam             = init_camera()
    last_sent       = 0.0
    prev_processed  = None
    paused          = False
    sustained_count = 0   # Consecutive frames with confirmed motion

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    preview_w = int(CAMERA_RESOLUTION[0] * PREVIEW_SCALE)
    preview_h = int(CAMERA_RESOLUTION[1] * PREVIEW_SCALE)
    cv2.resizeWindow(WINDOW_NAME, preview_w, preview_h)

    log.info(
        "Motion detection started  (threshold=%d px², sustained=%d frames)",
        MOTION_THRESHOLD, SUSTAINED_FRAMES,
    )

    try:
        while True:
            frame          = cam.capture_array()
            curr_processed = preprocess(frame)

            if prev_processed is None:
                prev_processed = curr_processed
                continue

            motion, contours = detect_motion(prev_processed, curr_processed)

            now                = time.time()
            cooldown_remaining = max(0.0, COOLDOWN_SECONDS - (now - last_sent))

            # --- Sustained motion detection --------------------------------
            if not paused:
                if motion:
                    sustained_count += 1
                    log.debug("Sustained motion: %d/%d", sustained_count, SUSTAINED_FRAMES)
                else:
                    sustained_count = 0

                # Fire webhook once motion is sustained AND cooldown elapsed
                if sustained_count >= SUSTAINED_FRAMES and cooldown_remaining == 0:
                    log.info("Sustained motion confirmed — sending webhook...")
                    if send_webhook(frame, contours):
                        last_sent = now
                    sustained_count = 0

            # --- Preview ---------------------------------------------------
            canvas = build_preview(
                frame, contours,
                motion=(motion and not paused),
                paused=paused,
                cooldown_remaining=cooldown_remaining,
                sustained_count=sustained_count,
            )
            cv2.imshow(WINDOW_NAME, canvas)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                log.info("Quit key pressed.")
                break
            elif key == ord("p"):
                paused = not paused
                sustained_count = 0
                log.info("Detection %s.", "paused" if paused else "resumed")
            elif key == ord("s"):
                save_snapshot(frame)

            if not paused:
                prev_processed = curr_processed

            time.sleep(1.0 / FRAME_RATE)

    except KeyboardInterrupt:
        log.info("Shutting down — Ctrl+C received.")
    finally:
        cam.stop()
        cv2.destroyAllWindows()
        log.info("Camera stopped.")


if __name__ == "__main__":
    run()