"""
Smart Camera System with Motion Detection for Raspberry Pi
---------------------------------------------------------
Dependencies:
    pip install picamera2 opencv-python requests numpy

Usage:
    python smart_camera.py

Preview window controls:
    q  — quit
    h  — toggle heatmap overlay
    p  — pause / resume detection
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

WINDOW_NAME = "Smart Camera — Motion Detection"
PREVIEW_SCALE = 1.0        # Scale the preview window (e.g. 0.5 for half size)
SHOW_HEATMAP = True        # Start with heatmap overlay enabled
HEATMAP_ALPHA = 0.35       # Heatmap blend strength (0.0–1.0)

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
        (motion_detected: bool, contours, delta_mask)
        delta_mask is the thresholded diff image (used for the heatmap).
    """
    delta = cv2.absdiff(prev_frame, curr_frame)
    _, thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)

    # Dilate to fill small holes in the foreground mask
    thresh = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    significant = [c for c in contours if cv2.contourArea(c) >= MIN_CONTOUR_AREA]
    total_area = sum(cv2.contourArea(c) for c in significant)

    if total_area < MOTION_THRESHOLD or not significant:
        return False, [], thresh

    return True, significant, thresh


# ---------------------------------------------------------------------------
# Canvas / preview
# ---------------------------------------------------------------------------

def build_preview(
    frame: np.ndarray,
    contours: list,
    delta_mask: np.ndarray,
    motion: bool,
    paused: bool,
    show_heatmap: bool,
    cooldown_remaining: float,
) -> np.ndarray:
    """
    Draw the live preview canvas:
      • Bounding boxes around motion regions
      • Colour heatmap overlay of changed pixels (toggleable)
      • Status bar at the top with timestamp, state and FPS
      • Legend at the bottom

    Args:
        frame: raw RGB frame from the camera
        contours: significant motion contours (may be empty)
        delta_mask: thresholded diff image (greyscale, same H×W as frame)
        motion: whether motion was detected this frame
        paused: whether detection is paused
        show_heatmap: whether to blend the heatmap
        cooldown_remaining: seconds left in the send cooldown (0 if ready)

    Returns:
        BGR preview image ready for cv2.imshow
    """
    # Work in BGR so OpenCV drawing functions use expected colours
    canvas = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # --- Heatmap overlay ---------------------------------------------------
    if show_heatmap and delta_mask is not None:
        heatmap = cv2.applyColorMap(delta_mask, cv2.COLORMAP_JET)
        # Only colour pixels that actually changed (non-zero in mask)
        mask_bool = delta_mask > 0
        blended = canvas.copy()
        blended[mask_bool] = cv2.addWeighted(
            canvas, 1 - HEATMAP_ALPHA,
            heatmap, HEATMAP_ALPHA, 0,
        )[mask_bool]
        canvas = blended

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
        status_text = "PAUSED"
        status_colour = (0, 200, 255)          # amber
    elif motion:
        status_text = "MOTION DETECTED"
        status_colour = (0, 60, 255)           # red
    else:
        status_text = "Monitoring…"
        status_colour = (0, 200, 80)           # green

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

    # --- Region count (bottom-left) ---------------------------------------
    region_text = f"Regions: {len(contours)}"
    cv2.putText(canvas, region_text,
                (8, canvas.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    # --- Key legend (bottom-right) ----------------------------------------
    legend_lines = [
        "q — quit",
        f"h — heatmap [{'ON' if show_heatmap else 'OFF'}]",
        "p — pause",
    ]
    for i, line in enumerate(reversed(legend_lines)):
        cv2.putText(
            canvas, line,
            (canvas.shape[1] - 160, canvas.shape[0] - 10 - i * 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1, cv2.LINE_AA,
        )

    # --- Optional downscale -----------------------------------------------
    if PREVIEW_SCALE != 1.0:
        new_w = int(canvas.shape[1] * PREVIEW_SCALE)
        new_h = int(canvas.shape[0] * PREVIEW_SCALE)
        canvas = cv2.resize(canvas, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return canvas


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
    show_heatmap = SHOW_HEATMAP
    paused = False

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    preview_w = int(CAMERA_RESOLUTION[0] * PREVIEW_SCALE)
    preview_h = int(CAMERA_RESOLUTION[1] * PREVIEW_SCALE)
    cv2.resizeWindow(WINDOW_NAME, preview_w, preview_h)

    log.info("Motion detection started (threshold=%d px²)", MOTION_THRESHOLD)
    log.info("Preview window open — press q to quit, h to toggle heatmap, p to pause")

    try:
        while True:
            frame = cam.capture_array()            # RGB numpy array
            curr_processed = preprocess(frame)

            if prev_processed is None:
                prev_processed = curr_processed
                continue

            motion, contours, delta_mask = detect_motion(prev_processed, curr_processed)

            now = time.time()
            cooldown_remaining = max(0.0, COOLDOWN_SECONDS - (now - last_sent))

            if not paused and motion:
                if cooldown_remaining == 0:
                    log.info("Motion detected — sending webhook...")
                    if send_webhook(frame, contours):
                        last_sent = now
                else:
                    log.debug("Motion detected but cooling down (%.1f s left)",
                              cooldown_remaining)

            # Build and show the preview canvas
            canvas = build_preview(
                frame, contours, delta_mask,
                motion=(motion and not paused),
                paused=paused,
                show_heatmap=show_heatmap,
                cooldown_remaining=cooldown_remaining,
            )
            cv2.imshow(WINDOW_NAME, canvas)

            # Handle key presses (1 ms poll so the loop stays responsive)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                log.info("Quit key pressed.")
                break
            elif key == ord("h"):
                show_heatmap = not show_heatmap
                log.info("Heatmap overlay: %s", "ON" if show_heatmap else "OFF")
            elif key == ord("p"):
                paused = not paused
                log.info("Detection %s.", "paused" if paused else "resumed")

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