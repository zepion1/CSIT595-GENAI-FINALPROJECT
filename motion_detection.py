"""
Smart Camera System — Motion Detection + Web UI
------------------------------------------------
Dependencies:
    pip install picamera2 opencv-python-headless requests numpy flask

Access the dashboard at:
    http://<raspberry-pi-ip>:8080
"""

import cv2
import time
import json
import threading
import requests
import logging
import numpy as np
from io import BytesIO
from datetime import datetime
from flask import Flask, Response, render_template, stream_with_context
from picamera2 import Picamera2

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WEBHOOK_URL        = "http://localhost:3000/motion"
CAMERA_RESOLUTION  = (1280, 720)
FRAME_RATE         = 10           # fps fed to motion analyser
STREAM_QUALITY     = 70           # JPEG quality for browser stream
MOTION_THRESHOLD   = 5000         # min changed-pixel area to trigger
BLUR_KERNEL        = (21, 21)
MIN_CONTOUR_AREA   = 1500
COOLDOWN_SECONDS   = 5
WEBHOOK_QUALITY    = 85           # JPEG quality for webhook image
FLASK_HOST         = "0.0.0.0"
FLASK_PORT         = 8080
LOG_LEVEL          = logging.INFO

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
# Shared state (thread-safe via lock)
# ---------------------------------------------------------------------------

state_lock = threading.Lock()
shared_state = {
    "latest_frame":    None,   # raw RGB numpy array from camera
    "annotated_frame": None,   # frame with bounding boxes drawn
    "motion_active":   False,  # is motion happening right now?
    "motion_count":    0,      # total events since start
    "last_event_ts":   None,   # ISO timestamp of last detection
    "cooldown_until":  0.0,    # epoch time when cooldown expires
    "detection_on":    True,   # user toggle
    "webhook_status":  "idle", # idle | ok | error
    "uptime_start":    time.time(),
}

# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

def init_camera() -> Picamera2:
    log.info("Initialising camera...")
    cam = Picamera2()
    cam.configure(
        cam.create_video_configuration(
            main={"size": CAMERA_RESOLUTION, "format": "RGB888"}
        )
    )
    cam.start()
    time.sleep(2)
    log.info("Camera ready at %dx%d", *CAMERA_RESOLUTION)
    return cam

# ---------------------------------------------------------------------------
# Motion detection helpers
# ---------------------------------------------------------------------------

def preprocess(frame: np.ndarray) -> np.ndarray:
    grey = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return cv2.GaussianBlur(grey, BLUR_KERNEL, 0)


def detect_motion(prev: np.ndarray, curr: np.ndarray):
    delta     = cv2.absdiff(prev, curr)
    _, thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)
    thresh    = cv2.dilate(thresh, None, iterations=2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    significant  = [c for c in contours if cv2.contourArea(c) >= MIN_CONTOUR_AREA]
    total_area   = sum(cv2.contourArea(c) for c in significant)
    if total_area < MOTION_THRESHOLD or not significant:
        return False, []
    return True, significant

# ---------------------------------------------------------------------------
# Webhook
# ---------------------------------------------------------------------------

def send_webhook(frame: np.ndarray, contours) -> bool:
    annotated = frame.copy()
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 80), 2)

    ok, buf = cv2.imencode(
        ".jpg",
        cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, WEBHOOK_QUALITY],
    )
    if not ok:
        return False

    timestamp = datetime.utcnow().isoformat() + "Z"
    try:
        resp = requests.post(
            WEBHOOK_URL,
            files={"image": ("motion.jpg", BytesIO(buf.tobytes()), "image/jpeg")},
            data={"timestamp": timestamp, "motion_regions": len(contours)},
            timeout=5,
        )
        resp.raise_for_status()
        log.info("Webhook OK [%d]", resp.status_code)
        return True
    except Exception as exc:
        log.error("Webhook failed: %s", exc)
        return False

# ---------------------------------------------------------------------------
# Background camera + detection thread
# ---------------------------------------------------------------------------

def camera_loop():
    cam = init_camera()
    prev_processed = None
    frame_interval = 1.0 / FRAME_RATE

    while True:
        frame = cam.capture_array()   # RGB

        with state_lock:
            detect = shared_state["detection_on"]

        if not detect:
            with state_lock:
                shared_state["latest_frame"]    = frame
                shared_state["annotated_frame"] = frame
                shared_state["motion_active"]   = False
            time.sleep(frame_interval)
            continue

        curr_processed = preprocess(frame)

        if prev_processed is None:
            prev_processed = curr_processed
            with state_lock:
                shared_state["latest_frame"]    = frame
                shared_state["annotated_frame"] = frame
            time.sleep(frame_interval)
            continue

        motion, contours = detect_motion(prev_processed, curr_processed)

        # Build annotated copy for the stream
        annotated = frame.copy()
        if motion:
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 80), 2)
            cv2.rectangle(annotated, (0, 0), (CAMERA_RESOLUTION[0], 36), (0, 200, 60), -1)
            cv2.putText(
                annotated, "MOTION DETECTED",
                (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 2,
            )

        now = time.time()
        with state_lock:
            shared_state["latest_frame"]    = frame
            shared_state["annotated_frame"] = annotated
            shared_state["motion_active"]   = motion

            if motion and now >= shared_state["cooldown_until"]:
                shared_state["motion_count"]  += 1
                shared_state["last_event_ts"]  = datetime.utcnow().isoformat() + "Z"
                shared_state["cooldown_until"] = now + COOLDOWN_SECONDS
                threading.Thread(
                    target=_webhook_thread, args=(frame, contours), daemon=True
                ).start()

        prev_processed = curr_processed
        time.sleep(frame_interval)


def _webhook_thread(frame, contours):
    ok = send_webhook(frame, contours)
    with state_lock:
        shared_state["webhook_status"] = "ok" if ok else "error"
    time.sleep(3)
    with state_lock:
        shared_state["webhook_status"] = "idle"

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)


def _encode_frame(frame: np.ndarray) -> bytes:
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, STREAM_QUALITY])
    return buf.tobytes()


def _mjpeg_generator():
    while True:
        with state_lock:
            frame = shared_state["annotated_frame"]
        if frame is None:
            time.sleep(0.05)
            continue
        jpg = _encode_frame(frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
        )
        time.sleep(1.0 / FRAME_RATE)


@app.route("/")
def index():
    return render_template("index.html", cooldown_seconds=COOLDOWN_SECONDS)


@app.route("/stream")
def stream():
    return Response(
        stream_with_context(_mjpeg_generator()),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/stats")
def stats_sse():
    """Server-Sent Events — pushes JSON state every second."""
    def generate():
        while True:
            with state_lock:
                s = shared_state
                payload = {
                    "motionActive":  s["motion_active"],
                    "motionCount":   s["motion_count"],
                    "lastEventTs":   s["last_event_ts"],
                    "cooldownLeft":  max(0.0, round(s["cooldown_until"] - time.time(), 1)),
                    "detectionOn":   s["detection_on"],
                    "webhookStatus": s["webhook_status"],
                    "uptimeSeconds": round(time.time() - s["uptime_start"]),
                }
            yield f"data: {json.dumps(payload)}\n\n"
            time.sleep(1)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/toggle", methods=["POST"])
def toggle():
    with state_lock:
        shared_state["detection_on"] = not shared_state["detection_on"]
        status = shared_state["detection_on"]
    return {"detectionOn": status}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    threading.Thread(target=camera_loop, daemon=True).start()
    log.info("Dashboard → http://<pi-ip>:%d", FLASK_PORT)
    app.run(host=FLASK_HOST, port=FLASK_PORT, threaded=True)