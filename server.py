"""
FastAPI middleware server — receives image from the Raspberry Pi,
queries an Ollama vision model, stores recent alerts in memory,
and serves them to a frontend dashboard.
"""

from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware

import httpx
import base64
import logging
import traceback
import tempfile
import os
import uuid
from datetime import datetime
from collections import deque
from pathlib import Path

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger("server")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OLLAMA_URL   = "http://localhost:11434/api/generate"
MODEL        = "llava:7b"
API_KEY      = "change-me-to-something-secret"

VIDEO_KEY_FRAMES   = 4
SNAPSHOTS_DIR      = Path("/content/snapshots")
MAX_ALERT_HISTORY  = 100

SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

PROMPT_IMAGE = (
    "You are a security camera AI. Describe what is happening in this image "
    "in 1-2 sentences. Focus on: people, animals, vehicles, or unusual activity."
)

PROMPT_VIDEO = (
    "You are a security camera AI. I'm showing you {n} frames sampled evenly "
    "from a short motion-triggered video clip. Describe what is happening "
    "across these frames in 2-3 sentences. Focus on movement, people, animals, "
    "vehicles, or any unusual activity. Note any changes between frames."
)

# In-memory alert store, newest appended last
alert_history = deque(maxlen=MAX_ALERT_HISTORY)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Video frame extraction (kept for future use; not called by the image-only Pi)
# ---------------------------------------------------------------------------

def extract_key_frames(video_bytes, n_frames=VIDEO_KEY_FRAMES):
    import cv2

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            new_path = tmp_path.replace(".mp4", ".avi")
            os.rename(tmp_path, new_path)
            tmp_path = new_path
            cap = cv2.VideoCapture(tmp_path)
            if not cap.isOpened():
                return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            return []

        n_frames = min(n_frames, total_frames)
        sample_idxs = [int(total_frames * (i + 0.5) / n_frames) for i in range(n_frames)]

        b64_frames = []
        for idx in sample_idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            b64_frames.append(base64.b64encode(buf.tobytes()).decode())

        cap.release()
        return b64_frames

    except Exception:
        log.error("Frame extraction failed: %s", traceback.format_exc())
        return []
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------

async def query_ollama(b64_images, prompt):
    log.debug("Sending %d image(s) to Ollama '%s'", len(b64_images), MODEL)
    try:
        async with httpx.AsyncClient(timeout=180) as client:
            r = await client.post(OLLAMA_URL, json={
                "model":  MODEL,
                "prompt": prompt,
                "images": b64_images,
                "stream": False,
            })
            r.raise_for_status()
            description = r.json()["response"].strip()
            log.info("Ollama description: %s", description)
            return description

    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Ollama is not running")
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Ollama timed out")
    except httpx.HTTPStatusError as exc:
        body = exc.response.text[:400]
        raise HTTPException(status_code=502, detail=f"Ollama returned {exc.response.status_code}: {body}")
    except Exception as e:
        log.error("Ollama query failed: %s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get("http://localhost:11434/api/tags")
            models = [m["name"] for m in r.json().get("models", [])]
        return {
            "fastapi":     "ok",
            "ollama":      "ok",
            "models":      models,
            "alert_count": len(alert_history),
        }
    except Exception as e:
        return {"fastapi": "ok", "ollama": "error", "detail": str(e)}


@app.get("/alerts")
async def get_alerts(limit: int = 50):
    """Return recent alerts, newest first."""
    items = list(alert_history)[-limit:]
    items.reverse()
    return {"alerts": items}


@app.delete("/alerts")
async def clear_alerts():
    """Clear all stored alerts (snapshots on disk are kept)."""
    n = len(alert_history)
    alert_history.clear()
    return {"cleared": n}


@app.post("/analyze")
async def analyze(
    image:          UploadFile = File(...),
    timestamp:      str        = Form(...),
    motion_regions: int        = Form(...),
    x_api_key:      str        = Header(...),
    video:          UploadFile = File(None),
):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # --- Read snapshot ---
    try:
        image_bytes = await image.read()
        image_b64   = base64.b64encode(image_bytes).decode()
        log.debug("Snapshot received: %d bytes", len(image_bytes))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image field")

    # --- Save snapshot to disk so frontend can render it ---
    alert_id  = uuid.uuid4().hex[:12]
    base_name = datetime.utcnow().strftime("%Y%m%d_%H%M%S") + f"_{alert_id}"
    snap_path = SNAPSHOTS_DIR / f"{base_name}.jpg"
    with open(snap_path, "wb") as f:
        f.write(image_bytes)
    image_url = f"/snapshots/{snap_path.name}"

    # --- Decide image vs video pipeline ---
    use_video  = False
    b64_images = [image_b64]
    prompt     = PROMPT_IMAGE

    if video is not None:
        try:
            video_bytes = await video.read()
            if len(video_bytes) > 0:
                key_frames = extract_key_frames(video_bytes, VIDEO_KEY_FRAMES)
                if key_frames:
                    use_video  = True
                    b64_images = key_frames
                    prompt     = PROMPT_VIDEO.format(n=len(key_frames))
        except Exception:
            log.error("Video processing failed: %s", traceback.format_exc())

    # --- Query Ollama ---
    description = await query_ollama(b64_images, prompt)

    # --- Build alert record + store + return ---
    alert = {
        "id":             alert_id,
        "timestamp":      timestamp,
        "description":    description,
        "motion_regions": motion_regions,
        "image_url":      image_url,
        "used_video":     use_video,
        "frames_used":    len(b64_images),
        "model":          MODEL,
    }

    alert_history.append(alert)
    log.info("Alert stored [%s] %d regions — %s", alert_id, motion_regions, description[:80])

    return JSONResponse({
        "status":      "ok",
        "alert":       alert,
        "description": description,
        "used_video":  use_video,
        "frames_used": len(b64_images),
    })


@app.get("/snapshots/{filename}")
async def serve_snapshot(filename: str):
    img_path = SNAPSHOTS_DIR / filename
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return Response(content=img_path.read_bytes(), media_type="image/jpeg")
