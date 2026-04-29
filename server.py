"""
FastAPI middleware server — receives image/video from the Raspberry Pi,
queries an Ollama vision model, and forwards the result to the frontend.
"""

from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
from fastapi.responses import JSONResponse
import httpx
import base64
import logging
import traceback
import tempfile
import os

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger("server")

app = FastAPI()

OLLAMA_URL   = "http://localhost:11434/api/generate"
FRONTEND_URL = ""
MODEL        = "llava:7b"
API_KEY      = "change-me-to-something-secret"
VIDEO_KEY_FRAMES = 4

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
                log.error("OpenCV could not open the video file")
                return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        log.debug("Video has %d frames, extracting %d", total_frames, n_frames)

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
        log.info("Extracted %d key frames from video", len(b64_frames))
        return b64_frames

    except Exception:
        log.error("Frame extraction failed: %s", traceback.format_exc())
        return []
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


async def query_ollama(b64_images, prompt):
    log.debug("Sending %d image(s) to Ollama model '%s'", len(b64_images), MODEL)
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
        log.error("Ollama HTTP %d: %s", exc.response.status_code, body)
        raise HTTPException(status_code=502, detail=f"Ollama returned {exc.response.status_code}: {body}")
    except Exception as e:
        log.error("Ollama query failed: %s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


@app.get("/health")
async def health():
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get("http://localhost:11434/api/tags")
            models = [m["name"] for m in r.json().get("models", [])]
        return {"fastapi": "ok", "ollama": "ok", "models": models}
    except Exception as e:
        return {"fastapi": "ok", "ollama": "error", "detail": str(e)}


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

    try:
        image_bytes = await image.read()
        image_b64   = base64.b64encode(image_bytes).decode()
        log.debug("Snapshot received: %d bytes", len(image_bytes))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image field")

    use_video = False
    b64_images = [image_b64]
    prompt = PROMPT_IMAGE

    if video is not None:
        try:
            video_bytes = await video.read()
            log.debug("Video received: %d bytes", len(video_bytes))
            if len(video_bytes) > 0:
                key_frames = extract_key_frames(video_bytes, VIDEO_KEY_FRAMES)
                if key_frames:
                    use_video  = True
                    b64_images = key_frames
                    prompt     = PROMPT_VIDEO.format(n=len(key_frames))
                    log.info("Using %d video key frames", len(key_frames))
        except Exception:
            log.error("Video processing failed: %s", traceback.format_exc())
            log.warning("Falling back to snapshot")

    if not use_video:
        log.info("Using snapshot image")

    description = await query_ollama(b64_images, prompt)

    if FRONTEND_URL:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                await client.post(FRONTEND_URL, json={
                    "timestamp":      timestamp,
                    "motion_regions": motion_regions,
                    "description":    description,
                    "image_b64":      image_b64,
                    "used_video":     use_video,
                })
        except Exception as e:
            log.warning("Could not reach frontend: %s", e)

    return JSONResponse({
        "status":      "ok",
        "description": description,
        "used_video":  use_video,
        "frames_used": len(b64_images),
    })