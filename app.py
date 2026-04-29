import os
import shutil
import time
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile

from inference import QwenActivityDescriber


ALLOWED_EXTENSIONS = {".mp4", ".mov", ".webm", ".mkv", ".avi"}


app = FastAPI(title="Activity Describer", version="0.1.0")

describer = QwenActivityDescriber()


def _validate_extension(filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type '{ext}'. "
                f"Allowed: {sorted(ALLOWED_EXTENSIONS)}"
            ),
        )
    return ext


@app.post("/describe")
async def upload_and_describe(file: UploadFile = File(...)):
    ext = _validate_extension(file.filename or "")

    temp_path = f"temp_{uuid.uuid4().hex}{ext}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        abs_path = os.path.abspath(temp_path)
        start = time.perf_counter()
        description = describer.describe_activity(
            video_source=f"file://{abs_path}"
        )
        latency_ms = int((time.perf_counter() - start) * 1000)

        return {
            "filename": file.filename,
            "description": description,
            "latency_ms": latency_ms,
        }

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/detect")
async def upload_and_detect(file: UploadFile = File(...)):
    """Backwards-compatible alias for /describe."""
    return await upload_and_describe(file)


@app.get("/health")
def health():
    return {"status": "ok", "device": describer.device}


@app.get("/")
def home():
    return {
        "message": (
            "Activity describer is running. "
            "POST a video to /describe (multipart, field name 'file'). "
            "See /docs for the interactive UI."
        )
    }
