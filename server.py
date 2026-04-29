"""
FastAPI middleware server — receives an image from the Raspberry Pi,
queries an Ollama vision model, and forwards the result to the frontend.

Image-only version: video handling has been removed.

Dependencies:
    pip install fastapi uvicorn httpx python-multipart

Run:
    uvicorn server:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
from fastapi.responses import JSONResponse
import httpx
import base64
import logging
import traceback

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger("server")

app = FastAPI()

# ---------------------------------------------------------------------------
# Configuration — update these to match your setup
# ---------------------------------------------------------------------------

OLLAMA_URL   = "http://localhost:11434/api/generate"
FRONTEND_URL = ""          # Set to your frontend URL, or leave "" to skip
MODEL        = "llava:7b"  # Must exactly match `ollama list` output
API_KEY      = "change-me-to-something-secret"

PROMPT = (
    "You are a security camera AI. Describe what is happening in this image "
    "in 1-2 sentences. Focus on: people, animals, vehicles, or unusual activity."
)


# ---------------------------------------------------------------------------
# Ollama query
# ---------------------------------------------------------------------------

async def query_ollama(b64_image: str) -> str:
    """
    Send a base64-encoded image to Ollama and return the text description.
    Raises HTTPException on any failure so the caller gets a meaningful status code.
    """
    log.debug("Sending image to Ollama model '%s'", MODEL)

    try:
        async with httpx.AsyncClient(timeout=180) as client:
            r = await client.post(OLLAMA_URL, json={
                "model":  MODEL,
                "prompt": PROMPT,
                "images": [b64_image],
                "stream": False,
            })
            r.raise_for_status()
            description = r.json()["response"].strip()
            log.info("Ollama description: %s", description)
            return description

    except httpx.ConnectError:
        log.error("Cannot connect to Ollama at %s", OLLAMA_URL)
        raise HTTPException(status_code=503, detail="Ollama is not running — start it with `ollama serve`")

    except httpx.TimeoutException:
        log.error("Ollama timed out")
        raise HTTPException(status_code=504, detail="Ollama timed out — try a lighter model like moondream")

    except httpx.HTTPStatusError as exc:
        body = exc.response.text[:400]
        log.error("Ollama HTTP error %d: %s", exc.response.status_code, body)
        raise HTTPException(
            status_code=502,
            detail=f"Ollama returned {exc.response.status_code}: {body}"
        )

    except KeyError:
        log.error("Unexpected Ollama response shape")
        raise HTTPException(status_code=502, detail="Ollama response missing 'response' field")

    except Exception:
        log.error("Ollama query failed:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail="Unexpected error querying Ollama")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    """Check FastAPI is up and Ollama is reachable, and list available models."""
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
):
    # --- Auth ---
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # --- Read image ---
    try:
        image_bytes = await image.read()
        image_b64   = base64.b64encode(image_bytes).decode()
        log.debug("Image received: %d bytes", len(image_bytes))
    except Exception:
        log.error("Failed to read image:\n%s", traceback.format_exc())
        raise HTTPException(status_code=400, detail="Could not read image field")

    # --- Query Ollama ---
    description = await query_ollama(image_b64)

    # --- Forward to frontend (optional) ---
    if FRONTEND_URL:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                await client.post(FRONTEND_URL, json={
                    "timestamp":      timestamp,
                    "motion_regions": motion_regions,
                    "description":    description,
                    "image_b64":      image_b64,
                })
            log.info("Forwarded result to frontend")
        except Exception as e:
            # Don't fail the whole request just because the frontend is unreachable
            log.warning("Could not reach frontend (%s): %s", FRONTEND_URL, e)

    return JSONResponse({
        "status":      "ok",
        "description": description,
    })