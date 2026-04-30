#!/usr/bin/env python3
"""
Send a local image to the server and verify the frontend can receive it.

Usage:
    python test_server.py <image_file> [server_url]

Examples:
    python test_server.py photo.jpg
    python test_server.py photo.jpg https://xxxx.ngrok-free.app
"""

import sys
import io
import requests
from datetime import datetime, timezone

if len(sys.argv) < 2:
    print("Usage: python test_server.py <image_file> [server_url]")
    sys.exit(1)

IMAGE_FILE = sys.argv[1]
BASE       = sys.argv[2].rstrip("/") if len(sys.argv) > 2 else "http://localhost:8000"
API_KEY    = "change-me-to-something-secret"
HEADERS    = {"ngrok-skip-browser-warning": "true"}

print(f"\nServer : {BASE}")
print(f"Image  : {IMAGE_FILE}\n")

# ── 1. Read the image ─────────────────────────────────────────────────
with open(IMAGE_FILE, "rb") as f:
    image_bytes = f.read()
print(f"[1] Image loaded — {len(image_bytes)} bytes")

# ── 2. POST to /analyze ───────────────────────────────────────────────
timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

print("[2] Sending to /analyze ...")
r = requests.post(
    f"{BASE}/analyze",
    files={"image": (IMAGE_FILE, io.BytesIO(image_bytes), "image/jpeg")},
    data={"timestamp": timestamp, "motion_regions": "1"},
    headers={**HEADERS, "X-API-Key": API_KEY},
    timeout=180,
)

if r.status_code != 200:
    print(f"    ERROR — HTTP {r.status_code}")
    print(f"    {r.text[:400]}")
    sys.exit(1)

body      = r.json()
alert     = body.get("alert", {})
alert_id  = alert.get("id")
image_url = alert.get("image_url")
desc      = alert.get("description", "")

print(f"    OK — alert id   : {alert_id}")
print(f"         image_url  : {image_url}")
print(f"         description: {desc[:120]}")

# ── 3. Confirm the alert appears in /alerts ───────────────────────────
print("[3] Checking /alerts ...")
r2    = requests.get(f"{BASE}/alerts", headers=HEADERS, timeout=10)
items = r2.json().get("alerts", [])
ids   = [a.get("id") for a in items]

if alert_id in ids:
    print(f"    OK — alert {alert_id} is in the feed ({len(items)} total)")
else:
    print(f"    WARN — alert {alert_id} not found in feed. Got ids: {ids}")

# ── 4. Confirm the image is actually served ───────────────────────────
print("[4] Fetching served image ...")
img_url = BASE + image_url + "?ngrok-skip-browser-warning=true"
r3      = requests.get(img_url, timeout=15)

if r3.status_code == 200:
    print(f"    OK — image served ({len(r3.content)} bytes, {r3.headers.get('content-type')})")
else:
    print(f"    ERROR — HTTP {r3.status_code} for {img_url}")

print(f"\nDone. Open the frontend, connect to {BASE}, and the alert should appear.\n")
