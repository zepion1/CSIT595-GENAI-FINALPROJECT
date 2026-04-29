"""HTTP client smoke test.

Posts a video to the running FastAPI server and prints the response.
Useful on Windows where curl may not be available.

Prereq:
    pip install requests
    # and have the server running:
    # uvicorn app:app --host 0.0.0.0 --port 8000

Usage:
    python client_test.py                        # uses sample.mp4 + localhost
    python client_test.py my_clip.mp4
    python client_test.py my_clip.mp4 http://192.168.1.50:8000
"""

import json
import os
import sys
import time

import requests


def main(
    video_path: str = "sample.mp4",
    base_url: str = "http://localhost:8000",
) -> int:
    if not os.path.exists(video_path):
        print(f"[error] video not found: {video_path}")
        return 1

    url = f"{base_url.rstrip('/')}/describe"
    print(f"[info] POST {url}")
    print(f"[info] file: {os.path.abspath(video_path)}")

    t0 = time.perf_counter()
    with open(video_path, "rb") as f:
        resp = requests.post(
            url,
            files={"file": (os.path.basename(video_path), f, "video/mp4")},
            timeout=300,
        )
    wall = time.perf_counter() - t0

    print(f"[info] HTTP {resp.status_code} in {wall:.2f}s wall-clock")

    try:
        data = resp.json()
    except ValueError:
        print(resp.text)
        return 1 if resp.status_code >= 400 else 0

    print(json.dumps(data, indent=2, ensure_ascii=False))
    return 0 if resp.ok else 1


if __name__ == "__main__":
    args = sys.argv[1:]
    video = args[0] if len(args) > 0 else "sample.mp4"
    base = args[1] if len(args) > 1 else "http://localhost:8000"
    raise SystemExit(main(video, base))
