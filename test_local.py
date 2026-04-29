"""Local smoke test for QwenActivityDescriber.

Drop a `sample.mp4` (any short clip) next to this file and run:
    python test_local.py

Useful for iterating on the prompt and frame-sampling settings without
spinning up the FastAPI server every time.
"""

import os
import sys
import time

from inference import QwenActivityDescriber


def main(video_path: str = "sample.mp4") -> int:
    if not os.path.exists(video_path):
        print(f"[error] video not found: {video_path}")
        print("        place a short .mp4 here or pass a path as an argument.")
        return 1

    abs_path = os.path.abspath(video_path)
    video_uri = f"file://{abs_path}"

    print(f"[info] loading model...")
    t0 = time.perf_counter()
    describer = QwenActivityDescriber()
    print(f"[info] model ready in {time.perf_counter() - t0:.1f}s")

    print(f"[info] describing: {abs_path}")
    t1 = time.perf_counter()
    description = describer.describe_activity(video_source=video_uri)
    elapsed = time.perf_counter() - t1

    print("\n--- Activity Description ---")
    print(description)
    print("----------------------------")
    print(f"[info] inference latency: {elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "sample.mp4"
    raise SystemExit(main(path))
