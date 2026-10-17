"""Compare the streaming acoustic detector with the legacy whole-buffer one.

Each detector runs in its own subprocess so peak RSS can be reported.

Usage (from the repository root or /app inside the image):
    PYTHONPATH=src python3 scripts/compare_acoustic_detector.py CUE EPISODE...
"""

import json
import subprocess
import sys
from pathlib import Path

RUNNER = r"""
import json, resource, sys
from dataclasses import asdict
from pathlib import Path
module = sys.argv[1]
if module == "legacy":
    from tests.legacy_acoustic_detector import detect_and_plan
else:
    from podcast_processor.acoustic_detector import detect_and_plan
plan = detect_and_plan(Path(sys.argv[2]), Path(sys.argv[3]), threshold=float(sys.argv[4]))
print(json.dumps({
    "matches": [asdict(m) for m in plan.matches],
    "cuts": [list(c) for c in plan.cuts],
    "abstained": plan.abstained,
    "peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
}))
"""

FRAME_SECONDS = 1 / 4000


def _run(module: str, episode: str, cue: str, threshold: str) -> dict:
    output = subprocess.run(
        [sys.executable, "-c", RUNNER, module, episode, cue, threshold],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return json.loads(output.strip().splitlines()[-1])


def main() -> int:
    cue, *episodes = sys.argv[1:]
    threshold = "0.90"
    failures = 0
    for episode in episodes:
        old = _run("legacy", episode, cue, threshold)
        new = _run("streaming", episode, cue, threshold)
        same_count = len(old["matches"]) == len(new["matches"])
        max_delta = (
            max(
                (
                    abs(a["start_seconds"] - b["start_seconds"])
                    for a, b in zip(old["matches"], new["matches"])
                ),
                default=0.0,
            )
            if same_count
            else None
        )
        identical = old["matches"] == new["matches"] and old["cuts"] == new["cuts"]
        within_frame = same_count and max_delta is not None and max_delta <= FRAME_SECONDS
        ok = identical or within_frame
        failures += 0 if ok else 1
        print(
            json.dumps(
                {
                    "episode": Path(episode).name,
                    "identical": identical,
                    "matches": len(new["matches"]),
                    "max_start_delta_s": max_delta,
                    "cuts_old": old["cuts"],
                    "cuts_new": new["cuts"],
                    "peak_rss_mib_old": old["peak_rss_mib"],
                    "peak_rss_mib_new": new["peak_rss_mib"],
                    "ok": ok,
                }
            )
        )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
