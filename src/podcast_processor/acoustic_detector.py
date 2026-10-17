"""Conservative acoustic cue matching for Podly experiments.

The detector streams audio decoded by ffmpeg and performs chunked normalized
cross-correlation against a short PCM cue, keeping only one chunk plus a
cue-length overlap in memory. It reports matches only; it never decides or
applies cuts on its own.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np


@dataclass(frozen=True)
class CueMatch:
    start_seconds: float
    end_seconds: float
    confidence: float


@dataclass(frozen=True)
class AcousticPlan:
    matches: tuple[CueMatch, ...]
    cuts: tuple[tuple[float, float], ...]
    abstained: bool
    reason: str


def _decode_command(path: Path, sample_rate: int) -> list[str]:
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "pipe:1",
    ]


def decode_audio(path: Path, sample_rate: int) -> np.ndarray:
    command = _decode_command(path, sample_rate)
    completed = subprocess.run(command, check=True, stdout=subprocess.PIPE)
    samples = np.frombuffer(completed.stdout, dtype="<f4").astype(np.float64)
    if samples.size == 0:
        raise ValueError(f"No audio samples decoded from {path}")
    return samples


def iter_audio_blocks(
    path: Path, sample_rate: int, block_samples: int
) -> Iterator[np.ndarray]:
    """Yield float64 sample blocks straight from the ffmpeg pipe."""
    command = _decode_command(path, sample_rate)
    block_bytes = block_samples * 4
    with subprocess.Popen(command, stdout=subprocess.PIPE) as proc:
        assert proc.stdout is not None
        while True:
            data = proc.stdout.read(block_bytes)
            if not data:
                break
            usable = len(data) - len(data) % 4
            yield np.frombuffer(data[:usable], dtype="<f4").astype(np.float64)
        returncode = proc.wait()
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)


def _next_power_of_two(value: int) -> int:
    return 1 << max(1, math.ceil(math.log2(value)))


def _normalized_scores(segment: np.ndarray, cue: np.ndarray) -> np.ndarray:
    cue_centered = cue - np.mean(cue)
    cue_energy = float(np.dot(cue_centered, cue_centered))
    if cue_energy <= 1e-12:
        raise ValueError("Cue has insufficient energy")

    valid_count = segment.size - cue.size + 1
    if valid_count <= 0:
        return np.empty(0, dtype=np.float64)

    fft_size = _next_power_of_two(segment.size + cue.size - 1)
    convolution = np.fft.irfft(
        np.fft.rfft(segment, fft_size)
        * np.fft.rfft(cue_centered[::-1], fft_size),
        fft_size,
    )
    numerator = convolution[cue.size - 1 : cue.size - 1 + valid_count]

    prefix = np.concatenate(([0.0], np.cumsum(segment)))
    prefix_sq = np.concatenate(([0.0], np.cumsum(segment * segment)))
    window_sum = prefix[cue.size :] - prefix[: -cue.size]
    window_sum_sq = prefix_sq[cue.size :] - prefix_sq[: -cue.size]
    window_energy = window_sum_sq - (window_sum * window_sum / cue.size)
    denominator = np.sqrt(np.maximum(window_energy * cue_energy, 1e-18))
    return numerator / denominator


def _cluster_peaks(
    candidates: Iterable[tuple[int, float]], min_distance_samples: int
) -> list[tuple[int, float]]:
    ordered = sorted(candidates, key=lambda item: item[1], reverse=True)
    selected: list[tuple[int, float]] = []
    for sample_index, score in ordered:
        if all(
            abs(sample_index - existing_index) >= min_distance_samples
            for existing_index, _ in selected
        ):
            selected.append((sample_index, score))
    return sorted(selected)


def _chunk_samples(cue: np.ndarray, sample_rate: int, chunk_seconds: float) -> int:
    return max(cue.size * 2, int(chunk_seconds * sample_rate))


def _owned_candidates(
    segment: np.ndarray,
    cue: np.ndarray,
    *,
    global_start: int,
    segment_start: int,
    owned_end: int,
    threshold: float,
) -> list[tuple[int, float]]:
    scores = _normalized_scores(segment, cue)
    local_owned_start = global_start - segment_start
    local_owned_end = owned_end - segment_start
    owned_scores = scores[local_owned_start:local_owned_end]
    above = np.flatnonzero(owned_scores >= threshold)
    if above.size == 0:
        return []

    candidates: list[tuple[int, float]] = []
    contiguous_groups = np.split(above, np.where(np.diff(above) > 1)[0] + 1)
    for group in contiguous_groups:
        group_scores = owned_scores[group]
        best_offset = int(group[int(np.argmax(group_scores))])
        candidates.append(
            (global_start + best_offset, float(owned_scores[best_offset]))
        )
    return candidates


def scan_cue_matches(
    blocks: Iterable[np.ndarray],
    cue: np.ndarray,
    *,
    sample_rate: int,
    threshold: float,
    chunk_seconds: float = 90.0,
    min_distance_seconds: float = 1.0,
) -> tuple[list[CueMatch], int]:
    """Match ``cue`` against audio arriving as consecutive sample blocks.

    Chunk boundaries, segments (owned chunk plus ``cue.size - 1`` samples of
    overlap on each side) and offsets are exactly those of a whole-buffer scan,
    so results are identical while memory stays bounded to roughly two chunks.
    Returns the matches and the total number of samples seen.
    """
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1]")

    chunk_samples = _chunk_samples(cue, sample_rate, chunk_seconds)
    overlap = cue.size - 1
    block_iter = iter(blocks)
    buffer = np.empty(0, dtype=np.float64)
    buffer_start = 0
    total_samples = 0
    exhausted = False
    global_start = 0
    candidates: list[tuple[int, float]] = []

    def scan(owned_end: int, segment_end: int) -> None:
        segment_start = max(0, global_start - overlap)
        segment = buffer[segment_start - buffer_start : segment_end - buffer_start]
        candidates.extend(
            _owned_candidates(
                segment,
                cue,
                global_start=global_start,
                segment_start=segment_start,
                owned_end=owned_end,
                threshold=threshold,
            )
        )

    def advance() -> None:
        nonlocal global_start, buffer, buffer_start
        global_start += chunk_samples
        keep_from = max(0, global_start - overlap)
        buffer = buffer[keep_from - buffer_start :]
        buffer_start = keep_from

    # Full chunks: enough samples are buffered to know the chunk is not the last.
    while True:
        needed_end = global_start + chunk_samples + overlap
        pending = [buffer]
        available_end = buffer_start + buffer.size
        while available_end < needed_end:
            try:
                block = next(block_iter)
            except StopIteration:
                exhausted = True
                break
            pending.append(block)
            available_end += block.size
            total_samples += block.size
        if len(pending) > 1:
            buffer = np.concatenate(pending)
        if exhausted:
            break
        scan(global_start + chunk_samples, needed_end)
        advance()

    # Tail: the total length is now known.
    if total_samples == 0:
        raise ValueError("No audio samples decoded")
    if cue.size >= total_samples:
        return [], total_samples
    last_start = total_samples - cue.size
    while global_start <= last_start:
        owned_end = min(global_start + chunk_samples, last_start + 1)
        scan(owned_end, min(total_samples, owned_end + overlap))
        advance()

    peaks = _cluster_peaks(
        candidates, max(1, int(min_distance_seconds * sample_rate))
    )
    cue_duration = cue.size / sample_rate
    return [
        CueMatch(
            start_seconds=round(sample_index / sample_rate, 3),
            end_seconds=round(sample_index / sample_rate + cue_duration, 3),
            confidence=round(score, 6),
        )
        for sample_index, score in peaks
    ], total_samples


def find_cue_matches(
    audio: np.ndarray,
    cue: np.ndarray,
    *,
    sample_rate: int,
    threshold: float,
    chunk_seconds: float = 90.0,
    min_distance_seconds: float = 1.0,
) -> list[CueMatch]:
    if cue.size >= audio.size:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        return []
    matches, _ = scan_cue_matches(
        [audio],
        cue,
        sample_rate=sample_rate,
        threshold=threshold,
        chunk_seconds=chunk_seconds,
        min_distance_seconds=min_distance_seconds,
    )
    return matches


def detect_cue_matches(
    audio_path: Path,
    cue: np.ndarray,
    *,
    sample_rate: int,
    threshold: float,
    chunk_seconds: float = 90.0,
) -> tuple[list[CueMatch], int]:
    """Stream ``audio_path`` through ffmpeg and match ``cue`` chunk by chunk."""
    blocks = iter_audio_blocks(
        audio_path, sample_rate, _chunk_samples(cue, sample_rate, chunk_seconds)
    )
    try:
        with closing(blocks):
            return scan_cue_matches(
                blocks,
                cue,
                sample_rate=sample_rate,
                threshold=threshold,
                chunk_seconds=chunk_seconds,
            )
    except ValueError as exc:
        if str(exc) == "No audio samples decoded":
            raise ValueError(f"No audio samples decoded from {audio_path}") from exc
        raise


def plan_conservative_cuts(
    matches: list[CueMatch],
    *,
    audio_duration_seconds: float,
    min_preroll_seconds: float = 5.0,
    max_preroll_seconds: float = 180.0,
    max_postroll_seconds: float = 180.0,
    min_midroll_seconds: float = 5.0,
    max_midroll_seconds: float = 300.0,
    max_removed_fraction: float = 0.30,
) -> AcousticPlan:
    """Build only high-confidence cuts, preserving the return cue.

    A cue near the beginning closes a preroll. A cue near the end opens a
    postroll. Remaining cues are paired in chronological order around midrolls.
    Unpaired or implausibly spaced cues are left untouched.
    """

    ordered = sorted(matches, key=lambda match: match.start_seconds)
    remaining = list(ordered)
    cuts: list[tuple[float, float]] = []

    if remaining and min_preroll_seconds <= remaining[0].start_seconds <= max_preroll_seconds:
        first = remaining.pop(0)
        cuts.append((0.0, first.start_seconds))

    postroll: CueMatch | None = None
    if (
        remaining
        and len(remaining) % 2 == 1
        and audio_duration_seconds - remaining[-1].start_seconds
        <= max_postroll_seconds
    ):
        postroll = remaining.pop()

    for index in range(0, len(remaining) - 1, 2):
        opening = remaining[index]
        closing = remaining[index + 1]
        duration = closing.start_seconds - opening.start_seconds
        if min_midroll_seconds <= duration <= max_midroll_seconds:
            cuts.append((opening.start_seconds, closing.start_seconds))

    if postroll is not None:
        cuts.append((postroll.start_seconds, audio_duration_seconds))

    cuts.sort()
    removed_seconds = sum(end - start for start, end in cuts)
    removed_fraction = removed_seconds / max(audio_duration_seconds, 1.0)
    if removed_fraction > max_removed_fraction:
        return AcousticPlan(
            matches=tuple(ordered),
            cuts=(),
            abstained=True,
            reason=(
                f"planned removal {removed_fraction:.1%} exceeds "
                f"limit {max_removed_fraction:.1%}"
            ),
        )

    return AcousticPlan(
        matches=tuple(ordered),
        cuts=tuple(cuts),
        abstained=False,
        reason="conservative acoustic plan",
    )


def detect_and_plan(
    audio_path: Path,
    cue_path: Path,
    *,
    sample_rate: int = 4000,
    threshold: float = 0.90,
) -> AcousticPlan:
    cue = decode_audio(cue_path, sample_rate)
    matches, total_samples = detect_cue_matches(
        audio_path,
        cue,
        sample_rate=sample_rate,
        threshold=threshold,
    )
    return plan_conservative_cuts(
        matches,
        audio_duration_seconds=total_samples / sample_rate,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument("cue", type=Path)
    parser.add_argument("--sample-rate", type=int, default=4000)
    parser.add_argument("--threshold", type=float, default=0.72)
    args = parser.parse_args()

    cue = decode_audio(args.cue, args.sample_rate)
    matches, total_samples = detect_cue_matches(
        args.audio,
        cue,
        sample_rate=args.sample_rate,
        threshold=args.threshold,
    )
    plan = plan_conservative_cuts(
        matches,
        audio_duration_seconds=total_samples / args.sample_rate,
    )
    print(
        json.dumps(
            {
                "audio_duration_seconds": round(total_samples / args.sample_rate, 3),
                "cue_duration_seconds": round(cue.size / args.sample_rate, 3),
                "threshold": args.threshold,
                "matches": [asdict(match) for match in matches],
                "cuts": [list(cut) for cut in plan.cuts],
                "abstained": plan.abstained,
                "reason": plan.reason,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
