"""The streaming acoustic detector must match the legacy whole-buffer scan."""

import shutil
import subprocess
import wave
from dataclasses import astuple
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pytest

from podcast_processor import acoustic_detector as streaming
from tests import legacy_acoustic_detector as legacy

SAMPLE_RATE = 4000
CHUNK = int(90.0 * SAMPLE_RATE)


def _cue() -> np.ndarray:
    rng = np.random.default_rng(7)
    t = np.arange(int(1.5 * SAMPLE_RATE)) / SAMPLE_RATE
    chirp = np.sin(2 * np.pi * (200 + 600 * t) * t)
    return (chirp + 0.05 * rng.standard_normal(t.size)).astype(np.float64)


def _episode(cue: np.ndarray, seconds: int, positions: list[int]) -> np.ndarray:
    rng = np.random.default_rng(11)
    audio = 0.3 * rng.standard_normal(seconds * SAMPLE_RATE)
    for position in positions:
        audio[position : position + cue.size] += cue
    # Round-trip through float32 like the ffmpeg f32le pipe does.
    return audio.astype(np.float32).astype(np.float64)


def _rows(matches: Iterable[Any]) -> list[tuple[Any, ...]]:
    # Legacy and streaming modules define distinct (but identical) dataclasses.
    return [astuple(match) for match in matches]


def _blocks(audio: np.ndarray, size: int) -> list[np.ndarray]:
    return [audio[i : i + size] for i in range(0, audio.size, size)]


@pytest.mark.parametrize("block_size", [997, 4000, CHUNK - 1, CHUNK, CHUNK + 1, 10**7])
def test_streaming_matches_legacy_exactly(block_size: int) -> None:
    cue = _cue()
    positions = [
        30 * SAMPLE_RATE,
        CHUNK - cue.size // 2,  # straddles the first chunk boundary
        CHUNK * 2 - 1,
        CHUNK * 3 + 17,
        CHUNK * 5 - cue.size,  # ends exactly on a boundary
        CHUNK * 6 + 123,
    ]
    audio = _episode(cue, 10 * 60, positions)

    expected = legacy.find_cue_matches(
        audio, cue, sample_rate=SAMPLE_RATE, threshold=0.9
    )
    matches, total = streaming.scan_cue_matches(
        _blocks(audio, block_size), cue, sample_rate=SAMPLE_RATE, threshold=0.9
    )

    assert total == audio.size
    assert len(expected) == len(positions)
    assert _rows(matches) == _rows(expected)


@pytest.mark.parametrize("seconds", [1, 2, 89, 90, 91, 181])
def test_streaming_matches_legacy_on_short_and_edge_lengths(seconds: int) -> None:
    cue = _cue()
    positions = [0] if seconds * SAMPLE_RATE > cue.size else []
    if seconds > 60:
        positions.append(seconds * SAMPLE_RATE - cue.size)
    audio = _episode(cue, seconds, positions)

    expected = legacy.find_cue_matches(
        audio, cue, sample_rate=SAMPLE_RATE, threshold=0.9
    )
    matches, _ = streaming.scan_cue_matches(
        _blocks(audio, 5000), cue, sample_rate=SAMPLE_RATE, threshold=0.9
    )
    assert _rows(matches) == _rows(expected)


def test_in_memory_api_is_unchanged() -> None:
    cue = _cue()
    audio = _episode(cue, 200, [CHUNK - 10, CHUNK + 40 * SAMPLE_RATE])
    actual = streaming.find_cue_matches(
        audio, cue, sample_rate=SAMPLE_RATE, threshold=0.9
    )
    expected = legacy.find_cue_matches(
        audio, cue, sample_rate=SAMPLE_RATE, threshold=0.9
    )
    assert _rows(actual) == _rows(expected)


def _write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    pcm = np.clip(audio / np.max(np.abs(audio)), -1, 1)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes((pcm * 32000).astype("<i2").tobytes())


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")
def test_detect_and_plan_through_ffmpeg_matches_legacy(tmp_path: Path) -> None:
    rate = 16000
    rng = np.random.default_rng(3)
    t = np.arange(int(1.5 * rate)) / rate
    cue = np.sin(2 * np.pi * (200 + 600 * t) * t)
    audio = 0.3 * rng.standard_normal(12 * 60 * rate)
    for seconds in (40, 89.9, 200, 330, 690):
        start = int(seconds * rate)
        audio[start : start + cue.size] += cue
    episode = tmp_path / "episode.wav"
    cue_path = tmp_path / "cue.wav"
    _write_wav(episode, audio, rate)
    _write_wav(cue_path, cue, rate)

    expected = legacy.detect_and_plan(episode, cue_path, threshold=0.9)
    actual = streaming.detect_and_plan(episode, cue_path, threshold=0.9)

    assert expected.matches
    assert _rows(actual.matches) == _rows(expected.matches)
    assert actual.cuts == expected.cuts
    assert (actual.abstained, actual.reason) == (expected.abstained, expected.reason)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")
def test_ffmpeg_failure_is_raised(tmp_path: Path) -> None:
    missing = tmp_path / "missing.mp3"
    with pytest.raises(subprocess.CalledProcessError):
        streaming.detect_cue_matches(
            missing, _cue(), sample_rate=SAMPLE_RATE, threshold=0.9
        )
