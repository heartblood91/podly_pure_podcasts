import logging
import re
import shutil
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from groq import Groq
from openai import OpenAI
from openai.types.audio.transcription_segment import TranscriptionSegment
from pydantic import BaseModel

from podcast_processor.audio import get_audio_duration_ms, split_audio, trim_file
from shared.config import GroqWhisperConfig, RemoteWhisperConfig


class Segment(BaseModel):
    start: float
    end: float
    text: str


class Transcriber(ABC):

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

    @abstractmethod
    def transcribe(self, audio_file_path: str) -> List[Segment]:
        pass


class LocalTranscriptSegment(BaseModel):
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: List[int]
    temperature: float
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float

    def to_segment(self) -> Segment:
        return Segment(start=self.start, end=self.end, text=self.text)


class TestWhisperTranscriber(Transcriber):

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    @property
    def model_name(self) -> str:
        return "test_whisper"

    def transcribe(self, _: str) -> List[Segment]:
        self.logger.info("Using test whisper")
        return [
            Segment(start=0, end=1, text="This is a test"),
            Segment(start=1, end=2, text="This is another test"),
        ]


class LocalWhisperTranscriber(Transcriber):

    def __init__(self, logger: logging.Logger, whisper_model: str):
        self.logger = logger
        self.whisper_model = whisper_model

    @property
    def model_name(self) -> str:
        return f"local_{self.whisper_model}"

    @staticmethod
    def convert_to_pydantic(
        transcript_data: List[Any],
    ) -> List[LocalTranscriptSegment]:
        return [LocalTranscriptSegment(**item) for item in transcript_data]

    @staticmethod
    def local_seg_to_seg(local_segments: List[LocalTranscriptSegment]) -> List[Segment]:
        return [seg.to_segment() for seg in local_segments]

    def transcribe(self, audio_file_path: str) -> List[Segment]:
        # Import whisper only when needed to avoid CUDA dependencies during module import
        try:
            import whisper  # type: ignore[import-untyped]
        except ImportError as e:
            self.logger.error(f"Failed to import whisper: {e}")
            raise ImportError(
                "whisper library is required for LocalWhisperTranscriber"
            ) from e

        self.logger.info("Using local whisper")
        models = whisper.available_models()
        self.logger.info(f"Available models: {models}")

        model = whisper.load_model(name=self.whisper_model)

        self.logger.info("Beginning transcription")
        start = time.time()
        result = model.transcribe(audio_file_path, fp16=False, language="English")
        end = time.time()
        elapsed = end - start
        self.logger.info(f"Transcription completed in {elapsed}")
        segments = result["segments"]
        typed_segments = self.convert_to_pydantic(segments)

        return self.local_seg_to_seg(typed_segments)


class OpenAIWhisperTranscriber(Transcriber):

    def __init__(self, logger: logging.Logger, config: RemoteWhisperConfig):
        self.logger = logger
        self.config = config

        self.openai_client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.timeout_sec,
        )

    @property
    def model_name(self) -> str:
        return self.config.model  # e.g. "whisper-1"

    def transcribe(self, audio_file_path: str) -> List[Segment]:
        self.logger.info(
            "[WHISPER_REMOTE] Starting remote whisper transcription for: %s",
            audio_file_path,
        )
        audio_chunk_path = audio_file_path + "_parts"

        chunks = split_audio(
            Path(audio_file_path),
            Path(audio_chunk_path),
            self.config.chunksize_mb * 1024 * 1024,
        )

        self.logger.info("[WHISPER_REMOTE] Processing %d chunks", len(chunks))
        all_segments: List[TranscriptionSegment] = []

        for idx, chunk in enumerate(chunks):
            chunk_path, offset = chunk
            self.logger.info(
                "[WHISPER_REMOTE] Processing chunk %d/%d: %s",
                idx + 1,
                len(chunks),
                chunk_path,
            )
            segments = self.get_segments_for_chunk(str(chunk_path))
            self.logger.info(
                "[WHISPER_REMOTE] Chunk %d/%d complete: %d segments",
                idx + 1,
                len(chunks),
                len(segments),
            )
            all_segments.extend(self.add_offset_to_segments(segments, offset))

        shutil.rmtree(audio_chunk_path)
        self.logger.info(
            "[WHISPER_REMOTE] Transcription complete: %d total segments",
            len(all_segments),
        )
        return self.convert_segments(all_segments)

    @staticmethod
    def convert_segments(segments: List[TranscriptionSegment]) -> List[Segment]:
        return [
            Segment(
                start=seg.start,
                end=seg.end,
                text=seg.text,
            )
            for seg in segments
        ]

    @staticmethod
    def add_offset_to_segments(
        segments: List[TranscriptionSegment], offset_ms: int
    ) -> List[TranscriptionSegment]:
        offset_sec = float(offset_ms) / 1000.0
        for segment in segments:
            segment.start += offset_sec
            segment.end += offset_sec

        return segments

    def get_segments_for_chunk(self, chunk_path: str) -> List[TranscriptionSegment]:
        with open(chunk_path, "rb") as f:
            self.logger.info(
                "[WHISPER_API_CALL] Sending chunk to API: %s (timeout=%ds)",
                chunk_path,
                self.config.timeout_sec,
            )

            transcription = self.openai_client.audio.transcriptions.create(
                model=self.config.model,
                file=f,
                timestamp_granularities=["segment"],
                language=self.config.language,
                response_format="verbose_json",
            )

            self.logger.debug("Got transcription")

            segments = transcription.segments
            assert segments is not None

            self.logger.debug(f"Got {len(segments)} segments")

            return segments


class GroqTranscriptionSegment(BaseModel):
    start: float
    end: float
    text: str


class GroqWhisperTranscriber(Transcriber):

    LANGUAGE_RECOVERY_MIN_GAP_SEC = 12.0
    LANGUAGE_RECOVERY_MAX_GAP_SEC = 180.0
    LANGUAGE_RECOVERY_BOUNDARY_SEC = 120.0
    FRENCH_MARKERS = {
        "alors",
        "avec",
        "avoir",
        "beaucoup",
        "bien",
        "bonjour",
        "cette",
        "comme",
        "dans",
        "donc",
        "elle",
        "entre",
        "est",
        "faire",
        "mais",
        "nous",
        "parce",
        "pour",
        "quand",
        "sans",
        "sont",
        "sur",
        "tout",
        "une",
        "vous",
    }
    ENGLISH_MARKERS = {
        "about",
        "also",
        "and",
        "are",
        "because",
        "been",
        "but",
        "can",
        "for",
        "from",
        "have",
        "into",
        "just",
        "more",
        "not",
        "our",
        "that",
        "the",
        "their",
        "this",
        "what",
        "when",
        "with",
        "you",
        "your",
    }
    TRANSCRIPTION_HALLUCINATION_MARKERS = {
        "sous-titrage",
        "sous titrage",
        "subtitles by",
        "amara.org",
        "st' 501",
    }
    COMMERCIAL_MARKERS = {
        "sponsor",
        "sponsored",
        "book tickets",
        "buy now",
        "discount",
        "offer valid",
        "promo code",
        "search tfl",
        ".com",
        ".co.uk",
        "£",
        "$",
    }

    def __init__(self, logger: logging.Logger, config: GroqWhisperConfig):
        self.logger = logger
        self.config = config
        self.client = Groq(
            api_key=config.api_key,
            max_retries=config.max_retries,
        )

    @property
    def model_name(self) -> str:
        return f"groq_{self.config.model}"

    def transcribe(self, audio_file_path: str) -> List[Segment]:
        self.logger.info(
            "[WHISPER_GROQ] Starting Groq whisper transcription for: %s",
            audio_file_path,
        )
        audio_chunk_path = audio_file_path + "_parts"

        chunks = split_audio(
            Path(audio_file_path), Path(audio_chunk_path), 12 * 1024 * 1024
        )

        self.logger.info("[WHISPER_GROQ] Processing %d chunks", len(chunks))
        all_segments: List[GroqTranscriptionSegment] = []

        for idx, chunk in enumerate(chunks):
            chunk_path, offset = chunk
            self.logger.info(
                "[WHISPER_GROQ] Processing chunk %d/%d: %s",
                idx + 1,
                len(chunks),
                chunk_path,
            )
            segments = self.get_segments_for_chunk(str(chunk_path))
            self.logger.info(
                "[WHISPER_GROQ] Chunk %d/%d complete: %d segments",
                idx + 1,
                len(chunks),
                len(segments),
            )
            all_segments.extend(self.add_offset_to_segments(segments, offset))

        all_segments = self._recover_english_segments_in_french_audio(
            audio_file_path,
            Path(audio_chunk_path),
            all_segments,
        )

        shutil.rmtree(audio_chunk_path)
        self.logger.info(
            "[WHISPER_GROQ] Transcription complete: %d total segments",
            len(all_segments),
        )
        return self.convert_segments(all_segments)

    @staticmethod
    def convert_segments(segments: List[GroqTranscriptionSegment]) -> List[Segment]:
        return [
            Segment(
                start=seg.start,
                end=seg.end,
                text=seg.text,
            )
            for seg in segments
        ]

    @staticmethod
    def add_offset_to_segments(
        segments: List[GroqTranscriptionSegment], offset_ms: int
    ) -> List[GroqTranscriptionSegment]:
        offset_sec = float(offset_ms) / 1000.0
        for segment in segments:
            segment.start += offset_sec
            segment.end += offset_sec

        return segments

    @classmethod
    def _language_marker_counts(cls, text: str) -> Tuple[int, int]:
        words = re.findall(r"[a-zà-ÿ']+", text.lower())
        french_hits = sum(word in cls.FRENCH_MARKERS for word in words)
        english_hits = sum(word in cls.ENGLISH_MARKERS for word in words)
        return french_hits, english_hits

    @classmethod
    def _is_predominantly_french(
        cls, segments: List[GroqTranscriptionSegment]
    ) -> bool:
        text = " ".join(segment.text for segment in segments)
        french_hits, english_hits = cls._language_marker_counts(text)
        return french_hits >= 12 and french_hits >= max(english_hits * 1.5, 12)

    @classmethod
    def _looks_english(cls, segments: List[GroqTranscriptionSegment]) -> bool:
        text = " ".join(segment.text for segment in segments)
        french_hits, english_hits = cls._language_marker_counts(text)
        return english_hits >= 3 and english_hits >= max(french_hits * 1.5, 3)

    @classmethod
    def _looks_commercial(cls, segments: List[GroqTranscriptionSegment]) -> bool:
        text = " ".join(segment.text.lower() for segment in segments)
        return any(marker in text for marker in cls.COMMERCIAL_MARKERS)

    @classmethod
    def _looks_hallucinated(cls, text: str) -> bool:
        normalized_text = " ".join(text.lower().split())
        return any(
            marker in normalized_text
            for marker in cls.TRANSCRIPTION_HALLUCINATION_MARKERS
        )

    @classmethod
    def _candidate_transcript_gaps(
        cls,
        segments: List[GroqTranscriptionSegment],
        duration_sec: float,
    ) -> List[Tuple[float, float]]:
        if not segments or duration_sec <= 0:
            return []

        ordered = sorted(segments, key=lambda segment: (segment.start, segment.end))
        gaps: List[Tuple[float, float]] = []

        leading_end = min(ordered[0].start, cls.LANGUAGE_RECOVERY_BOUNDARY_SEC)
        if leading_end >= cls.LANGUAGE_RECOVERY_MIN_GAP_SEC:
            gaps.append((0.0, leading_end))

        for previous, current in zip(ordered, ordered[1:]):
            gap_start = max(previous.end, 0.0)
            gap_end = min(current.start, duration_sec)
            gap_duration = gap_end - gap_start
            if (
                cls.LANGUAGE_RECOVERY_MIN_GAP_SEC
                <= gap_duration
                <= cls.LANGUAGE_RECOVERY_MAX_GAP_SEC
            ):
                gaps.append((gap_start, gap_end))

        for segment in ordered:
            segment_duration = segment.end - segment.start
            normalized_text = " ".join(segment.text.lower().split())
            word_count = len(re.findall(r"[a-zà-ÿ']+", normalized_text))
            looks_hallucinated = any(
                marker in normalized_text
                for marker in cls.TRANSCRIPTION_HALLUCINATION_MARKERS
            )
            has_too_little_text = word_count <= max(5, int(segment_duration / 5))
            is_boundary_segment = (
                segment.start <= cls.LANGUAGE_RECOVERY_BOUNDARY_SEC
                or segment.end
                >= duration_sec - cls.LANGUAGE_RECOVERY_BOUNDARY_SEC
            )
            if (
                (
                    looks_hallucinated
                    and is_boundary_segment
                    and segment_duration >= 2.0
                )
                or (
                    segment_duration >= cls.LANGUAGE_RECOVERY_MIN_GAP_SEC
                    and has_too_little_text
                )
            ):
                gaps.append(
                    (
                        max(segment.start, 0.0),
                        min(segment.end, duration_sec),
                    )
                )

        trailing_start = max(
            ordered[-1].end,
            duration_sec - cls.LANGUAGE_RECOVERY_BOUNDARY_SEC,
        )
        if duration_sec - trailing_start >= cls.LANGUAGE_RECOVERY_MIN_GAP_SEC:
            gaps.append((trailing_start, duration_sec))

        merged_gaps: List[Tuple[float, float]] = []
        for gap_start, gap_end in sorted(set(gaps)):
            if merged_gaps and gap_start <= merged_gaps[-1][1] + 0.5:
                previous_start, previous_end = merged_gaps[-1]
                merged_gaps[-1] = (previous_start, max(previous_end, gap_end))
            else:
                merged_gaps.append((gap_start, gap_end))

        return merged_gaps

    def _recover_english_segments_in_french_audio(
        self,
        audio_file_path: str,
        work_directory: Path,
        segments: List[GroqTranscriptionSegment],
    ) -> List[GroqTranscriptionSegment]:
        if not self._is_predominantly_french(segments):
            self.logger.info(
                "[WHISPER_GROQ] Transcript is not predominantly French; "
                "skipping targeted English recovery"
            )
            return segments

        duration_ms = get_audio_duration_ms(audio_file_path)
        if duration_ms is None:
            self.logger.warning(
                "[WHISPER_GROQ] Could not determine audio duration; "
                "skipping targeted English recovery"
            )
            return segments

        duration_sec = duration_ms / 1000.0
        segments = [
            GroqTranscriptionSegment(
                start=max(0.0, min(segment.start, duration_sec)),
                end=max(0.0, min(segment.end, duration_sec)),
                text=segment.text,
            )
            for segment in segments
            if min(segment.end, duration_sec) > max(segment.start, 0.0)
        ]
        gaps = self._candidate_transcript_gaps(segments, duration_sec)
        self.logger.info(
            "[WHISPER_GROQ] Predominantly French transcript; checking %d "
            "transcript gaps for missed English speech",
            len(gaps),
        )

        recovered_segments: List[GroqTranscriptionSegment] = []
        replaced_ranges: List[Tuple[float, float]] = []
        for idx, (gap_start, gap_end) in enumerate(gaps):
            recovery_path = work_directory / f"english-recovery-{idx}.mp3"
            try:
                trim_file(
                    Path(audio_file_path),
                    recovery_path,
                    int(gap_start * 1000),
                    int(gap_end * 1000),
                )
                recovered = self.get_segments_for_chunk(
                    str(recovery_path), language_override="en"
                )
            except Exception as exc:  # noqa: BLE001
                self.logger.warning(
                    "[WHISPER_GROQ] English recovery failed for %.1f-%.1fs: %s",
                    gap_start,
                    gap_end,
                    exc,
                )
                continue

            gap_duration = gap_end - gap_start
            recovered = [
                GroqTranscriptionSegment(
                    start=max(0.0, min(segment.start, gap_duration)),
                    end=max(0.0, min(segment.end, gap_duration)),
                    text=segment.text,
                )
                for segment in recovered
                if min(segment.end, gap_duration) > max(segment.start, 0.0)
            ]
            if not self._looks_english(recovered):
                self.logger.info(
                    "[WHISPER_GROQ] Gap %.1f-%.1fs did not contain clear "
                    "English speech",
                    gap_start,
                    gap_end,
                )
                continue

            is_post_roll = gap_end >= duration_sec - 0.5
            if is_post_roll and self._looks_commercial(recovered):
                recovered = [
                    GroqTranscriptionSegment(
                        start=segment.start,
                        end=segment.end,
                        text=(
                            " English post-roll advertisement continues."
                            if self._looks_hallucinated(segment.text)
                            else segment.text
                        ),
                    )
                    for segment in recovered
                ]

            replaced_ranges.append((gap_start, gap_end))
            recovered_segments.extend(
                self.add_offset_to_segments(recovered, int(gap_start * 1000))
            )
            self.logger.info(
                "[WHISPER_GROQ] Recovered %d English segments from %.1f-%.1fs",
                len(recovered),
                gap_start,
                gap_end,
            )

        if not recovered_segments:
            return segments

        retained_segments = [
            segment
            for segment in segments
            if not any(
                segment.start >= range_start - 0.1
                and segment.end <= range_end + 0.1
                for range_start, range_end in replaced_ranges
            )
        ]
        return sorted(
            [*retained_segments, *recovered_segments],
            key=lambda segment: (segment.start, segment.end),
        )

    def get_segments_for_chunk(
        self, chunk_path: str, language_override: Optional[str] = None
    ) -> List[GroqTranscriptionSegment]:

        self.logger.info("[GROQ_API_CALL] Sending chunk to Groq API: %s", chunk_path)
        request_args: Dict[str, Any] = {
            "file": Path(chunk_path),
            "model": self.config.model,
            "response_format": "verbose_json",  # Ensure segments are included
        }
        language = (
            language_override
            if language_override is not None
            else (self.config.language or "").strip()
        )
        if language and language.lower() != "auto":
            request_args["language"] = language
            if language.lower() == "en" and language_override is not None:
                request_args["temperature"] = 0
                request_args["prompt"] = (
                    "Transcribe every spoken English word verbatim, including brand "
                    "names, prices, URLs, and legal terms. Do not translate and do not "
                    "invent subtitle credits."
                )
        else:
            self.logger.info(
                "[GROQ_API_CALL] Automatic language detection enabled for: %s",
                chunk_path,
            )

        transcription = self.client.audio.transcriptions.create(**request_args)
        self.logger.info(
            "[GROQ_API_CALL] Received response from Groq API for: %s", chunk_path
        )

        if transcription.segments is None:  # type: ignore [attr-defined]
            self.logger.warning(
                "[GROQ_API_CALL] No segments found in transcription for %s", chunk_path
            )
            return []

        groq_segments = [
            GroqTranscriptionSegment(
                start=seg["start"], end=seg["end"], text=seg["text"]
            )
            for seg in transcription.segments  # type: ignore [attr-defined]
        ]

        self.logger.info(
            "[GROQ_API_CALL] Got %d segments from chunk", len(groq_segments)
        )
        return groq_segments
