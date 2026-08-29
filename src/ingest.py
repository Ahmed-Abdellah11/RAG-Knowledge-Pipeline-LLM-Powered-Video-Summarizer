"""
ingest.py
---------
Handles ingestion of source text for the RAG pipeline.

Primary path: pull the transcript directly from YouTube using the video ID
or full URL (youtube-transcript-api, no API key required).

Fallback path: accept raw pasted text (useful for videos without captions,
or for testing offline).
"""

import re
from dataclasses import dataclass
from typing import List, Optional

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)


@dataclass
class TranscriptSegment:
    text: str
    start: float
    duration: float


def extract_video_id(url_or_id: str) -> str:
    """Accepts a raw video ID or a full YouTube URL and returns the video ID."""
    url_or_id = url_or_id.strip()

    # Already looks like a bare video ID (11 chars, no slashes/dots)
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url_or_id):
        return url_or_id

    patterns = [
        r"(?:v=|\/videos\/|embed\/|youtu\.be\/|\/v\/|\/e\/|watch\?v=)([A-Za-z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)

    raise ValueError(f"Could not extract a YouTube video ID from: {url_or_id!r}")


def fetch_transcript(
    url_or_id: str,
    languages: Optional[List[str]] = None,
) -> List[TranscriptSegment]:
    """
    Fetch the transcript for a YouTube video. Tries Arabic first, then
    English, then falls back to whatever caption track the video has
    (manually created or auto-generated) so both Arabic- and
    English-language videos work out of the box without the caller having
    to know which language a given video is in.

    Raises:
        TranscriptsDisabled, NoTranscriptFound, VideoUnavailable: bubbled up
        from youtube-transcript-api so the caller can show a clear message.
    """
    video_id = extract_video_id(url_or_id)
    languages = languages or ["ar", "en"]

    try:
        raw = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
    except NoTranscriptFound:
        # Fall back to any available transcript (manual or auto-generated,
        # in whatever language YouTube offers) rather than failing outright.
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript = next(iter(transcript_list))
        raw = transcript.fetch()

    return [
        TranscriptSegment(text=item["text"], start=item["start"], duration=item["duration"])
        for item in raw
    ]


_ARABIC_RANGE = re.compile(r"[\u0600-\u06FF]")


def detect_language(text: str) -> str:
    """
    Lightweight heuristic language check used only for UI labeling (the
    embedding and summarization models themselves are multilingual and
    don't need to be told which language they're looking at). Returns
    'ar' if at least ~15% of characters are Arabic script, else 'en'.
    """
    sample = text[:2000]
    letters = [c for c in sample if c.isalpha()]
    if not letters:
        return "en"
    arabic_ratio = sum(1 for c in letters if _ARABIC_RANGE.match(c)) / len(letters)
    return "ar" if arabic_ratio > 0.15 else "en"


def segments_to_text(segments: List[TranscriptSegment]) -> str:
    """Flattens transcript segments into a single cleaned block of text."""
    text = " ".join(seg.text for seg in segments)
    text = re.sub(r"\[.*?\]", " ", text)   # strip [Music], [Applause], etc.
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_pasted_text(raw_text: str) -> str:
    """Fallback path: normalize manually-pasted transcript text."""
    text = re.sub(r"\s+", " ", raw_text).strip()
    if not text:
        raise ValueError("Pasted transcript text is empty.")
    return text
