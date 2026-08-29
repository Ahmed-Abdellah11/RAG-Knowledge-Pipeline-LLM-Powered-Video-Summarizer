"""
chunking.py
-----------
Splits long transcripts into overlapping, semantically coherent chunks
before embedding. Sentence-aware so chunk boundaries don't cut words
or ideas in half.
"""

import re
from dataclasses import dataclass
from typing import List


@dataclass
class Chunk:
    id: int
    text: str
    start_word: int
    end_word: int


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?\u061F\u06D4])\s+")


def split_sentences(text: str) -> List[str]:
    sentences = _SENTENCE_SPLIT.split(text.strip())
    return [s.strip() for s in sentences if s.strip()]


def chunk_transcript(
    text: str,
    max_words: int = 180,
    overlap_words: int = 30,
) -> List[Chunk]:
    """
    Sentence-aware sliding-window chunking.

    We greedily pack sentences into a chunk until adding the next sentence
    would exceed `max_words`, then start a new chunk that re-includes the
    trailing `overlap_words` words from the previous chunk so retrieval
    doesn't lose context that straddles a boundary.
    """
    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks: List[Chunk] = []
    current_words: List[str] = []
    word_cursor = 0
    chunk_start_word = 0

    def flush(next_start_offset: int):
        nonlocal current_words, chunk_start_word
        if not current_words:
            return
        chunk_text = " ".join(current_words)
        chunks.append(
            Chunk(
                id=len(chunks),
                text=chunk_text,
                start_word=chunk_start_word,
                end_word=chunk_start_word + len(current_words),
            )
        )
        overlap = current_words[-overlap_words:] if overlap_words > 0 else []
        chunk_start_word = chunk_start_word + len(current_words) - len(overlap)
        current_words = list(overlap)

    for sentence in sentences:
        words = sentence.split()
        if current_words and len(current_words) + len(words) > max_words:
            flush(word_cursor)
        current_words.extend(words)
        word_cursor += len(words)

    flush(word_cursor)
    return chunks
