"""
pipeline.py
-----------
Orchestrates the full RAG Knowledge Pipeline:

    Ingest -> Chunk -> Retrieve (Augment) -> Generate

This is the class the Streamlit app (and any other frontend, or a CLI)
should call into. Keeping orchestration separate from the UI makes the
pipeline reusable and testable on its own.
"""

from dataclasses import dataclass
from typing import List, Optional

from .chunking import chunk_transcript, Chunk
from .embed_store import VectorStore, RetrievedChunk
from .ingest import fetch_transcript, segments_to_text, load_pasted_text, detect_language
from .summarizer import Summarizer


@dataclass
class PipelineResult:
    transcript: str
    chunks: List[Chunk]
    retrieved: List[RetrievedChunk]
    mode: str          # "summary" or "qa"
    output: str
    language: str       # "ar" or "en", detected from the transcript


class RAGVideoSummarizer:
    def __init__(
        self,
        embed_model: Optional[str] = None,
        summarizer_model: Optional[str] = None,
        max_words_per_chunk: int = 180,
        overlap_words: int = 30,
    ):
        self.store = VectorStore(embed_model) if embed_model else VectorStore()
        self.summarizer = Summarizer(summarizer_model) if summarizer_model else Summarizer()
        self.max_words_per_chunk = max_words_per_chunk
        self.overlap_words = overlap_words

    def run_from_youtube(
        self,
        url_or_id: str,
        question: Optional[str] = None,
        top_k: int = 4,
        languages: Optional[List[str]] = None,
    ) -> PipelineResult:
        segments = fetch_transcript(url_or_id, languages=languages)
        transcript = segments_to_text(segments)
        return self._run(transcript, question=question, top_k=top_k)

    def run_from_text(
        self,
        raw_text: str,
        question: Optional[str] = None,
        top_k: int = 4,
    ) -> PipelineResult:
        transcript = load_pasted_text(raw_text)
        return self._run(transcript, question=question, top_k=top_k)

    def _run(self, transcript: str, question: Optional[str], top_k: int) -> PipelineResult:
        # 1. Chunk
        chunks = chunk_transcript(
            transcript,
            max_words=self.max_words_per_chunk,
            overlap_words=self.overlap_words,
        )
        if not chunks:
            raise ValueError("Transcript produced no usable chunks.")

        # 2. Embed + index (retrieval side of RAG)
        self.store.build(chunks)

        # 3. Retrieve (+ augment)
        if question:
            retrieved = self.store.search(question, top_k=top_k)
        else:
            retrieved = self.store.most_salient(top_k=top_k)

        chunk_texts = [r.chunk.text for r in retrieved]

        # 4. Generate
        if question:
            output = self.summarizer.answer_question(chunk_texts, question)
            mode = "qa"
        else:
            output = self.summarizer.summarize_chunks(chunk_texts)
            mode = "summary"

        return PipelineResult(
            transcript=transcript,
            chunks=chunks,
            retrieved=retrieved,
            mode=mode,
            output=output,
            language=detect_language(transcript),
        )
