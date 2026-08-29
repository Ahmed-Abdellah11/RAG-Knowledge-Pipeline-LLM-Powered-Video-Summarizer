"""
embed_store.py
--------------
Embeds chunks with a sentence-transformer model and indexes them in FAISS
for fast semantic (cosine) similarity search. This is the "retrieve" half
of Retrieve-Augment-Generate.
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from .chunking import Chunk

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


class VectorStore:
    """Thin wrapper around a FAISS flat index using cosine similarity
    (via L2-normalized inner product)."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model = SentenceTransformer(model_name)
        self.index: faiss.Index | None = None
        self.chunks: List[Chunk] = []

    def build(self, chunks: List[Chunk]) -> None:
        self.chunks = chunks
        texts = [c.text for c in chunks]
        embeddings = self.model.encode(
            texts, convert_to_numpy=True, show_progress_bar=False
        )
        faiss.normalize_L2(embeddings)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # inner product == cosine on normalized vecs
        self.index.add(embeddings.astype(np.float32))

    def search(self, query: str, top_k: int = 4) -> List[RetrievedChunk]:
        if self.index is None:
            raise RuntimeError("VectorStore.build() must be called before search().")

        q_emb = self.model.encode([query], convert_to_numpy=True, show_progress_bar=False)
        faiss.normalize_L2(q_emb)
        scores, idxs = self.index.search(q_emb.astype(np.float32), top_k)

        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            results.append(RetrievedChunk(chunk=self.chunks[idx], score=float(score)))
        return results

    def most_salient(self, top_k: int = 4) -> List[RetrievedChunk]:
        """
        No-query mode (used for whole-video summarization): rank chunks by
        how 'central' they are to the rest of the transcript, using mean
        similarity to every other chunk as a proxy for salience.
        """
        if self.index is None:
            raise RuntimeError("VectorStore.build() must be called before most_salient().")

        all_embeddings = self.index.reconstruct_n(0, self.index.ntotal)
        sim_matrix = all_embeddings @ all_embeddings.T
        centrality = sim_matrix.mean(axis=1)

        ranked_idx = np.argsort(-centrality)[:top_k]
        return [
            RetrievedChunk(chunk=self.chunks[i], score=float(centrality[i]))
            for i in ranked_idx
        ]
