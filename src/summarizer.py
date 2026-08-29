"""
summarizer.py
-------------
The "generate" half of RAG: takes the augmented context (retrieved chunks)
and produces a fluent abstractive summary or a direct answer to a user
question.

Model: csebuetnlp/mT5_multilingual_XLSum, an mT5 checkpoint fine-tuned on
the XL-Sum dataset across 45 languages, including Arabic and English -- so
the same single model handles Arabic-language and English-language video
transcripts without needing separate per-language models. (The project
previously used facebook/bart-large-cnn, which is English-only; swapped
out for multilingual support.)

The model's encoder is capped at 512 input tokens (per the model card), so
long augmented contexts are map-reduced: each retrieved chunk is summarized
individually, then the partial summaries are combined and summarized once
more into the final output.

Implementation note: we deliberately do NOT use
`pipeline("summarization", ...)`. Transformers 5.x removed the
`SummarizationPipeline` / `Text2TextGenerationPipeline` task registrations
(see https://github.com/huggingface/transformers/blob/main/MIGRATION_GUIDE_V5.md),
so that call raises `KeyError: Unknown task summarization` on any
environment that resolved transformers>=5. Calling
`AutoModelForSeq2SeqLM.generate()` directly is the underlying mechanism the
old pipeline used internally, and it's stable across both v4 and v5.
"""

import re
from typing import List

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

DEFAULT_MODEL = "csebuetnlp/mT5_multilingual_XLSum"
MAX_INPUT_TOKENS = 512  # per the mT5_multilingual_XLSum model card

_WHITESPACE_RE = re.compile(r"\s+")


def _clean_whitespace(text: str) -> str:
    """Matches the WHITESPACE_HANDLER preprocessing used in the model card
    recipe -- mT5_multilingual_XLSum is sensitive to irregular whitespace/
    newlines, which are common in scraped transcripts."""
    return _WHITESPACE_RE.sub(" ", text.replace("\n", " ")).strip()


class Summarizer:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: int = -1):
        # device: -1 -> CPU, 0 -> first CUDA GPU (mirrors the old pipeline() convention).
        self.device = torch.device("cuda" if device >= 0 and torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def _summarize_text(self, text: str, max_length: int = 84, min_length: int = 20) -> str:
        text = _clean_whitespace(text)
        words = text.split()
        if len(words) < 15:
            return text  # too short to bother summarizing further

        inputs = self.tokenizer(
            [text],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
        ).to(self.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_length=max_length,
                min_length=min_length,
                num_beams=4,
                no_repeat_ngram_size=2,
            )

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    def summarize_chunks(self, chunk_texts: List[str]) -> str:
        """Map-reduce summarization across the retrieved/augmented chunks."""
        if not chunk_texts:
            return ""

        partial_summaries = [self._summarize_text(t, max_length=100, min_length=25) for t in chunk_texts]
        combined = " ".join(partial_summaries)

        if len(chunk_texts) == 1:
            return partial_summaries[0]

        return self._summarize_text(combined, max_length=180, min_length=50)

    def answer_question(self, chunk_texts: List[str], question: str) -> str:
        """
        mT5_multilingual_XLSum is fine-tuned for summarization, not
        instruction-following QA, so for question-answering we summarize
        the augmented context down to the essentials, keeping only the
        parts most relevant to the question by prepending it as framing
        context before summarization. Works in Arabic or English depending
        on the language the question and transcript are written in.
        """
        context = " ".join(chunk_texts)
        framed = f"Question: {question}\nRelevant transcript context: {context}"
        return self._summarize_text(framed, max_length=150, min_length=30)
