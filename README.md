# RAG Knowledge Pipeline — LLM-Powered Video Summarizer

A Retrieve-Augment-Generate (RAG) pipeline that ingests YouTube video transcripts,
semantically retrieves the most relevant/salient segments, and generates
abstractive summaries or answers to questions using a Hugging Face
summarization model. **Works for both Arabic and English videos** (and, as a
side effect of the model choice, 43 other languages) through the same code
path — no per-language branching needed.

```
Ingest              Chunk                 Retrieve                  Generate
──────────          ──────────            ──────────────            ────────────────
YouTube      -->    sentence-aware  -->   multilingual        -->   mT5_multilingual
transcript          overlapping           sentence-transformer      _XLSum abstractive
API (ar/en)          chunks                + FAISS cosine sim.       summary / answer
```

## Why RAG here

Feeding an entire transcript into a summarization model isn't efficient or
scalable — long videos blow past model context limits and dilute what
actually matters. This pipeline instead:

1. **Chunks** the transcript into overlapping, sentence-aware segments so no
   idea gets split across a boundary — sentence boundaries are detected for
   both Latin punctuation (`.!?`) and Arabic punctuation (`؟`, `۔`).
2. **Embeds** every chunk with
   `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (trained
   across 50+ languages, including Arabic) and indexes them in a **FAISS**
   vector store.
3. **Retrieves** the top-k most relevant chunks — either by similarity to a
   user's question, or by centrality/salience when the goal is a whole-video
   summary (no query). Works the same way regardless of language, since the
   embedding space is shared across languages.
4. **Generates** the final abstractive summary or answer with
   `csebuetnlp/mT5_multilingual_XLSum` (mT5 fine-tuned on the XL-Sum dataset
   across 45 languages), using map-reduce summarization so retrieval context
   that exceeds the model's 512-token input limit is handled safely.

**Language handling:** `src/ingest.py` fetches YouTube captions trying
Arabic first, then English, then whatever track is available (manual or
auto-generated), and `detect_language()` labels the transcript for the UI.
No language-specific branching is needed elsewhere in the pipeline — the
embedding and generation models are multilingual by design, so an Arabic
video and an English video flow through the exact same code path.

## Project structure

```
rag-video-summarizer/
├── app.py                  # Streamlit UI (live demo)
├── src/
│   ├── ingest.py            # YouTube transcript fetching + text normalization
│   ├── chunking.py          # Sentence-aware overlapping chunking
│   ├── embed_store.py       # sentence-transformers + FAISS retrieval
│   ├── summarizer.py        # BART-based abstractive generation
│   └── pipeline.py          # Orchestrates ingest -> chunk -> retrieve -> generate
├── tests/
│   └── test_chunking.py     # Offline unit tests (no model download required)
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone <your-repo-url>
cd rag-video-summarizer
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

First run will download the embedding model (~90MB) and BART-large-CNN
(~1.6GB) from Hugging Face — this requires internet access and a few minutes,
one time only (cached locally afterwards).

## Run the demo locally

```bash
streamlit run app.py
```

Then paste a YouTube URL (or raw transcript text) and click **Run pipeline**.
Leave the question field blank for a whole-video summary, or fill it in to
get a retrieval-grounded answer to a specific question about the video.

## Run the pipeline from code

```python
from src.pipeline import RAGVideoSummarizer

pipeline = RAGVideoSummarizer()

# Whole-video summary
result = pipeline.run_from_youtube("https://www.youtube.com/watch?v=VIDEO_ID")
print(result.output)

# Question-answering over the video (works in Arabic or English —
# whichever language the transcript and question are in)
result = pipeline.run_from_youtube(
    "https://www.youtube.com/watch?v=VIDEO_ID",
    question="What are the main steps described?",
)
print(result.output)
print(result.language)  # "en" or "ar"
```

## Tests

```bash
pytest tests/ -v
```

The chunking logic is covered by offline unit tests that don't require any
model downloads, so they run instantly in CI.

## Deploying a live demo (for a portfolio / CV link)

Free options that work well with this repo as-is:

- **Streamlit Community Cloud** — connect your GitHub repo at
  [streamlit.io/cloud](https://streamlit.io/cloud), set the main file to
  `app.py`, deploy. Free tier is CPU-only; first load is slower while models
  download and cache.
- **Hugging Face Spaces** — create a Space with the "Streamlit" SDK, push
  this repo to it. Good fit since the models already come from the HF Hub.

Either gives you a public URL you can link directly from your CV/portfolio.

## Notes on design decisions

- **Sentence-aware chunking with overlap** avoids cutting sentences (and
  therefore meaning) at arbitrary character boundaries, while overlap
  prevents losing context that straddles two chunks.
- **Cosine similarity via FAISS `IndexFlatIP`** on L2-normalized vectors —
  exact search, appropriate at this scale (a single video's transcript);
  swap in `IndexIVFFlat` or a managed vector DB if scaling to a large
  corpus of videos.
- **Map-reduce summarization** in `summarizer.py` handles the case where
  retrieved context exceeds BART's input limit, instead of silently
  truncating and losing content from later chunks.
- **Salience-based retrieval when there's no question** (`most_salient` in
  `embed_store.py`) — ranks chunks by average similarity to every other
  chunk, a lightweight proxy for "how central is this idea to the video as
  a whole," so the summary isn't just the first few minutes of content.

## Troubleshooting

**`ImportError` / tokenizer errors related to `sentencepiece`**
The mT5 tokenizer requires `sentencepiece`. It's in `requirements.txt`, but
if you're updating an existing environment run
`pip install sentencepiece` explicitly.

**`KeyError: Unknown task summarization, available tasks are [...]`**
This happens if `pip install -r requirements.txt` resolved `transformers>=5`,
which removed the `pipeline("summarization", ...)` task shortcut entirely
(see the [v5 migration guide](https://github.com/huggingface/transformers/blob/main/MIGRATION_GUIDE_V5.md)).
`src/summarizer.py` already avoids this by calling
`AutoModelForSeq2SeqLM.generate()` directly instead of the `pipeline()`
task-string API, so it works on both transformers v4 and v5 — make sure
you're on the current version of this file if you hit that error.

## Possible extensions

- Swap `facebook/bart-large-cnn` for a larger instruction-tuned model for
  better question-answering (BART is summarization-specialized, not QA).
- Add persistent vector storage (e.g. per-channel or per-playlist indexes)
  instead of rebuilding the FAISS index per run.
- Add multi-video retrieval — index several videos and retrieve across all
  of them for a "search my watch history" style assistant.

## License

MIT
