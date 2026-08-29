"""
app.py
------
Streamlit front-end for the RAG Knowledge Pipeline video summarizer.

Run locally:
    streamlit run app.py

Deploy for free (to get a live link for your CV/portfolio):
    - Streamlit Community Cloud (streamlit.io/cloud) -> point at this repo
    - or Hugging Face Spaces (Streamlit SDK)
"""

import streamlit as st
from src.pipeline import RAGVideoSummarizer

st.set_page_config(page_title="RAG Video Summarizer", page_icon="🎬", layout="wide")


@st.cache_resource(show_spinner="Loading embedding + summarization models (first run only)...")
def get_pipeline():
    return RAGVideoSummarizer()


st.title("🎬 RAG Knowledge Pipeline — Video Summarizer")
st.caption(
    "Ingest → Chunk → Retrieve (semantic search) → Generate (abstractive summary). "
    "Retrieval-Augmented Generation applied to YouTube transcripts — works for "
    "both Arabic (عربي) and English videos."
)

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Chunks to retrieve (k)", min_value=1, max_value=10, value=4)
    st.markdown("---")
    st.markdown(
        "**Architecture**\n\n"
        "1. `ingest.py` — pulls the transcript via `youtube-transcript-api` "
        "(tries Arabic captions, then English, then whatever's available)\n"
        "2. `chunking.py` — sentence-aware overlapping chunks (Arabic + Latin "
        "sentence boundaries)\n"
        "3. `embed_store.py` — multilingual `sentence-transformers` "
        "(`paraphrase-multilingual-MiniLM-L12-v2`) + FAISS cosine search\n"
        "4. `summarizer.py` — `csebuetnlp/mT5_multilingual_XLSum`, fine-tuned "
        "on 45 languages including Arabic and English"
    )

source = st.radio("Source", ["YouTube URL / ID", "Paste transcript text"], horizontal=True)

transcript_input = None
if source == "YouTube URL / ID":
    transcript_input = st.text_input(
        "YouTube video URL or ID", placeholder="https://www.youtube.com/watch?v=..."
    )
else:
    transcript_input = st.text_area("Paste transcript text", height=200)

question = st.text_input(
    "Optional question about the video (leave blank to summarize the whole video)"
)

run = st.button("▶ Run pipeline", type="primary")

if run:
    if not transcript_input or not transcript_input.strip():
        st.warning("Please provide a YouTube URL/ID or paste a transcript first.")
        st.stop()

    pipeline = get_pipeline()

    with st.status("Running RAG pipeline...", expanded=True) as status:
        try:
            st.write("📥 Ingesting source...")
            if source == "YouTube URL / ID":
                result = pipeline.run_from_youtube(
                    transcript_input, question=question or None, top_k=top_k
                )
            else:
                result = pipeline.run_from_text(
                    transcript_input, question=question or None, top_k=top_k
                )
            st.write(f"🧩 Chunked into **{len(result.chunks)}** segments")
            st.write(f"🎯 Retrieved top **{len(result.retrieved)}** by "
                      f"{'query relevance' if result.mode == 'qa' else 'salience'}")
            st.write("✨ Generated output")
            status.update(label="Done", state="complete")
        except Exception as e:
            status.update(label="Failed", state="error")
            st.error(f"Pipeline error: {e}")
            st.stop()

    lang_label = "🇸🇦 Arabic" if result.language == "ar" else "🇬🇧 English"
    st.subheader(f"📝 Result — detected language: {lang_label}")
    st.write(result.output)

    with st.expander("🎯 Retrieved chunks (the 'augment' context)"):
        for i, r in enumerate(result.retrieved, start=1):
            st.markdown(f"**Chunk {i}** — similarity score `{r.score:.3f}`")
            st.text(r.chunk.text)
            st.divider()

    with st.expander("📄 Full transcript"):
        st.text(result.transcript)
else:
    st.info("Paste a YouTube link or transcript, then click **Run pipeline**.")
