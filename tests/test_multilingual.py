from src.ingest import detect_language, load_pasted_text
from src.chunking import chunk_transcript, split_sentences

ARABIC_SAMPLE = (
    "مرحبا بكم في هذا الفيديو. سنتحدث اليوم عن الذكاء الاصطناعي وتطبيقاته المختلفة. "
    "هذا موضوع مهم جدا."
)
ENGLISH_SAMPLE = (
    "Welcome to this video. Today we will talk about artificial intelligence "
    "and its applications. This is a very important topic."
)


def test_detect_language_arabic():
    assert detect_language(ARABIC_SAMPLE) == "ar"


def test_detect_language_english():
    assert detect_language(ENGLISH_SAMPLE) == "en"


def test_arabic_sentence_splitting():
    sentences = split_sentences(ARABIC_SAMPLE)
    assert len(sentences) == 3
    assert sentences[0].endswith("الفيديو.")


def test_arabic_chunking_produces_multiple_chunks():
    long_text = (ARABIC_SAMPLE + " ") * 8
    chunks = chunk_transcript(long_text, max_words=15, overlap_words=3)
    assert len(chunks) > 1
    for c in chunks:
        assert c.text.strip()


def test_load_pasted_text_normalizes_whitespace():
    messy = "  hello   world  \n\n  test  "
    assert load_pasted_text(messy) == "hello world test"
