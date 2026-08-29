from src.chunking import chunk_transcript, split_sentences


def test_split_sentences_basic():
    text = "Hello world. This is a test! Is it working?"
    sentences = split_sentences(text)
    assert sentences == ["Hello world.", "This is a test!", "Is it working?"]


def test_chunk_transcript_respects_max_words():
    text = ("This is sentence one. This is sentence two. " * 50).strip()
    chunks = chunk_transcript(text, max_words=30, overlap_words=5)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text.split()) <= 40  # allows small slack from sentence packing


def test_chunk_transcript_overlap_present():
    text = ("Alpha beta gamma delta epsilon zeta eta theta. " * 20).strip()
    chunks = chunk_transcript(text, max_words=20, overlap_words=5)
    assert len(chunks) >= 2
    first_tail = chunks[0].text.split()[-5:]
    second_head = chunks[1].text.split()[:5]
    assert set(first_tail) & set(second_head)


def test_empty_text_returns_no_chunks():
    assert chunk_transcript("   ") == []
