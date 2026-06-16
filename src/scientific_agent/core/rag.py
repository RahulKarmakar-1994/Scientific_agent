import json
import math
import re
from collections import Counter
from pathlib import Path


DEFAULT_INDEX_DIR = ".vector_store"
CHUNK_WORDS = 220
CHUNK_OVERLAP = 50
STOPWORDS = {
    "about",
    "also",
    "and",
    "are",
    "can",
    "code",
    "could",
    "demo",
    "demonstrate",
    "does",
    "explain",
    "from",
    "give",
    "how",
    "know",
    "learn",
    "like",
    "me",
    "please",
    "python",
    "show",
    "teach",
    "that",
    "the",
    "this",
    "using",
    "what",
    "when",
    "where",
    "with",
    "you",
}


class LocalRAG:
    """Lightweight local RAG index.

    This first version uses PDF/TXT extraction plus lexical scoring. It avoids
    API keys and model downloads while keeping the same ingest/search shape we
    can later back with embeddings.
    """

    def __init__(self, index_dir=DEFAULT_INDEX_DIR):
        self.index_dir = Path(index_dir)
        self.index_path = self.index_dir / "chunks.json"

    def ingest(self, paths):
        self.index_dir.mkdir(parents=True, exist_ok=True)
        documents = []
        for path in paths:
            documents.extend(_load_path(Path(path)))

        chunks = []
        for document in documents:
            chunks.extend(_chunk_document(document))

        payload = {
            "version": 1,
            "chunk_words": CHUNK_WORDS,
            "chunk_overlap": CHUNK_OVERLAP,
            "chunks": chunks,
        }
        self.index_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return {
            "index_path": str(self.index_path),
            "documents": len(documents),
            "chunks": len(chunks),
        }

    def search(self, query, top_k=5):
        if not self.index_path.exists():
            raise FileNotFoundError(
                f"RAG index not found: {self.index_path}. Run ingest first."
            )

        payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        query_terms = _terms(query)
        document_frequency = _document_frequency(payload["chunks"])
        results = []
        for chunk in payload["chunks"]:
            score = _score(query_terms, chunk["text"], document_frequency, len(payload["chunks"]))
            if score > 0:
                results.append(
                    {
                        "score": score,
                        "source": chunk["source"],
                        "page": chunk.get("page"),
                        "chunk_id": chunk["chunk_id"],
                        "text": chunk["text"],
                    }
                )

        return sorted(results, key=lambda item: item["score"], reverse=True)[:top_k]


def _load_path(path):
    if path.is_dir():
        documents = []
        for child in sorted(path.rglob("*")):
            if child.suffix.lower() in {".pdf", ".txt", ".md"}:
                documents.extend(_load_path(child))
        return documents

    if path.suffix.lower() == ".pdf":
        return _load_pdf(path)
    if path.suffix.lower() in {".txt", ".md"}:
        return [
            {
                "source": str(path),
                "page": None,
                "text": path.read_text(encoding="utf-8", errors="ignore"),
            }
        ]
    return []


def _load_pdf(path):
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    documents = []
    for page_index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            documents.append(
                {
                    "source": str(path),
                    "page": page_index,
                    "text": text,
                }
            )
    return documents


def _chunk_document(document):
    words = _normalize_space(document["text"]).split()
    chunks = []
    start = 0
    chunk_index = 0
    step = max(CHUNK_WORDS - CHUNK_OVERLAP, 1)
    while start < len(words):
        end = min(start + CHUNK_WORDS, len(words))
        chunk_text = " ".join(words[start:end])
        if chunk_text:
            chunks.append(
                {
                    "chunk_id": f"{Path(document['source']).name}:p{document.get('page')}:c{chunk_index}",
                    "source": document["source"],
                    "page": document.get("page"),
                    "text": chunk_text,
                }
            )
        if end == len(words):
            break
        start += step
        chunk_index += 1
    return chunks


def _document_frequency(chunks):
    frequency = Counter()
    for chunk in chunks:
        frequency.update(set(_terms(chunk["text"])))
    return frequency


def _score(query_terms, text, document_frequency, chunk_count):
    if not query_terms:
        return 0

    text_terms = Counter(_terms(text))
    length_norm = math.sqrt(sum(count * count for count in text_terms.values())) or 1.0
    score = 0.0
    for term in set(query_terms):
        inverse_document_frequency = math.log(
            (1 + chunk_count) / (1 + document_frequency.get(term, 0))
        ) + 1.0
        score += text_terms.get(term, 0) * inverse_document_frequency
    return score / length_norm


def _terms(text):
    return [
        term
        for term in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", text.lower())
        if len(term) > 2 and term not in STOPWORDS
    ]


def _normalize_space(text):
    return re.sub(r"\s+", " ", text).strip()
