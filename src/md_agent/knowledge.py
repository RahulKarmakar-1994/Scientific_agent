from pathlib import Path


class KnowledgeAgent:
    def __init__(self, knowledgebase_dir="knowledgebase"):
        self.knowledgebase_dir = Path(knowledgebase_dir)

    def search(self, query, limit=3):
        query_terms = _terms(query)
        matches = []

        for path in self.knowledgebase_dir.rglob("*.txt"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            score = sum(text.lower().count(term) for term in query_terms)
            if score > 0:
                matches.append(
                    {
                        "source": str(path),
                        "score": score,
                        "excerpt": _best_excerpt(text, query_terms),
                    }
                )

        return sorted(matches, key=lambda item: item["score"], reverse=True)[:limit]


def _terms(query):
    return [term for term in query.lower().split() if len(term) > 3]


def _best_excerpt(text, terms, window=260):
    lower_text = text.lower()
    positions = [lower_text.find(term) for term in terms if lower_text.find(term) >= 0]
    if not positions:
        return text[:window].strip()

    start = max(min(positions) - 80, 0)
    excerpt = text[start : start + window].replace("\n", " ").strip()
    return excerpt
