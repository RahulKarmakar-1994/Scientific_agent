from src.scientific_agent.core.rag import LocalRAG


class GroundingService:
    """Retrieve local source evidence for scientific answers."""

    def __init__(self, index_dir=".vector_store"):
        self.rag = LocalRAG(index_dir=index_dir)

    def retrieve(self, request, conversation_context="", search_queries=None):
        search_queries = list(search_queries or [])
        if not search_queries:
            search_queries.insert(0, request)
        search_queries = _dedupe_search_queries(search_queries)

        try:
            hits = _merge_hits(
                self.rag.search(query, top_k=6)
                for query in search_queries
                if query and query.strip()
            )
        except FileNotFoundError:
            return {
                "status": "missing",
                "reason": "No local RAG index found. Run `python -m src.scientific_agent.rag ingest ...` first.",
                "sources": [],
                "search_queries": search_queries,
            }

        sources = _compact_evidence(hits)
        if not sources:
            return {
                "status": "missing",
                "reason": "No relevant local source chunks were found.",
                "sources": [],
                "search_queries": search_queries,
            }

        strong_source_count = sum(1 for source in sources if source["score"] >= 1.5)
        status = "grounded" if strong_source_count >= 2 else "weak"
        reason = (
            "Relevant local source chunks were found."
            if status == "grounded"
            else "Only weak or partial local source evidence was found."
        )
        return {
            "status": status,
            "reason": reason,
            "sources": sources,
            "search_queries": search_queries,
        }


def _compact_evidence(hits):
    compact = []
    for hit in hits:
        compact.append(
            {
                "score": round(float(hit["score"]), 4),
                "source": hit["source"],
                "page": hit.get("page"),
                "chunk_id": hit["chunk_id"],
                "text": hit["text"][:900],
            }
        )
    return compact


def _merge_hits(hit_groups):
    merged = {}
    for hits in hit_groups:
        for hit in hits:
            chunk_id = hit["chunk_id"]
            current = merged.get(chunk_id)
            if current is None or hit["score"] > current["score"]:
                merged[chunk_id] = hit
    return sorted(merged.values(), key=lambda item: item["score"], reverse=True)[:6]


def _dedupe_search_queries(search_queries):
    seen = set()
    deduped = []
    for query in search_queries:
        normalized = " ".join(str(query).split())
        key = normalized.lower()
        if normalized and key not in seen:
            seen.add(key)
            deduped.append(normalized)
    return deduped
