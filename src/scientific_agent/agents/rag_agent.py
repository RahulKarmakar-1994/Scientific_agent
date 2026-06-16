from src.scientific_agent.core.llm import LLMClient
from src.scientific_agent.core.rag import LocalRAG


class RAGAgent:
    """Answer questions from the local document index."""

    def __init__(self, provider="ollama", model=None, index_dir=".vector_store"):
        self.rag = LocalRAG(index_dir=index_dir)
        self.llm = LLMClient(provider=provider, model=model)

    def run(self, request, top_k=5):
        hits = self.rag.search(request, top_k=top_k)
        answer = None
        if self.llm.available:
            answer = self.llm.respond(_answer_prompt(request, hits))
        return {
            "agent": "rag",
            "status": "completed",
            "request": request,
            "answer": answer,
            "evidence": hits,
            "llm": {
                "enabled": self.llm.available,
                "provider": self.llm.provider,
                "model": self.llm.model,
                "error": self.llm.error,
            },
        }


def _answer_prompt(request, hits):
    return f"""
Answer using only this retrieved evidence. If the evidence is insufficient,
say what is missing. Mention source/page/chunk when useful.

Question:
{request}

Evidence:
{hits}
""".strip()
