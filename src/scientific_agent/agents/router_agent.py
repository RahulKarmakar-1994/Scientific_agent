class RouterAgent:
    """Route natural-language requests to task-specialized agents."""

    def route(self, request):
        lower_request = request.lower()
        if _contains_any(lower_request, LEARNING_MARKERS):
            return {
                "route": "learning_demo",
                "reason": "request asks to learn, explain, demonstrate, plot, or use Python",
            }
        if _contains_any(lower_request, SIMULATION_MARKERS):
            return {
                "route": "simulation",
                "reason": "request asks to run or analyze an MD/simulation workflow",
            }
        if _contains_any(lower_request, RAG_MARKERS):
            return {
                "route": "rag",
                "reason": "request asks about papers, documents, or references",
            }
        return {
            "route": "learning_demo",
            "reason": "default route for exploratory scientific questions",
        }


LEARNING_MARKERS = [
    "teach",
    "learn",
    "explain",
    "demo",
    "demonstrate",
    "python",
    "plot",
    "show me",
]

SIMULATION_MARKERS = [
    "md",
    "molecular dynamics",
    "nvt",
    "nve",
    "npt",
    "gromacs",
    "lammps",
    "run simulation",
    "simulate",
]

RAG_MARKERS = [
    "paper",
    "papers",
    "pdf",
    "reference",
    "from my documents",
    "from my papers",
]


def _contains_any(text, markers):
    return any(marker in text for marker in markers)
