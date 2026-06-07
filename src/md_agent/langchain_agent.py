import json
import os

from .analysis import analyze_md_result
from .knowledge import KnowledgeAgent
from .parser import parse_md_request
from .runner import MDRunner


SYSTEM_PROMPT = """
You are an MD research agent for molecular dynamics workflows.

Available tools can:
- parse an MD request into a structured config
- search local MD papers and force-field notes
- run a local or Docker MD calculation
- analyze MD results

Use tools when the user asks for a simulation, setup, analysis, or evidence.
For execution requests, follow this order:
1. parse_md_request
2. search_md_knowledge
3. run_md_simulation
4. analyze_md_output
5. summarize clearly

Be explicit when the calculation is a toy placeholder rather than physical MD.
Do not invent results that are not returned by a tool.
""".strip()


def run_langchain_tool_agent(query, model=None, engine="local"):
    try:
        from langchain.agents import create_agent
        from langchain.tools import tool
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as exc:
        return {
            "mode": "tool-agent",
            "status": "failed",
            "error": f"Missing LangChain dependency: {exc}",
            "hint": "Install requirements.txt in the ai-agent conda environment.",
        }

    if not os.getenv("GOOGLE_API_KEY") and not os.getenv("GEMINI_API_KEY"):
        return {
            "mode": "tool-agent",
            "status": "failed",
            "error": "GOOGLE_API_KEY or GEMINI_API_KEY is not set.",
        }

    knowledge_agent = KnowledgeAgent()
    runner = MDRunner()

    @tool
    def parse_md_request_tool(user_query: str) -> str:
        """Parse a natural-language molecular dynamics request into JSON config."""
        return json.dumps(parse_md_request(user_query), indent=2)

    @tool
    def search_md_knowledge_tool(user_query: str) -> str:
        """Search local MD papers and force-field notes for relevant evidence."""
        return json.dumps(knowledge_agent.search(user_query), indent=2)

    @tool
    def run_md_simulation_tool(config_json: str) -> str:
        """Run an MD calculation from config JSON using the selected engine."""
        config = json.loads(config_json)
        result = runner.run(config, engine=engine)
        return json.dumps(result, indent=2)

    @tool
    def analyze_md_output_tool(result_json: str) -> str:
        """Analyze MD result JSON and report stability, energy, and temperature."""
        result = json.loads(result_json)
        return json.dumps(analyze_md_result(result), indent=2)

    llm = ChatGoogleGenerativeAI(
        model=model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        temperature=0.0,
    )
    agent = create_agent(
        model=llm,
        tools=[
            parse_md_request_tool,
            search_md_knowledge_tool,
            run_md_simulation_tool,
            analyze_md_output_tool,
        ],
        system_prompt=SYSTEM_PROMPT,
    )

    try:
        response = agent.invoke({"messages": [{"role": "user", "content": query}]})
    except Exception as exc:
        return {
            "mode": "tool-agent",
            "status": "failed",
            "model": model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            "engine": engine,
            "error": str(exc),
        }

    return {
        "mode": "tool-agent",
        "status": "completed",
        "model": model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        "engine": engine,
        "response": _serialize_response(response),
    }


def _serialize_response(response):
    try:
        return json.loads(json.dumps(response, default=_message_to_dict))
    except TypeError:
        return str(response)


def _message_to_dict(value):
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return str(value)
