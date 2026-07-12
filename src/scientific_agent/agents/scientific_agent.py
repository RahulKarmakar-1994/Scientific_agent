from .learning_demo_agent import LearningDemoAgent
from .rag_agent import RAGAgent
from .router_agent import RouterAgent
from .simulation_agent import SimulationAgent
from .verifier_agent import PhysicsVerifierAgent
from src.scientific_agent.core.job_store import JobStore
from src.scientific_agent.core.session_store import SessionStore


class ScientificAgent:
    """Top-level product agent for scientific learning and execution."""

    def __init__(self, provider="ollama", model=None, engine="local", index_dir=".vector_store"):
        self.provider = provider
        self.model = model
        self.engine = engine
        self.job_store = JobStore()
        self.session_store = SessionStore()
        self.router = RouterAgent()
        self.learning_agent = LearningDemoAgent(
            provider=provider,
            model=model,
            index_dir=index_dir,
        )
        self.simulation_agent = SimulationAgent(engine=engine)
        self.rag_agent = RAGAgent(provider=provider, model=model, index_dir=index_dir)
        self.verifier_agent = PhysicsVerifierAgent(provider=provider, model=model)

    def run(self, request, session_id=None, learner_prediction=None):
        conversation_context = ""
        pending_demo = None
        original_user_request = request
        if session_id:
            conversation_context = self.session_store.context_text(session_id)
            self.session_store.append_message(session_id, "user", request)
            pending_demo = self.session_store.read_state(session_id, "pending_demo")

        if pending_demo:
            if _is_cancel_request(request):
                self.session_store.clear_state(session_id, "pending_demo")
                return self._record_simple_response(
                    request,
                    session_id,
                    "Cancelled the pending demo. You can ask a new question whenever you are ready.",
                    status="cancelled",
                )
            learner_prediction = learner_prediction or _prediction_from_reply(
                request,
                pending_demo.get("prediction") or {},
            )
            request = pending_demo.get("request") or request
            self.session_store.clear_state(session_id, "pending_demo")

        job_id, job_dir = self.job_store.create(request)
        if session_id:
            self.job_store.write_json(
                job_dir,
                "session.json",
                {
                    "session_id": session_id,
                    "conversation_context": conversation_context,
                    "original_user_request": original_user_request,
                    "pending_demo_resumed": bool(pending_demo),
                },
            )

        routing_request = _with_context(request, conversation_context)
        route = self.router.route(routing_request)
        self.job_store.write_json(job_dir, "route.json", route)

        if (
            session_id
            and not pending_demo
            and not learner_prediction
            and route["route"] == "learning_demo"
            and _should_pause_for_prediction(request)
        ):
            result = self.learning_agent.prepare_prediction(
                request,
                conversation_context=conversation_context,
            )
            if result.get("status") == "awaiting_prediction":
                self.session_store.write_state(
                    session_id,
                    "pending_demo",
                    {
                        "request": request,
                        "route": route,
                        "prediction": result.get("prediction") or {},
                        "simulation_spec": result.get("simulation_spec") or {},
                    },
                )
        elif route["route"] == "simulation":
            result = self.simulation_agent.run(request)
        elif route["route"] == "rag":
            result = self.rag_agent.run(request)
        else:
            result = self.learning_agent.run(
                request,
                conversation_context=conversation_context,
                learner_prediction=learner_prediction,
                prepared_simulation_spec=(pending_demo or {}).get("simulation_spec"),
            )

        if route["route"] in {"learning_demo", "rag"} and result.get("status") != "awaiting_prediction":
            verification = self.verifier_agent.run(request, result)
            result["verification"] = verification
            result["final_answer"] = _append_verification_note(
                result.get("final_answer") or result.get("answer") or "",
                verification,
            )

        copied_files = self.job_store.copy_files(job_dir, _generated_files(result))
        if copied_files:
            result.setdefault("job_files", {})["artifacts"] = copied_files
        if result.get("generated_code"):
            generated_code_path = self.job_store.write_text(
                job_dir,
                "generated_code.py",
                result["generated_code"],
            )
            result.setdefault("job_files", {})["generated_code"] = generated_code_path

        self.job_store.write_json(job_dir, "result.json", result)
        self.job_store.write_text(job_dir, "report.md", _report_text(result))
        report = {
            "job_id": job_id,
            "job_dir": str(job_dir),
            "session_id": session_id,
            "request": request,
            "route": route,
            "provider": self.provider,
            "model": self.model,
            "engine": self.engine,
            "result": result,
        }
        if session_id:
            self.session_store.append_message(
                session_id,
                "assistant",
                result.get("final_answer") or result.get("answer") or "",
                metadata={
                    "job_id": job_id,
                    "job_dir": str(job_dir),
                    "route": route,
                    "status": result.get("status"),
                },
            )
        return report

    def _record_simple_response(self, request, session_id, answer, status="answered"):
        job_id, job_dir = self.job_store.create(request)
        route = {"route": "learning_demo", "reason": "session control response"}
        result = {
            "agent": "scientific_agent",
            "status": status,
            "request": request,
            "final_answer": answer,
        }
        self.job_store.write_json(job_dir, "route.json", route)
        self.job_store.write_json(job_dir, "result.json", result)
        self.job_store.write_text(job_dir, "report.md", answer)
        if session_id:
            self.session_store.append_message(
                session_id,
                "assistant",
                answer,
                metadata={
                    "job_id": job_id,
                    "job_dir": str(job_dir),
                    "route": route,
                    "status": status,
                },
            )
        return {
            "job_id": job_id,
            "job_dir": str(job_dir),
            "session_id": session_id,
            "request": request,
            "route": route,
            "provider": self.provider,
            "model": self.model,
            "engine": self.engine,
            "result": result,
        }


def _generated_files(result):
    files = []
    demo_result = result.get("result")
    if isinstance(demo_result, dict):
        files.extend(demo_result.get("generated_files") or [])
    files.extend(result.get("generated_files") or [])
    return files


def _report_text(result):
    text = result.get("final_answer") or result.get("answer") or ""
    artifact_paths = (result.get("job_files") or {}).get("artifacts") or []
    if artifact_paths:
        text += "\n\nJob-local artifacts:\n"
        text += "\n".join(f"- {path}" for path in artifact_paths)
    generated_code_path = (result.get("job_files") or {}).get("generated_code")
    if generated_code_path:
        text += "\n\nGenerated code:\n"
        text += f"- {generated_code_path}"
    return text


def _with_context(request, conversation_context):
    if not conversation_context:
        return request
    return (
        "Conversation context:\n"
        f"{conversation_context}\n\n"
        "Current user request:\n"
        f"{request}"
    )


def _append_verification_note(answer, verification):
    if not verification:
        return answer
    note = verification.get("suggested_note")
    verdict = verification.get("verdict")
    confidence = verification.get("confidence")
    if not note:
        note = f"Verifier verdict: {verdict}, confidence: {confidence}."
    return f"{answer.rstrip()}\n\nVerification: {verdict} ({confidence})\n{note}"


def _should_pause_for_prediction(request):
    lower_request = str(request or "").lower()
    return any(
        marker in lower_request
        for marker in [
            "demo",
            "demonstrate",
            "plot",
            "python",
            "simulate",
            "simulation",
        ]
    )


def _prediction_from_reply(reply, prediction):
    text = str(reply or "").strip()
    normalized = text.rstrip(".:").strip().upper()
    if len(normalized) == 1:
        for option in prediction.get("options") or []:
            if str(option.get("id") or "").upper() == normalized:
                return option.get("text") or text
    return text


def _is_cancel_request(request):
    return str(request or "").strip().lower() in {
        "cancel",
        "stop",
        "skip",
        "never mind",
        "nevermind",
    }
