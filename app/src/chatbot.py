import time
import uuid
from dataclasses import dataclass, field

from src.config import get_secret, load_config
from src.knowledge_base import KnowledgeBase
from src.llm import LLMClient
from src.logging_analytics import EventLogger
from src.memory import ShortTermMemory
from src.preprocessing import classify_topics, preprocess_user_input
from src.profiles import ProfileStore
from src.prompts import (
    NO_CONTEXT_MESSAGE, SYSTEM_PROMPT, build_messages, check_citations, format_extractive_answer,
)
from src.secop import SecopClient, detect_intent

ERROR_MESSAGE = "Ocurrió un problema al generar la respuesta. Intenta de nuevo en unos minutos."
EMPTY_MESSAGE = "Escribe una pregunta para poder ayudarte."


@dataclass
class Session:
    memory: ShortTermMemory
    user_id: str | None = None
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    turn: int = 0


@dataclass
class ChatResponse:
    answer: str
    status: str                       # answered | no_context | blocked | error
    sources: list[dict] = field(default_factory=list)
    latency_ms: float = 0.0


class Chatbot:
    def __init__(self, cfg, kb, llm, profiles, logger, secop=None):
        self.cfg = cfg
        self.kb = kb
        self.llm = llm
        self.profiles = profiles
        self.logger = logger
        self.secop = secop
        self.retrieval = cfg["retrieval"]
        self.memory_cfg = cfg["memory"]
        self.safety = cfg["safety"]
        self.store_text = cfg["logging"].get("store_query_text", False)

    def new_session(self, user_id: str | None = None) -> Session:
        return Session(
            memory=ShortTermMemory(self.memory_cfg["max_turns"], self.memory_cfg["max_chars"]),
            user_id=user_id,
        )

    def _process_chunks(self, text: str) -> tuple[list[dict], str | None]:
        """Devuelve (fragmentos, tipo_de_error). Lista vacía si no aplica o si falla."""
        if self.secop is None:
            return [], None
        intent, ref = detect_intent(text)
        if intent == "docs":
            return [], None
        try:
            if intent == "process":
                rows = self.secop.get_by_reference(ref)
            else:
                rows = self.secop.search(text)
        except Exception as exc:
            return [], type(exc).__name__
        return [self.secop.to_chunk(r) for r in rows], None

    def answer(self, session: Session, raw_text: str) -> ChatResponse:
        t0 = time.perf_counter()
        session.turn += 1
        cleaned = preprocess_user_input(raw_text, self.safety["max_input_chars"])

        if not cleaned.text:
            return self._finish(session, cleaned, EMPTY_MESSAGE, "blocked", [], t0)

        # 1. Procesos de SECOP, solo si la consulta lo pide (no reemplaza a los documentos)
        secop_chunks, secop_error = self._process_chunks(cleaned.text)

        # 2. Documentos: se consultan siempre
        doc_chunks = self.kb.search(cleaned.text, self.retrieval["top_k"], self.retrieval["min_score"])

        # 3. Unión: procesos primero, luego documentos, con límite total de fuentes
        max_sources = self.retrieval["top_k"] + 2
        chunks = (secop_chunks + doc_chunks)[:max_sources]

        sources: list[dict] = []
        invalid: list[int] = []
        error = None

        if not chunks:
            answer, status = NO_CONTEXT_MESSAGE, "no_context"
        else:
            sources = [
                {"id": f"F{i}", "source": c["source"], "page": c.get("page"),
                 "chunk_id": c["chunk_id"], "score": c.get("score"),
                 "origin": c.get("origin", "documento")}
                for i, c in enumerate(chunks, 1)
            ]
            try:
                if self.llm is None:  # modo extractivo: sin LLM
                    raw = format_extractive_answer(chunks)
                else:
                    hint = self.profiles.hint(session.user_id) if session.user_id else ""
                    messages = build_messages(
                        SYSTEM_PROMPT + hint, session.memory.as_messages(), cleaned.text, chunks
                    )
                    raw = self.llm.chat(messages)
                answer, invalid = check_citations(raw, len(chunks))
                status = "answered"
            except Exception as exc:  # se registra solo el tipo de error
                answer, status, error = ERROR_MESSAGE, "error", type(exc).__name__

        if status in ("answered", "no_context"):
            session.memory.add_exchange(cleaned.text, answer)
        if status == "answered" and session.user_id:
            self.profiles.record_topics(session.user_id, classify_topics(cleaned.text))

        return self._finish(
            session, cleaned, answer, status, sources, t0,
            invalid=invalid, error=error, secop_error=secop_error,
        )

    def _finish(self, session, cleaned, answer, status, sources, t0,
                invalid=None, error=None, secop_error=None) -> ChatResponse:
        latency = round((time.perf_counter() - t0) * 1000, 1)
        event = {
            "session_id": session.session_id,  # aleatorio; nunca se registra el user_id
            "turn": session.turn,
            "status": status,
            "latency_ms": latency,
            "topics": classify_topics(cleaned.text),
            "redactions": cleaned.redactions,
            "suspicious": cleaned.suspicious,
            "truncated": cleaned.truncated,
            "retrieved": [(s["chunk_id"], s["score"], s["origin"]) for s in sources],
            "origin_counts": {
                o: sum(1 for s in sources if s["origin"] == o)
                for o in {s["origin"] for s in sources}
            },
            "invalid_citations": invalid or [],
            "error": error,
            "secop_error": secop_error,
        }
        if self.store_text:
            event["query_text"] = cleaned.text[:300]
        self.logger.log(event)
        return ChatResponse(answer, status, sources, latency)

def create_chatbot(cfg: dict | None = None) -> Chatbot:
    cfg = cfg or load_config()
    kb = KnowledgeBase(cfg["knowledge"]["index_dir"], cfg["embeddings"])
    llm = None if cfg["llm"]["backend"] == "extractive" else LLMClient(cfg["llm"])
    profiles = ProfileStore(cfg["profiles"]["db_path"], get_secret("PROFILE_SALT", required=True))
    logger = EventLogger(cfg["logging"]["log_path"])

    secop = None
    if cfg["datos_abiertos"]["enabled"]:
        secop = SecopClient(cfg["datos_abiertos"])

    return Chatbot(cfg, kb, llm, profiles, logger, secop=secop)