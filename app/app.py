import sys
from pathlib import Path

import streamlit as st

import os

try:
    for key in ("HF_TOKEN", "PROFILE_SALT"):
        if key in st.secrets and not os.getenv(key):
            os.environ[key] = st.secrets[key]
except Exception:
    pass  # sin archivo de secretos (desarrollo local con .env)


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.chatbot import create_chatbot  # noqa: E402
from src.config import load_config      # noqa: E402
from src.logging_analytics import EventLogger  # noqa: E402

# Must be the first Streamlit call in the script.
st.set_page_config(page_title="Asistente de contratación pública", page_icon="📑", layout="centered")


@st.cache_resource(show_spinner="Preparando el asistente. La primera vez puede tardar varios minutos…")
def get_resources():
    """Loaded once per server process and shared by all users. Conversation state is per session."""
    cfg = load_config()
    bot = create_chatbot(cfg)
    bot.kb.load()  # fails early if the index is missing
    feedback_logger = EventLogger(cfg["logging"]["feedback_path"])
    return cfg, bot, feedback_logger


try:
    cfg, bot, feedback_logger = get_resources()
except Exception as exc:
    st.error(f"No se pudo iniciar el asistente: {exc}")
    st.stop()

ui = cfg["ui"]
st.title(ui["app_title"])
st.info(ui["disclaimer"])


def new_conversation():
    st.session_state.messages = []
    st.session_state.session = bot.new_session(user_id=None)  # anonymous: no profile is stored


if "session" not in st.session_state:
    new_conversation()

# ---------------- Sidebar: guidance, examples, feedback ----------------
with st.sidebar:
    st.header("Sobre este asistente")
    st.markdown("**Puede:**")
    for item in ui["capabilities"]:
        st.markdown(f"- {item}")
    st.markdown("**No puede:**")
    for item in ui["limitations"]:
        st.markdown(f"- {item}")

    st.subheader("Preguntas de ejemplo")
    for i, example in enumerate(ui["example_prompts"]):
        if st.button(example, key=f"example_{i}"):
            st.session_state["pending"] = example

    st.button("Nueva conversación", on_click=new_conversation)

    st.subheader("Comentarios")
    comment = st.text_area(
        "¿Algo que quieras reportar?",
        key="comment",
        help="No escribas datos personales.",
    )
    if st.button("Enviar comentario") and comment.strip():
        feedback_logger.log({
            "type": "comment",
            "session_id": st.session_state.session.session_id,
            "text": comment.strip()[:1000],
        })
        st.success("Gracias por tu comentario.")

# ---------------- Chat ----------------
prompt = st.chat_input("Escribe tu pregunta sobre contratación pública…") or st.session_state.pop("pending", None)

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.spinner("Buscando en las fuentes…"):
        resp = bot.answer(st.session_state.session, prompt)
    st.session_state.messages.append({
        "role": "assistant",
        "content": resp.answer,
        "sources": resp.sources,
        "status": resp.status,
        "turn": st.session_state.session.turn,
    })

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):  # distinct avatars and alignment for user and assistant
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            if ui["show_sources"] and msg.get("sources"):
                with st.expander(f"Fuentes ({len(msg['sources'])})"):
                    for s in msg["sources"]:
                        st.markdown(
                            f"**[{s['id']}]** {s['source']} · página {s['page'] or '-'} · "
                            f"similitud {s['score']}"
                        )
            rating = st.feedback("thumbs", key=f"fb_{i}")
            if rating is not None and rating != msg.get("feedback"):
                msg["feedback"] = rating
                feedback_logger.log({
                    "type": "rating",
                    "session_id": st.session_state.session.session_id,
                    "turn": msg["turn"],
                    "status": msg["status"],
                    "score": "up" if rating == 1 else "down",
                })