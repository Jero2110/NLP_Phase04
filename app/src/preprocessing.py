import re
import unicodedata
from dataclasses import dataclass, field

CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
PHONE = re.compile(r"(?<!\d)(?:\+?57\s?)?3\d{2}[\s-]?\d{3}[\s-]?\d{4}(?!\d)")
# 7–10 digit IDs (cédula). The lookbehind excludes digits after "." so process codes such as
# CO1.PCCNTR.1234567 are kept intact. NIT with dots (900.123.456-7) is NOT covered: known limitation.
ID_NUMBER = re.compile(r"(?<![\w.])\d{7,10}(?:-\d)?(?![\w.])")

INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"ignora(r)? (las |todas las )?instrucciones",
        r"ignore (all |previous |the )?instructions",
        r"(muestra|revela|imprime) (tu|el) (prompt|mensaje de sistema|system prompt)",
        r"reveal (your )?(system )?prompt",
        r"ahora eres",
        r"you are now",
    ]
]

TOPIC_KEYWORDS = {
    "modalidad": ["licitacion", "seleccion abreviada", "minima cuantia", "concurso de meritos", "contratacion directa"],
    "cronograma": ["cronograma", "plazo", "fecha", "cierre", "apertura"],
    "requisitos": ["requisito", "experiencia", "habilitante", "capacidad financiera"],
    "garantias": ["garantia", "poliza"],
    "sanciones": ["multa", "sancion", "caducidad", "incumplimiento"],
    "glosario": ["que significa", "definicion", "adenda"],
    "busqueda_procesos": ["buscar", "encontrar", "procesos abiertos", "oportunidades"],
}


@dataclass
class CleanedInput:
    text: str
    redactions: list[str] = field(default_factory=list)
    suspicious: bool = False
    truncated: bool = False


def _fold(text: str) -> str:
    """Lowercases and strips accents, for keyword matching."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = CONTROL_CHARS.sub("", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def redact_pii(text: str) -> tuple[str, list[str]]:
    found: list[str] = []
    for pattern, label in [(EMAIL, "CORREO"), (PHONE, "TELEFONO"), (ID_NUMBER, "NUMERO_ID")]:
        if pattern.search(text):
            found.append(label)
            text = pattern.sub(f"[{label}]", text)
    return text, found


def preprocess_user_input(raw: str, max_chars: int = 1500) -> CleanedInput:
    text = normalize_text(raw or "")
    truncated = len(text) > max_chars
    text = text[:max_chars]
    text, redactions = redact_pii(text)
    suspicious = any(p.search(text) for p in INJECTION_PATTERNS)
    return CleanedInput(text, redactions, suspicious, truncated)


def classify_topics(text: str) -> list[str]:
    folded = _fold(text)
    return [topic for topic, words in TOPIC_KEYWORDS.items() if any(w in folded for w in words)]