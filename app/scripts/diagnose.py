import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.knowledge_base import KnowledgeBase
from src.llm import LLMClient

cfg = load_config()

# 1. Búsqueda: sin filtro de similitud, para ver los puntajes reales
kb = KnowledgeBase(cfg["knowledge"]["index_dir"], cfg["embeddings"])
kb.load()
print("Fragmentos en el índice:", len(kb._chunks))

for q in [
    "¿Qué es una adenda en un proceso de contratación?",
    "¿Qué es la licitación pública?",
]:
    print(f"\nPregunta: {q}")
    for h in kb.search(q, top_k=4, min_score=-1.0):
        print(f"  {h['score']:.3f}  {h['chunk_id']}  {h['text'][:70]}")

# 2. LLM: una llamada mínima para ver el error real
llm = LLMClient(cfg["llm"])
try:
    print("\nRespuesta del LLM:", llm.chat([{"role": "user", "content": "Responde solo: OK"}]))
except Exception as exc:
    print("\nERROR del LLM:", type(exc).__name__, str(exc)[:500])