import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.knowledge_base import KnowledgeBase

if __name__ == "__main__":
    cfg = load_config()
    kb = KnowledgeBase(cfg["knowledge"]["index_dir"], cfg["embeddings"])
    n = kb.build(
        cfg["knowledge"]["documents_dir"],
        cfg["knowledge"]["manifest"],
        cfg["knowledge"]["chunk_chars"],
        cfg["knowledge"]["overlap_chars"],
    )
    print(f"Índice construido con {n} fragmentos en {cfg['knowledge']['index_dir']}")