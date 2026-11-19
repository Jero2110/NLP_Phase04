import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.embeddings import Embedder

if __name__ == "__main__":
    cfg = load_config()
    reference = Embedder({**cfg["embeddings"], "backend": "local"})
    production = Embedder({**cfg["embeddings"], "backend": "fastembed"})

    text = ["¿Qué es una adenda en un proceso de contratación?"]
    a = reference.encode_queries(text)[0]
    b = production.encode_queries(text)[0]
    cos = float(a @ b)
    print(f"dimensiones local={a.shape[0]} fastembed={b.shape[0]} | similitud coseno = {cos:.4f}")
    assert a.shape == b.shape and cos > 0.98, "Los vectores no coinciden: revisa el modelo o el prefijo"
    print("OK: fastembed es compatible con la referencia.")