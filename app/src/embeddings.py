import time

import numpy as np

from src.config import get_secret


class Embedder:
    """Misma interfaz para tres backends:
    - fastembed: ONNX en el propio proceso (gratuito, ligero). Recomendado para web y para construir el índice.
    - local:     sentence-transformers con PyTorch (solo para comparar en tu PC).
    - hf_api:    Inference API de Hugging Face (requiere que el modelo tenga tarea feature-extraction).
    Todos devuelven vectores float32 normalizados, así que la similitud coseno es el producto interno."""

    def __init__(self, cfg: dict):
        self.backend = cfg.get("backend", "fastembed")
        self.model_id = cfg["model_id"]
        self.query_prefix = cfg.get("query_prefix", "")
        self.passage_prefix = cfg.get("passage_prefix", "")
        self.batch_size = cfg.get("batch_size", 16)
        self._st = None
        self._fe = None
        self._client = None

        if self.backend == "fastembed":
            from fastembed import TextEmbedding
            self._fe = TextEmbedding(model_name=self.model_id)
        elif self.backend == "local":
            from sentence_transformers import SentenceTransformer
            self._st = SentenceTransformer(self.model_id)
        elif self.backend == "hf_api":
            from huggingface_hub import InferenceClient
            self._client = InferenceClient(token=get_secret("HF_TOKEN", required=True))
        else:
            raise ValueError(f"Backend de embeddings desconocido: {self.backend}")

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return self._encode([self.passage_prefix + t for t in texts])

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode([self.query_prefix + t for t in texts])

    def _encode(self, texts: list[str]) -> np.ndarray:
        if self.backend == "fastembed":
            vectors = np.vstack(list(self._fe.embed(texts, batch_size=self.batch_size)))
            return _normalize(vectors)
        if self.backend == "local":
            vectors = self._st.encode(texts, batch_size=self.batch_size, convert_to_numpy=True)
            return _normalize(vectors)
        return np.vstack([self._api_embed(t) for t in texts]).astype("float32")

    def _api_embed(self, text: str, retries: int = 3) -> np.ndarray:
        for attempt in range(retries):
            try:
                out = np.asarray(self._client.feature_extraction(text, model=self.model_id))
                while out.ndim > 1:
                    out = out.mean(axis=0)
                return _normalize(out[None, :])[0]
            except Exception:
                if attempt == retries - 1:
                    raise
                time.sleep(2 * (attempt + 1))


def _normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype="float32")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)