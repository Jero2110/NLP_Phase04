import time

import numpy as np

from src.config import get_secret


class Embedder:
    """Misma interfaz para dos backends:
    - local:  sentence-transformers en tu equipo (solo para construir el índice).
    - hf_api: Inference API de Hugging Face (para la app desplegada, sin PyTorch).
    Ambos devuelven vectores float32 normalizados (similitud coseno = producto interno)."""

    def __init__(self, cfg: dict):
        self.backend = cfg.get("backend", "hf_api")
        self.model_id = cfg["model_id"]
        self.query_prefix = cfg.get("query_prefix", "")
        self.passage_prefix = cfg.get("passage_prefix", "")
        self.batch_size = cfg.get("batch_size", 16)
        self._st = None
        self._client = None

        if self.backend == "local":
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
        if self.backend == "local":
            vectors = self._st.encode(
                texts, batch_size=self.batch_size, normalize_embeddings=True, convert_to_numpy=True
            )
            return vectors.astype("float32")
        return np.vstack([self._api_embed(t) for t in texts]).astype("float32")

    def _api_embed(self, text: str, retries: int = 3) -> np.ndarray:
        for attempt in range(retries):
            try:
                out = np.asarray(self._client.feature_extraction(text, model=self.model_id))
                # Según el modelo, la API puede devolver (tokens, dims) o (1, tokens, dims).
                # Hacemos mean pooling, igual que el modelo original de e5.
                while out.ndim > 1:
                    out = out.mean(axis=0)
                return out / np.linalg.norm(out)
            except Exception:
                if attempt == retries - 1:
                    raise
                time.sleep(2 * (attempt + 1))  # backoff ante límites de uso o carga del modelo