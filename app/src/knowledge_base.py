import json
import re
from pathlib import Path

import faiss
import numpy as np
import yaml
from pypdf import PdfReader
from src.embeddings import Embedder

SUPPORTED_EXT = {".pdf", ".txt", ".md"}


REQUIRED_FIELDS = ("file", "title", "license")


def load_manifest(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    entries = {}
    for i, doc in enumerate(data.get("documents", []), start=1):
        missing = [k for k in REQUIRED_FIELDS if not doc.get(k)]
        if missing:
            raise ValueError(
                f"En {path}, la entrada #{i} ({doc.get('file', 'sin nombre')}) "
                f"no tiene estos campos obligatorios: {', '.join(missing)}"
            )
        entries[doc["file"]] = doc
    return entries


def read_document(path: Path) -> list[tuple[int | None, str]]:
    """Returns (page_number, text) pairs. Text files have no page number."""
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return [(i, page.extract_text() or "") for i, page in enumerate(reader.pages, start=1)]
    return [(None, path.read_text(encoding="utf-8", errors="ignore"))]


def chunk_text(text: str, chunk_chars: int = 1000, overlap: int = 150) -> list[str]:
    """Splits by paragraph, packs into chunks up to chunk_chars, and carries `overlap`
    characters into the next chunk so that context is not cut abruptly."""
    units: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = " ".join(para.split())
        if para:
            units += [para[i:i + chunk_chars] for i in range(0, len(para), chunk_chars)]

    chunks: list[str] = []
    current = ""
    for unit in units:
        if current and len(current) + 1 + len(unit) > chunk_chars:
            chunks.append(current)
            current = current[-overlap:].strip() if overlap else ""
        current = f"{current} {unit}".strip()
    if current:
        chunks.append(current)
    return chunks


class KnowledgeBase:
    """Persistent FAISS index with a JSONL sidecar for chunk text and metadata.
    Uses inner product on L2-normalized vectors, which equals cosine similarity."""

    def __init__(self, index_dir: str | Path, embed_cfg: dict):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = Embedder(embed_cfg)
        self._index = None
        self._chunks: list[dict] = []   

    def build(self, documents_dir: str, manifest_path: str, chunk_chars: int, overlap: int) -> int:
        manifest = load_manifest(manifest_path)
        chunks: list[dict] = []
        for path in sorted(Path(documents_dir).iterdir()):
            if path.suffix.lower() not in SUPPORTED_EXT:
                continue
            meta = manifest.get(path.name)
            if meta is None:
                print(f"[omitido] {path.name}: no tiene licencia declarada en el manifest")
                continue
            for page, text in read_document(path):
                for k, piece in enumerate(chunk_text(text, chunk_chars, overlap)):
                    chunks.append({
                        "chunk_id": f"{path.stem}:{page or 0}:{k}",
                        "source": meta["title"],
                        "license": meta["license"],
                        "page": page,
                        "text": piece,
                    })
        if not chunks:
            raise ValueError("No se generaron fragmentos. Revisa documents_dir y el manifest.")

        
        vectors = self.embedder.encode_passages([c["text"] for c in chunks])

        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        index_bytes = faiss.serialize_index(index)
        (self.index_dir / "index.faiss").write_bytes(index_bytes.tobytes())
        with open(self.index_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")

        self._index, self._chunks = index, chunks
        return len(chunks)

    def load(self) -> None:
        index_file = self.index_dir / "index.faiss"
        if not index_file.exists():
            raise FileNotFoundError("No hay índice. Ejecuta scripts/build_index.py primero.")
        raw = np.frombuffer(index_file.read_bytes(), dtype="uint8")
        self._index = faiss.deserialize_index(raw)
        with open(self.index_dir / "chunks.jsonl", encoding="utf-8") as f:
            self._chunks = [json.loads(line) for line in f]

    def search(self, query: str, top_k: int = 4, min_score: float = 0.45) -> list[dict]:
        if self._index is None:
            self.load()
        
        q = self.embedder.encode_queries([query])
        scores, ids = self._index.search(q, top_k)
        return [
            {**self._chunks[i], "score": round(float(s), 3)}
            for s, i in zip(scores[0], ids[0])
            if i != -1 and s >= min_score
        ]