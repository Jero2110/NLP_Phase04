import os
import re
import time
from datetime import datetime, timezone

import requests

from src.preprocessing import _fold

PROCESS_CODE = re.compile(r"\bCO1\.[A-Z]+\.\d+\b", re.IGNORECASE)
SEARCH_HINTS = ("proceso", "licitacion", "convocatoria", "contrato", "oportunidad", "abierto")


def detect_intent(text: str) -> tuple[str, str | None]:
    """Devuelve ('process', código), ('search', None) o ('docs', None)."""
    m = PROCESS_CODE.search(text)
    if m:
        return "process", m.group(0).upper()
    folded = _fold(text)
    if any(h in folded for h in SEARCH_HINTS):
        return "search", None
    return "docs", None


def _clean_term(text: str) -> str:
    # Evita inyección en SoQL: quita comillas y comodines, y limita la longitud
    return re.sub(r"[%_'\"\\]", " ", text).strip()[:100]

CITY_STEMS = {
    "medell": "MEDELL",   # Medellín / Medellin; se compara sin tildes
}

def _city_from_text(text: str) -> str | None:
    folded = _fold(text)
    for stem, value in CITY_STEMS.items():
        if stem in folded:
            return value
    return None


class SecopClient:
    def __init__(self, cfg: dict):
        self.url = f"https://www.datos.gov.co/resource/{cfg['dataset_id']}.json"
        self.fields = cfg["fields"]
        self.max_results = cfg.get("max_results", 5)
        self.timeout = cfg.get("timeout_s", 10)
        self.ttl = cfg.get("cache_ttl_s", 3600)
        self.headers = {}
        if token := os.getenv("DATOS_GOV_APP_TOKEN"):
            self.headers["X-App-Token"] = token
        self._cache: dict = {}

    def _get(self, params: dict) -> list[dict]:
        key = tuple(sorted(params.items()))
        cached = self._cache.get(key)
        if cached and time.time() - cached[0] < self.ttl:
            return cached[1]
        resp = requests.get(self.url, params=params, headers=self.headers, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        self._cache[key] = (time.time(), data)
        return data

    def get_by_reference(self, ref: str) -> list[dict]:
        f = self.fields
        return self._get({"$where": f"{f['reference']} = '{ref}'", "$limit": 1})

    def search(self, text: str) -> list[dict]:
        f = self.fields
        term = _clean_term(text)
        if not term:
            return []
        clauses = [f"upper({f['description']}) like upper('%{term}%')"]
        city = _city_from_text(text)
        if city:
            clauses.append(f"upper({f['city']}) like '%{city}%'")
        params = {
            "$where": " AND ".join(clauses),
            "$limit": self.max_results,
        }
        return self._get(params)

    def to_chunk(self, row: dict) -> dict:
        f = self.fields

        def val(key: str) -> str:
            v = row.get(f[key])
            return str(v) if v not in (None, "") else "no disponible"

        url = row.get(f["url"])
        if isinstance(url, dict):          # Socrata devuelve las columnas de tipo URL como objeto
            url = url.get("url")

        ref = val("reference")
        text = (
            f"Proceso {ref}. Nombre: {val('name')}. "
            f"Entidad: {val('entity')} ({val('city')}, {val('department')}). "
            f"Objeto: {val('description')}. "
            f"Modalidad: {val('modality')}. Estado: {val('status')}. "
            f"Valor base: {val('value')}. Proveedor: {val('awarded_to')}. "
            f"Enlace: {url or 'no disponible'}. "
            f"Datos consultados el {datetime.now(timezone.utc):%Y-%m-%d} (UTC)."
        )
        return {
            "chunk_id": f"secop:{ref}",
            "source": f"SECOP II · {ref}",
            "page": None,
            "text": text,
            "score": None,
            "origin": "secop",
            "url": url,
        }