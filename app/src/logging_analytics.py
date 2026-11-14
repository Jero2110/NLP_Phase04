import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


class EventLogger:
    """Appends one JSON object per turn. Session IDs are random and not linked to users."""

    def __init__(self, log_path: str | Path):
        self.path = Path(log_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: dict) -> None:
        record = {"ts": datetime.now(timezone.utc).isoformat(), **event}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_report(log_path: str | Path) -> dict:
    if not Path(log_path).exists():
        return {"aviso": "Todavía no hay eventos registrados."}

    df = pd.read_json(log_path, lines=True)
    report = {
        "turnos_totales": len(df),
        "estados": df["status"].value_counts().to_dict(),
        "tasa_sin_contexto": round((df["status"] == "no_context").mean(), 3),
        "latencia_ms_mediana": round(df["latency_ms"].median(), 1),
        "latencia_ms_p95": round(df["latency_ms"].quantile(0.95), 1),
        "turnos_con_patron_sospechoso": int(df["suspicious"].sum()),
        "temas_frecuentes": df.explode("topics")["topics"].dropna().value_counts().head(10).to_dict(),
    }
    if "query_text" in df:
        no_ctx = df.loc[df["status"] == "no_context", "query_text"]
        report["consultas_sin_contexto_frecuentes"] = no_ctx.value_counts().head(10).to_dict()
    if "error" in df:
        report["errores"] = df["error"].dropna().value_counts().to_dict()
    return report