import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.secop import SecopClient

if __name__ == "__main__":
    cfg = load_config()
    client = SecopClient(cfg["datos_abiertos"])
    rows = client.search("papeleria")
    print(f"Filas encontradas: {len(rows)}")
    if rows:
        print("Campos disponibles:", sorted(rows[0].keys()))
        print(client.to_chunk(rows[0])["text"])