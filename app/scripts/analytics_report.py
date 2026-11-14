import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.logging_analytics import build_report

if __name__ == "__main__":
    report = build_report(load_config()["logging"]["log_path"])
    for key, value in report.items():
        print(f"\n== {key} ==\n{value}")