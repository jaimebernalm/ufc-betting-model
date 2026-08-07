from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
EXTERNAL = DATA / "external"
CONFIGS = ROOT / "configs"
MODELS = ROOT / "artifacts" / "models"
METRICS = ROOT / "artifacts" / "metrics"
