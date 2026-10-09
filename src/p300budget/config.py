"""Load the frozen configuration and resolve project paths."""
from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "frozen.yaml"


def load_config(path: Path | str = CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_hash"] = config_hash(path)
    return cfg


def config_hash(path: Path | str = CONFIG_PATH) -> str:
    """Short hash recorded in every output so results trace back to settings."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def select_dev(df, cfg: dict):
    """Dev mode: keep only the first few records of a few studies (quick runs).

    df needs 'study' and 'record' (or 'participant') columns.
    """
    dev = cfg.get("dev", {})
    studies = dev.get("studies", ["A", "B", "E"])
    n = int(dev.get("records_per_study", 2))
    key = "record" if "record" in df.columns else "participant"
    sub = df[df["study"].isin(studies)].sort_values(["study", key])
    return sub.groupby("study", group_keys=False).head(n)


def output_name(name: str, dev: bool) -> str:
    """'sweep_results.csv' -> 'sweep_results_dev.csv' in dev mode."""
    if not dev:
        return name
    stem, dot, ext = name.rpartition(".")
    return f"{stem}_dev.{ext}" if dot else f"{name}_dev"


def project_path(rel: str) -> Path:
    """Resolve a path from the config relative to the project root."""
    p = Path(rel)
    return p if p.is_absolute() else PROJECT_ROOT / p
