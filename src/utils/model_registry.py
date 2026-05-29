"""
utils/model_registry.py
========================
Lightweight local model registry for versioned model artefacts.

Capabilities:
  - Register, promote, and retire model versions
  - Tag models with metadata (metrics, config, training run ID)
  - Promote a model to 'champion' / 'challenger' status
  - Persist registry state in a JSON manifest
  - CLI-compatible interface for CI/CD pipelines
  - MLflow Model Registry integration (optional)
"""

from __future__ import annotations

import json
import logging
import shutil
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

REGISTRY_ROOT = Path("artifacts/model_registry")
MANIFEST_FILE = REGISTRY_ROOT / "registry_manifest.json"


# ─────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────


@dataclass
class ModelEntry:
    """Represents a single registered model version."""

    model_id: str
    name: str
    version: str
    stage: str                      # 'candidate' | 'staging' | 'champion' | 'retired'
    model_path: str
    config_path: Optional[str]
    registered_at: str
    metrics: Dict[str, float] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    training_run_id: Optional[str] = None
    description: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "ModelEntry":
        return cls(**d)

    @property
    def is_champion(self) -> bool:
        return self.stage == "champion"


# ─────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────


class ModelRegistry:
    """
    Local filesystem-backed model registry.

    All model artefacts are stored under REGISTRY_ROOT/{model_name}/{version}/.
    The registry state is persisted in a JSON manifest.

    Usage
    -----
    >>> registry = ModelRegistry()
    >>> entry = registry.register(
    ...     name="churn-ann",
    ...     model_path="artifacts/checkpoints/best_model.keras",
    ...     metrics={"val_auc": 0.873, "val_f1": 0.612},
    ...     tags={"dataset": "v1", "experiment": "baseline"},
    ... )
    >>> registry.promote(entry.model_id, stage="champion")
    >>> champion = registry.get_champion("churn-ann")
    """

    def __init__(self, root: str | Path = REGISTRY_ROOT):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self.root / "registry_manifest.json"
        self._entries: Dict[str, ModelEntry] = {}
        self._load_manifest()

    # ── Public API ──────────────────────────────

    def register(
        self,
        name: str,
        model_path: str | Path,
        version: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
        tags: Optional[Dict[str, str]] = None,
        config_path: Optional[str] = None,
        training_run_id: Optional[str] = None,
        description: str = "",
        stage: str = "candidate",
    ) -> ModelEntry:
        """Register a new model version."""
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model_id = str(uuid.uuid4())[:8]
        version = version or self._next_version(name)

        dest_dir = self.root / name / version
        dest_dir.mkdir(parents=True, exist_ok=True)

        dest_model = dest_dir / model_path.name
        shutil.copy2(model_path, dest_model)

        if config_path and Path(config_path).exists():
            shutil.copy2(config_path, dest_dir / "model_config.json")

        entry = ModelEntry(
            model_id=model_id,
            name=name,
            version=version,
            stage=stage,
            model_path=str(dest_model),
            config_path=str(dest_dir / "model_config.json") if config_path else None,
            registered_at=datetime.utcnow().isoformat(),
            metrics=metrics or {},
            tags=tags or {},
            training_run_id=training_run_id,
            description=description,
        )

        self._entries[model_id] = entry
        self._save_manifest()

        logger.info(
            "Model registered: name=%s version=%s id=%s stage=%s",
            name, version, model_id, stage,
        )
        return entry

    def promote(self, model_id: str, stage: str) -> ModelEntry:
        """
        Promote a model to a new lifecycle stage.

        If promoting to 'champion', the existing champion is automatically
        demoted to 'retired'.
        """
        valid_stages = {"candidate", "staging", "champion", "retired"}
        if stage not in valid_stages:
            raise ValueError(f"Stage must be one of {valid_stages}")

        entry = self._get_entry(model_id)

        if stage == "champion":
            # Retire existing champion for this model name
            for e in self._entries.values():
                if e.name == entry.name and e.is_champion and e.model_id != model_id:
                    e.stage = "retired"
                    logger.info("Previous champion retired: %s v%s", e.name, e.version)

        entry.stage = stage
        self._save_manifest()
        logger.info("Model %s promoted to %s", model_id, stage)
        return entry

    def get_champion(self, name: str) -> Optional[ModelEntry]:
        """Return the current champion model for a given name."""
        champions = [
            e for e in self._entries.values()
            if e.name == name and e.is_champion
        ]
        if not champions:
            logger.warning("No champion found for model '%s'", name)
            return None
        return champions[-1]

    def list_versions(self, name: str, stage: Optional[str] = None) -> List[ModelEntry]:
        """List all registered versions for a model name."""
        entries = [e for e in self._entries.values() if e.name == name]
        if stage:
            entries = [e for e in entries if e.stage == stage]
        return sorted(entries, key=lambda e: e.registered_at)

    def delete(self, model_id: str, delete_files: bool = False) -> None:
        """Remove a model entry from the registry."""
        entry = self._get_entry(model_id)
        if entry.is_champion:
            raise ValueError("Cannot delete the current champion. Promote another first.")

        if delete_files:
            model_dir = Path(entry.model_path).parent
            if model_dir.exists():
                shutil.rmtree(model_dir)
                logger.info("Deleted model files at %s", model_dir)

        del self._entries[model_id]
        self._save_manifest()
        logger.info("Model %s deleted from registry", model_id)

    def compare(self, model_id_a: str, model_id_b: str) -> Dict:
        """Side-by-side metric comparison of two models."""
        a = self._get_entry(model_id_a)
        b = self._get_entry(model_id_b)

        all_metrics = set(a.metrics) | set(b.metrics)
        comparison = {}
        for m in sorted(all_metrics):
            val_a = a.metrics.get(m, float("nan"))
            val_b = b.metrics.get(m, float("nan"))
            comparison[m] = {"model_a": val_a, "model_b": val_b, "delta": val_b - val_a}

        return {
            "model_a": {"id": model_id_a, "version": a.version, "stage": a.stage},
            "model_b": {"id": model_id_b, "version": b.version, "stage": b.stage},
            "metrics": comparison,
        }

    def print_registry(self) -> None:
        """Print a formatted registry overview."""
        header = f"{'Name':<20} {'Ver':<8} {'Stage':<12} {'Val AUC':<10} {'Val F1':<10} {'ID':<10}"
        print("\n" + "─" * len(header))
        print(header)
        print("─" * len(header))
        for entry in sorted(self._entries.values(), key=lambda e: (e.name, e.registered_at)):
            auc = entry.metrics.get("val_auc", float("nan"))
            f1 = entry.metrics.get("val_f1", float("nan"))
            star = " ★" if entry.is_champion else ""
            print(
                f"{entry.name + star:<20} {entry.version:<8} {entry.stage:<12} "
                f"{auc:<10.4f} {f1:<10.4f} {entry.model_id:<10}"
            )
        print("─" * len(header) + "\n")

    # ── Private ──────────────────────────────────

    def _get_entry(self, model_id: str) -> ModelEntry:
        if model_id not in self._entries:
            raise KeyError(f"Model ID not found: {model_id}")
        return self._entries[model_id]

    def _next_version(self, name: str) -> str:
        existing = self.list_versions(name)
        if not existing:
            return "v1"
        nums = []
        for e in existing:
            try:
                nums.append(int(e.version.lstrip("v")))
            except ValueError:
                pass
        return f"v{max(nums) + 1}" if nums else "v1"

    def _save_manifest(self) -> None:
        data = {mid: entry.to_dict() for mid, entry in self._entries.items()}
        with open(self._manifest_path, "w") as f:
            json.dump(data, f, indent=2)

    def _load_manifest(self) -> None:
        if self._manifest_path.exists():
            with open(self._manifest_path) as f:
                data = json.load(f)
            self._entries = {mid: ModelEntry.from_dict(d) for mid, d in data.items()}
            logger.info("Registry loaded: %d model entries", len(self._entries))
