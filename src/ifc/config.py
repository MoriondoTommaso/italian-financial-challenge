from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load YAML file and return a Python dictionary."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fid:
        return yaml.safe_load(fid)


def resolve_project_root(
    start: Optional[Path] = None,
    markers: tuple[str, ...] = ("configs", "src"),
    max_up: int = 15,
) -> Path:
    """
    Walk up from `start` (or current working directory) until we find a folder
    containing all marker directories (default: configs/ and src/).
    """
    p = (start or Path.cwd()).resolve()
    for _ in range(max_up):
        if all((p / m).exists() for m in markers):
            return p
        if p.parent == p:
            break
        p = p.parent
    raise FileNotFoundError(
        f"Could not resolve project root from {start or Path.cwd()}. "
        f"Expected folders: {markers} somewhere above."
    )


def load_data_config(
    config_filename: str = "data.yaml",
    project_root: Optional[Path] = None,
) -> "DataConfig":
    """
    Load DataConfig from <project_root>/configs/<config_filename>.
    If project_root is None, it auto-resolves from current working directory.
    """
    root = project_root or resolve_project_root()
    cfg_path = (root / "configs" / config_filename).resolve()
    return DataConfig.from_yaml(cfg_path)


def load_split_config(
    config_filename: str = "split.yaml",
    project_root: Optional[Path] = None,
) -> "SplitConfig":
    """
    Load SplitConfig from <project_root>/configs/<config_filename>.
    If project_root is None, it auto-resolves from current working directory.
    """
    root = project_root or resolve_project_root()
    cfg_path = (root / "configs" / config_filename).resolve()
    return SplitConfig.from_yaml(cfg_path)


@dataclass(frozen=True)
class DataConfig:
    """File Paths + Columns names"""
    train_path: Path
    test_path: Path
    id_col: str
    time_col: str
    target_col: str

    @staticmethod
    def from_yaml(path: str | Path) -> "DataConfig":
        d = load_yaml(path)
        return DataConfig(
            train_path=Path(d["train_path"]),
            test_path=Path(d["test_path"]),
            id_col=d["id_col"],
            time_col=d["time_col"],
            target_col=d["target_col"],
        )

    def resolve_paths(self, project_root: Optional[Path] = None) -> tuple[Path, Path]:
        """
        Return absolute (resolved) train/test paths under the project root.

        If project_root is None, it is auto-resolved (searching for configs/ and src/).
        """
        root = project_root or resolve_project_root()
        train_abs = (root / self.train_path).resolve()
        test_abs = (root / self.test_path).resolve()
        return train_abs, test_abs

@dataclass(frozen=True)
class SplitConfig:
    """Split configurations for train and validation sets"""
    strategy: str
    time_col: str
    id_col: str
    target_col: str
    shuffle: bool
    history_years: list[int]
    train_years: list[int]
    val_years: list[int]

    @staticmethod
    def from_yaml(path: str | Path) -> "SplitConfig":
        d = load_yaml(path)
        return SplitConfig(
            strategy=d["strategy"],
            time_col=d["time_col"],
            id_col=d["id_col"],
            target_col=d["target_col"],
            shuffle=bool(d.get("shuffle", False)),
            history_years=list(d.get("history_years", [])),
            train_years=list(d["train_years"]),
            val_years=list(d["val_years"]),
        )
