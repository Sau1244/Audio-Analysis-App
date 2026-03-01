import argparse
import os
from pathlib import Path
from typing import Optional

from AI_dio.training.common import load_yaml_config, resolve_path
from AI_dio.training.pipeline import run_training

ROOT = Path(__file__).parents[3].resolve()
DEFAULT_CONFIG = ROOT / "training_config.yml"


def load_config(config_path: Optional[Path] = None) -> tuple[dict, Path]:
    if config_path is None:
        config_env = os.environ.get("TRAINING_CONFIG")
        config_path = Path(config_env) if config_env else DEFAULT_CONFIG
    if not config_path.is_absolute():
        config_path = resolve_path(ROOT, config_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Missing training config: {config_path}. Create it or set TRAINING_CONFIG."
        )

    config = load_yaml_config(config_path)
    return config, config_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train from config.")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    config, config_path = load_config(args.config)
    print(f"Loaded training config from {config_path}")
    run_training(config, config_path)
