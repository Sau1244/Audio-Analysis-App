import argparse
from pathlib import Path

from AI_dio.data_preprocessing.sources import (
    ManifestBuildConfig,
    build_manifest_rows,
    validate_manifest,
    write_manifest,
)
from AI_dio.training.common import get_section, load_yaml_config, resolve_path

ROOT = Path(__file__).parents[3].resolve()
DEFAULT_CONFIG = ROOT / "training_config.yml"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build manifest.csv from config.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def _build_manifest_config(cfg: dict) -> tuple[ManifestBuildConfig, Path]:
    manifest_cfg = get_section(cfg, "manifest_build")
    output = resolve_path(ROOT, manifest_cfg.get("output") or "manifest.csv")
    build_cfg = ManifestBuildConfig(
        dfadd_dir=resolve_path(ROOT, manifest_cfg.get("dfadd_dir") or "data/raw/dfadd"),
        dfadd_audio_dir=resolve_path(
            ROOT, manifest_cfg.get("dfadd_audio_dir") or "data/raw/dfadd/audio"
        ),
        mlaad_dir=resolve_path(
            ROOT, manifest_cfg.get("mlaad_dir") or "data/raw/mlaad_tiny"
        ),
        ml_df_dir=resolve_path(ROOT, manifest_cfg.get("ml_df_dir") or "data/raw/ml-df"),
        in_the_wild_dir=resolve_path(
            ROOT, manifest_cfg.get("in_the_wild_dir") or "data/raw/in_the_wild"
        ),
        in_the_wild_zip=resolve_path(
            ROOT,
            manifest_cfg.get("in_the_wild_zip")
            or "data/raw/in_the_wild/release_in_the_wild.zip",
        ),
        val_ratio=float(manifest_cfg.get("val_ratio", 0.1)),
        seed=int(manifest_cfg.get("seed", 1337)),
    )
    return build_cfg, output


def main() -> None:
    args = _parse_args()
    config = load_yaml_config(args.config)
    build_cfg, output = _build_manifest_config(config)

    rows = build_manifest_rows(build_cfg)
    validate_manifest(rows)
    write_manifest(rows, output)
    print(f"Wrote manifest with {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
