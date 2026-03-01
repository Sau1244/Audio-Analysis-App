from __future__ import annotations

import csv
import hashlib
import random
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from tqdm import tqdm

SPLIT_MAP = {
    "train": "train",
    "trn": "train",
    "dev": "val",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "test": "test",
    "eval": "test",
}

EN_TOKENS = {"en", "eng", "english"}
DE_TOKENS = {"de", "deu", "ger", "german"}

IN_THE_WILD_SUBDIR = "release_in_the_wild"
IN_THE_WILD_META = "meta.csv"
IN_THE_WILD_AUDIO_EXTS = {".wav", ".flac"}


@dataclass
class ManifestRow:
    path: str
    label: int
    split: str
    source: str
    group_id: Optional[str] = None
    extras: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, str]:
        row = {
            "path": self.path,
            "label": str(int(self.label)),
            "split": self.split,
            "source": self.source,
        }
        if self.group_id:
            row["group_id"] = self.group_id
        for key, value in self.extras.items():
            if value is None:
                continue
            row[key] = str(value)
        return row


@dataclass
class ManifestBuildConfig:
    dfadd_dir: Path
    dfadd_audio_dir: Path
    mlaad_dir: Path
    ml_df_dir: Path
    in_the_wild_dir: Path
    in_the_wild_zip: Path
    val_ratio: float
    seed: int


@dataclass
class DatasetConfig:
    name: str
    data_dir: Path
    label_candidates: tuple[str, ...]
    path_candidates: tuple[str, ...]
    split_candidates: tuple[str, ...]
    speaker_candidates: tuple[str, ...]
    system_candidates: tuple[str, ...]
    language_candidates: tuple[str, ...]
    force_language_split: bool = False


def _normalize_split(value: str) -> str:
    return SPLIT_MAP.get(str(value).strip().lower(), str(value).strip().lower())


def _normalize_label(value) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        if int(value) in (0, 1):
            return int(value)
    if isinstance(value, str):
        norm = value.strip().lower()
        if norm in {"0", "1"}:
            return int(norm)
        if norm in {"bonafide", "bona-fide", "real", "genuine", "human"}:
            return 0
        if norm in {"spoof", "spoofed", "fake", "attack", "synthetic", "ai"}:
            return 1
    raise ValueError(f"Unrecognized label: {value!r}")


def _normalize_ml_df_label(tool: str) -> int:
    norm = tool.strip().lower()
    if norm in {"bonafide", "bona-fide", "real"}:
        return 0
    if norm in {"spoof", "fake"}:
        return 1
    return 1


def _language_from_wav_path(wav_file: str) -> str:
    p = Path(wav_file)
    if p.parts:
        first = p.parts[0]
        if first.startswith("dataset_"):
            return first.split("_", 1)[1]
    stem = p.stem
    if "_" in stem:
        return stem.split("_", 1)[0]
    return "unknown"


def _iter_metadata_rows(metadata_path: Path) -> list[dict]:
    rows: list[dict] = []
    with metadata_path.open("r", encoding="utf-8") as f:
        header = f.readline().strip().split()
        expected = ["wav_file", "tool", "gender", "group", "speaker"]
        if header[: len(expected)] != expected:
            raise ValueError(f"Unexpected metadata header in {metadata_path}: {header}")
        for line_no, line in enumerate(f, start=2):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) < 5:
                raise ValueError(
                    f"Malformed metadata row at {metadata_path}:{line_no}: {line!r}"
                )
            wav_file, tool, gender, group, speaker = parts[:5]
            rows.append(
                {
                    "wav_file": wav_file,
                    "tool": tool,
                    "gender": gender,
                    "group": group,
                    "speaker": speaker,
                }
            )
    return rows


def _rows_from_ml_df(ml_df_dir: Path) -> list[ManifestRow]:
    metadata_files = sorted(ml_df_dir.glob("metadata_*.csv"))
    if not metadata_files:
        raise FileNotFoundError(f"No metadata_*.csv found in {ml_df_dir}")

    def _speaker_seed(speaker: str) -> int:
        digest = hashlib.md5(speaker.encode("utf-8")).hexdigest()
        return 1337 + int(digest[:8], 16)

    def _split_it_records(records: list[dict]) -> dict[str, str]:
        by_speaker: dict[str, list[dict]] = {}
        for record in records:
            by_speaker.setdefault(str(record["speaker"]), []).append(record)

        split_map: dict[str, str] = {}
        for speaker, items in by_speaker.items():
            rng = random.Random(_speaker_seed(speaker))
            items = list(items)
            rng.shuffle(items)
            n_total = len(items)
            n_val = int(round(n_total * 0.1))
            n_train = max(n_total - n_val, 0)
            train_items = items[:n_train]
            val_items = items[n_train : n_train + n_val]
            for record in train_items:
                split_map[record["wav_file"]] = "train"
            for record in val_items:
                split_map[record["wav_file"]] = "val"
        return split_map

    rows: list[ManifestRow] = []
    missing = 0
    total = 0
    for metadata_path in metadata_files:
        records = _iter_metadata_rows(metadata_path)
        it_split_map = None
        if metadata_path.stem.endswith("_IT"):
            it_split_map = _split_it_records(records)
        for record in tqdm(
            records, desc=f"ml-df:{metadata_path.name}", total=len(records)
        ):
            total += 1
            wav_file = record["wav_file"]
            wav_path = ml_df_dir / wav_file
            if not (
                wav_path.is_file() and wav_path.suffix.lower() in {".wav", ".flac"}
            ):
                missing += 1
                continue
            language = _language_from_wav_path(wav_file)
            speaker = record["speaker"]
            group_id = f"{language}_{speaker}"
            if language == "IT" and it_split_map is not None:
                split_name = it_split_map.get(wav_file, "train")
            else:
                split_name = _normalize_split(record["group"])
            rows.append(
                ManifestRow(
                    path=str(wav_path.resolve()),
                    label=_normalize_ml_df_label(record["tool"]),
                    split=split_name,
                    source="ML-DF",
                    group_id=group_id,
                    extras={
                        "tool": record["tool"],
                        "gender": record["gender"],
                        "speaker_id": speaker,
                        "language": language,
                        "wav_file": wav_file,
                    },
                )
            )
    if missing:
        print(f"[warn] skipped {missing}/{total} missing audio files in ML-DF.")
    return rows


def _dedupe_rows(rows: list[ManifestRow]) -> list[ManifestRow]:
    seen = set()
    deduped = []
    for r in rows:
        if r.path in seen:
            continue
        seen.add(r.path)
        deduped.append(r)
    return deduped


def _first_present(columns: list[str], candidates: tuple[str, ...]) -> Optional[str]:
    for c in candidates:
        if c in columns:
            return c
    return None


def _extract_path(value, data_dir: Path) -> Optional[str]:
    path_value = value
    if isinstance(value, dict) and "path" in value:
        path_value = value["path"]
    if not path_value:
        return None
    p = Path(str(path_value))
    if not p.is_absolute():
        p = data_dir / p
    return str(p.resolve())


def _infer_language_from_path(path: str) -> Optional[str]:
    parts = [p.lower() for p in Path(path).parts]
    for token in parts:
        if token in EN_TOKENS:
            return "en"
        if token in DE_TOKENS:
            return "de"
    stem = Path(path).stem.lower()
    for token in stem.split("_") + stem.split("-"):
        if token in EN_TOKENS:
            return "en"
        if token in DE_TOKENS:
            return "de"
    return None


def _group_id(
    label: int, speaker: Optional[str], tts_system: Optional[str]
) -> Optional[str]:
    if label == 1 and tts_system:
        return f"tts:{tts_system}"
    if speaker:
        return f"spk:{speaker}"
    if tts_system:
        return f"tts:{tts_system}"
    return None


def _find_parquet_files(data_dir: Path) -> list[Path]:
    if not data_dir.exists():
        return []
    files = sorted(data_dir.rglob("*.parquet"))
    return files


def _resolve_parquet_root(data_dir: Path) -> Path:
    if _find_parquet_files(data_dir):
        return data_dir
    nested = data_dir / "data"
    if _find_parquet_files(nested):
        return nested
    return data_dir


def _load_parquet_dir(data_dir: Path):
    try:
        from datasets import Dataset, DatasetDict, load_dataset
    except Exception as exc:
        raise RuntimeError(
            "Loading parquet manifests requires the 'datasets' package."
        ) from exc

    parquet_root = _resolve_parquet_root(data_dir)
    parquet_files = _find_parquet_files(parquet_root)
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {data_dir}")
    split_files: dict[str, list[str]] = {"train": [], "test": [], "validation": []}
    for fp in parquet_files:
        name = fp.name.lower()
        if "train" in name:
            split_files["train"].append(str(fp))
        elif "test" in name:
            split_files["test"].append(str(fp))
        elif "valid" in name or "val" in name or "dev" in name:
            split_files["validation"].append(str(fp))
    data_files = {k: v for k, v in split_files.items() if v}
    ds = load_dataset(
        "parquet", data_files=data_files or [str(p) for p in parquet_files]
    )
    if isinstance(ds, DatasetDict):
        return ds
    if isinstance(ds, Dataset):
        return DatasetDict({"train": ds})
    raise TypeError(f"Unsupported dataset type from {data_dir}: {type(ds)}")


def _rows_from_dataset(ds_dict, cfg: DatasetConfig) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    skipped = 0
    total = 0
    for split_name, ds in ds_dict.items():
        columns = list(getattr(ds, "column_names", []))
        label_col = _first_present(columns, cfg.label_candidates)
        path_col = _first_present(columns, cfg.path_candidates)
        split_col = _first_present(columns, cfg.split_candidates)
        speaker_col = _first_present(columns, cfg.speaker_candidates)
        system_col = _first_present(columns, cfg.system_candidates)
        language_col = _first_present(columns, cfg.language_candidates)
        if not label_col or not path_col:
            raise ValueError(
                f"{cfg.name}: missing label/path column (label={label_col}, path={path_col}). "
                f"Columns: {columns}"
            )
        for row in tqdm(
            ds,
            total=len(ds),
            desc=f"{cfg.name}:{split_name}",
        ):
            total += 1
            try:
                path = _extract_path(row.get(path_col), cfg.data_dir)
                if path is None or not Path(path).exists():
                    skipped += 1
                    continue
                label = _normalize_label(row.get(label_col))
            except Exception:
                skipped += 1
                continue
            split = None
            if split_col and row.get(split_col) is not None:
                split = _normalize_split(row.get(split_col))
            if not split:
                split = _normalize_split(split_name)
            language = None
            if language_col and row.get(language_col) is not None:
                language = str(row.get(language_col)).strip().lower()
            if language is None:
                language = _infer_language_from_path(path)
            tts_system = None
            if system_col and row.get(system_col) is not None:
                tts_system = str(row.get(system_col))
            speaker = None
            if speaker_col and row.get(speaker_col) is not None:
                speaker = str(row.get(speaker_col))
            group_id = _group_id(label, speaker, tts_system)
            extras: dict[str, str] = {}
            if tts_system:
                extras["tts_system"] = tts_system
            if language:
                extras["language"] = language
            if speaker:
                extras["speaker_id"] = speaker
            rows.append(
                ManifestRow(
                    path=path,
                    label=label,
                    split=split,
                    source=cfg.name,
                    group_id=group_id,
                    extras=extras,
                )
            )
    if skipped:
        print(f"[warn] {cfg.name}: skipped {skipped}/{total} rows (missing/invalid).")
    return rows


def _maybe_write_audio(
    *,
    audio: dict,
    split: str,
    output_dir: Path,
    name_hint: Optional[str] = None,
) -> Optional[str]:
    try:
        import numpy as np
        import soundfile as sf
        import torch
    except Exception as exc:
        raise RuntimeError(
            "DFADD audio materialization requires numpy, soundfile, and torch."
        ) from exc

    audio_path = None
    audio_bytes = None
    audio_array = None
    audio_sr = None
    audio_decoder = None

    if isinstance(audio, dict):
        audio_bytes = audio.get("bytes")
        audio_array = audio.get("array")
        audio_sr = audio.get("sampling_rate")
        audio_path = audio.get("path")
    elif hasattr(audio, "get_all_samples"):
        audio_decoder = audio

    audio_path = audio_path or name_hint
    if not audio_path:
        return None

    rel_path = Path(str(audio_path)).name
    out_path = output_dir / split / rel_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not out_path.exists():
        if audio_bytes:
            out_path.write_bytes(audio_bytes)
        elif audio_array is not None and audio_sr:
            data = np.asarray(audio_array, dtype=np.float32)
            sf.write(out_path, data, int(audio_sr))
        elif audio_decoder is not None:
            samples = audio_decoder.get_all_samples()
            data = samples.data
            if isinstance(data, torch.Tensor):
                data = data.cpu().numpy()
            data = np.asarray(data, dtype=np.float32)
            if data.ndim == 2:
                data = data.T
            sf.write(out_path, data, int(samples.sample_rate))
        else:
            return None
    return str(out_path.resolve())


def _candidate_dfadd_path(
    *, path_value: Optional[str], split: str, audio_root: Path
) -> Optional[str]:
    if not path_value:
        return None
    base = Path(str(path_value)).name
    candidate = audio_root / split / base
    if candidate.exists():
        return str(candidate.resolve())
    candidate = audio_root / base
    if candidate.exists():
        return str(candidate.resolve())
    return None


def _apply_language_split(rows: list[ManifestRow]) -> None:
    for row in rows:
        lang = (row.extras.get("language") or "").lower()
        if lang in EN_TOKENS:
            row.split = "train"
        elif lang in DE_TOKENS:
            row.split = "test"


def _ensure_val_split(rows: list[ManifestRow], val_ratio: float, seed: int) -> None:
    if val_ratio <= 0:
        return
    if any(r.split == "val" for r in rows):
        return
    train_rows = [r for r in rows if r.split == "train"]
    if not train_rows:
        return
    grouped: dict[str, list[ManifestRow]] = {}
    for row in train_rows:
        group_id = row.group_id or row.path
        grouped.setdefault(group_id, []).append(row)
    groups = sorted(grouped.keys())
    rng = random.Random(seed)
    rng.shuffle(groups)
    n_val = int(round(len(groups) * val_ratio))
    val_groups = set(groups[:n_val])
    for row in train_rows:
        if (row.group_id or row.path) in val_groups:
            row.split = "val"


def _rows_from_mlaad_dir(data_dir: Path) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    original_dir = data_dir / "original"
    fake_dir = data_dir / "fake"
    if not original_dir.exists() or not fake_dir.exists():
        raise FileNotFoundError(
            f"Expected MLAAD-tiny structure with original/ and fake/ under {data_dir}"
        )

    for lang_dir in sorted(original_dir.iterdir()):
        if not lang_dir.is_dir():
            continue
        language = lang_dir.name.lower()
        wav_files = sorted(lang_dir.rglob("*.wav"))
        for wav_path in tqdm(
            wav_files,
            total=len(wav_files),
            desc=f"MLAAD-tiny:original:{language}",
        ):
            rows.append(
                ManifestRow(
                    path=str(wav_path.resolve()),
                    label=0,
                    split="train",
                    source="MLAAD-tiny",
                    extras={"language": language},
                )
            )

    for lang_dir in sorted(fake_dir.iterdir()):
        if not lang_dir.is_dir():
            continue
        language = lang_dir.name.lower()
        for system_dir in sorted(lang_dir.iterdir()):
            if not system_dir.is_dir():
                continue
            tts_system = system_dir.name
            wav_files = sorted(system_dir.rglob("*.wav"))
            for wav_path in tqdm(
                wav_files,
                total=len(wav_files),
                desc=f"MLAAD-tiny:fake:{language}:{tts_system}",
            ):
                rows.append(
                    ManifestRow(
                        path=str(wav_path.resolve()),
                        label=1,
                        split="train",
                        source="MLAAD-tiny",
                        group_id=f"tts:{tts_system}",
                        extras={
                            "tts_system": tts_system,
                            "language": language,
                        },
                    )
                )
    if not rows:
        raise RuntimeError(f"No MLAAD-tiny audio files found under {data_dir}")
    return rows


def _load_in_the_wild_meta(in_the_wild_dir: Path, in_the_wild_zip: Path) -> list[dict]:
    meta_path = in_the_wild_dir / IN_THE_WILD_SUBDIR / IN_THE_WILD_META
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return list(reader)
    if in_the_wild_zip.exists():
        with zipfile.ZipFile(in_the_wild_zip) as zf:
            with zf.open(f"{IN_THE_WILD_SUBDIR}/{IN_THE_WILD_META}") as f:
                reader = csv.DictReader(line.decode("utf-8") for line in f)
                return list(reader)
    raise FileNotFoundError(
        f"Missing In-the-Wild metadata at {meta_path} or {in_the_wild_zip}"
    )


def _rows_from_in_the_wild(
    in_the_wild_dir: Path, in_the_wild_zip: Path, split: str
) -> list[ManifestRow]:
    meta_rows = _load_in_the_wild_meta(in_the_wild_dir, in_the_wild_zip)
    audio_root = in_the_wild_dir / IN_THE_WILD_SUBDIR
    if not audio_root.exists():
        raise FileNotFoundError(
            f"In-the-Wild audio directory not found: {audio_root}. "
            "Unzip release_in_the_wild.zip first."
        )

    rows: list[ManifestRow] = []
    missing = 0
    for row in meta_rows:
        rel = row.get("file")
        if not rel:
            continue
        path = audio_root / rel
        if not (path.exists() and path.suffix.lower() in IN_THE_WILD_AUDIO_EXTS):
            missing += 1
            continue
        label = _normalize_label(row.get("label"))
        extras: dict[str, str] = {}
        speaker = row.get("speaker")
        if speaker:
            extras["speaker_id"] = speaker
        rows.append(
            ManifestRow(
                path=str(path.resolve()),
                label=label,
                split=split,
                source="In-the-Wild",
                extras=extras,
            )
        )
    if missing:
        print(f"[warn] In-the-Wild: skipped {missing} missing audio files.")
    return rows


def build_manifest_rows(cfg: ManifestBuildConfig) -> list[ManifestRow]:
    if not cfg.dfadd_dir.exists():
        raise FileNotFoundError(f"Missing DFADD dir: {cfg.dfadd_dir}")
    if not cfg.mlaad_dir.exists():
        raise FileNotFoundError(f"Missing MLAAD-tiny dir: {cfg.mlaad_dir}")
    if not cfg.ml_df_dir.exists():
        raise FileNotFoundError(f"Missing ML-DF directory: {cfg.ml_df_dir}")

    dfadd_cfg = DatasetConfig(
        name="DFADD",
        data_dir=cfg.dfadd_dir,
        label_candidates=("label", "is_spoof", "spoof", "class", "target", "type"),
        path_candidates=(
            "path",
            "audio_path",
            "file",
            "file_path",
            "wav_path",
            "audio",
        ),
        split_candidates=("split", "subset", "set"),
        speaker_candidates=("speaker_id", "speaker", "spk_id", "spk"),
        system_candidates=(
            "tts_system",
            "system_id",
            "system",
            "tts",
            "tts_id",
            "model_id",
            "vocoder",
        ),
        language_candidates=("language", "lang", "locale"),
    )
    mlaad_cfg = DatasetConfig(
        name="MLAAD-tiny",
        data_dir=cfg.mlaad_dir,
        label_candidates=("label", "is_spoof", "spoof", "class", "target", "type"),
        path_candidates=(
            "path",
            "audio_path",
            "file",
            "file_path",
            "wav_path",
            "audio",
        ),
        split_candidates=("split", "subset", "set"),
        speaker_candidates=("speaker_id", "speaker", "spk_id", "spk"),
        system_candidates=(
            "tts_system",
            "system_id",
            "system",
            "tts",
            "tts_id",
            "model_id",
            "vocoder",
        ),
        language_candidates=("language", "lang", "locale"),
        force_language_split=True,
    )

    dfadd_rows: list[ManifestRow] = []
    dfadd_ds = _load_parquet_dir(dfadd_cfg.data_dir)
    for split_name, ds in dfadd_ds.items():
        columns = list(getattr(ds, "column_names", []))
        label_col = _first_present(columns, dfadd_cfg.label_candidates)
        path_col = _first_present(columns, dfadd_cfg.path_candidates)
        audio_col = "audio" if "audio" in columns else None
        if not label_col or not (path_col or audio_col):
            raise ValueError(
                f"{dfadd_cfg.name}: missing label/path column (label={label_col}, path={path_col}). "
                f"Columns: {columns}"
            )
        for row in tqdm(
            ds,
            total=len(ds),
            desc=f"DFADD:{split_name}",
        ):
            try:
                label = _normalize_label(row.get(label_col))
            except Exception:
                continue
            split = _normalize_split(row.get("split") or split_name)
            path = None
            if path_col:
                path = _extract_path(
                    row.get(path_col),
                    _resolve_parquet_root(dfadd_cfg.data_dir),
                )
            if not path or not Path(path).exists():
                path = _candidate_dfadd_path(
                    path_value=path,
                    split=split,
                    audio_root=cfg.dfadd_audio_dir,
                )
            if (not path or not Path(path).exists()) and audio_col:
                name_hint = row.get("audio_name") or (
                    row.get(path_col) if path_col else None
                )
                path = _maybe_write_audio(
                    audio=row.get(audio_col),
                    split=split,
                    output_dir=cfg.dfadd_audio_dir,
                    name_hint=name_hint,
                )
            if not path or not Path(path).exists():
                continue
            dfadd_rows.append(
                ManifestRow(
                    path=str(Path(path).resolve()),
                    label=label,
                    split=split,
                    source=dfadd_cfg.name,
                )
            )
    if not dfadd_rows:
        print("[warn] DFADD: no rows were resolved; check parquet/audio files.")

    try:
        mlaad_rows = _rows_from_dataset(
            _load_parquet_dir(mlaad_cfg.data_dir), mlaad_cfg
        )
    except FileNotFoundError:
        mlaad_rows = _rows_from_mlaad_dir(mlaad_cfg.data_dir)

    if mlaad_rows and mlaad_cfg.force_language_split:
        only_train = all(r.split == "train" for r in mlaad_rows)
        if only_train or not any(r.split == "test" for r in mlaad_rows):
            _apply_language_split(mlaad_rows)

    ml_df_rows = _dedupe_rows(_rows_from_ml_df(cfg.ml_df_dir))

    _ensure_val_split(dfadd_rows, val_ratio=cfg.val_ratio, seed=cfg.seed)
    if mlaad_rows:
        _ensure_val_split(mlaad_rows, val_ratio=cfg.val_ratio, seed=cfg.seed)

    in_the_wild_rows = _rows_from_in_the_wild(
        cfg.in_the_wild_dir, cfg.in_the_wild_zip, split="val"
    )

    return dfadd_rows + mlaad_rows + ml_df_rows + in_the_wild_rows


def write_manifest(rows: list[ManifestRow], manifest_path: Path) -> Path:
    out_path = Path(manifest_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    base_fields = ["path", "label", "split", "source", "group_id"]
    extra_fields = sorted(
        {k for r in rows for k in r.to_dict().keys() if k not in base_fields}
    )
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=base_fields + extra_fields)
        writer.writeheader()
        writer.writerows([r.to_dict() for r in rows])
    return out_path


def read_manifest_rows(manifest_path: Path) -> list[ManifestRow]:
    base_fields = {"path", "label", "split"}
    rows: list[ManifestRow] = []
    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        if not base_fields.issubset(fieldnames):
            missing = ", ".join(sorted(base_fields - fieldnames))
            raise ValueError(f"Manifest missing required columns: {missing}")
        for row in reader:
            label = _normalize_label(row.get("label"))
            split = _normalize_split(row.get("split") or "")
            source = row.get("source") or row.get("source_ds") or "unknown"
            group_id = row.get("group_id") or None
            extras = {
                k: v
                for k, v in row.items()
                if k not in {"path", "label", "split", "source", "group_id"}
                and v not in (None, "")
            }
            rows.append(
                ManifestRow(
                    path=row.get("path") or "",
                    label=label,
                    split=split,
                    source=source,
                    group_id=group_id,
                    extras=extras,
                )
            )
    return rows


def split_manifest_rows(rows: list[ManifestRow]) -> dict[str, list[ManifestRow]]:
    splits: dict[str, list[ManifestRow]] = {"train": [], "val": [], "test": []}
    for row in rows:
        if row.split in splits:
            splits[row.split].append(row)
    return splits


def validate_manifest(rows: list[ManifestRow], *, check_paths: bool = True) -> None:
    if not rows:
        raise ValueError("Manifest is empty.")

    errors: list[str] = []
    allowed_splits = {"train", "val", "test"}
    for idx, row in enumerate(rows, start=1):
        if not row.path:
            errors.append(f"Row {idx}: missing path")
        if row.split not in allowed_splits:
            errors.append(f"Row {idx}: invalid split '{row.split}'")
        if row.label not in (0, 1):
            errors.append(f"Row {idx}: invalid label '{row.label}'")
        if check_paths and row.path and not Path(row.path).exists():
            errors.append(f"Row {idx}: missing file '{row.path}'")
        if len(errors) >= 20:
            break
    if errors:
        raise ValueError("Manifest validation failed:\n" + "\n".join(errors))
