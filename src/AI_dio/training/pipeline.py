from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from AI_dio.data_preprocessing.dataset import AIDetectDataset
from AI_dio.data_preprocessing.features import FeatureParams, params_from_config
from AI_dio.data_preprocessing.sources import read_manifest_rows, validate_manifest
from AI_dio.training.checkpoints import save_checkpoint
from AI_dio.training.common import (
    choose_device,
    collate_fn,
    get_section,
    is_better_metric,
    resolve_metric,
    resolve_optional_path,
    resolve_path,
)
from AI_dio.training.metrics import (
    BinaryMetricsAccumulator,
    _best_threshold_max_acc,
    _binary_metrics_from_scores,
)
from AI_dio.training.models import BaselineCNN

ROOT = Path(__file__).parents[3].resolve()
DEFAULT_CKPT_DIR = ROOT / "checkpoints"


@dataclass(frozen=True)
class DataSettings:
    manifest: Path
    train_cache_dir: Optional[Path]
    eval_cache_dir: Optional[Path]
    augment_cfg: Optional[dict]
    augment_root: Optional[Path]


def init_wandb(wandb_cfg: dict, config: dict, config_path: Path):
    if not wandb_cfg.get("enabled", False):
        return None
    run = wandb.init(
        project=wandb_cfg.get("project", "AI_dio"),
        entity=wandb_cfg.get("entity"),
        name=wandb_cfg.get("run_name"),
        mode=wandb_cfg.get("mode"),
        config=config,
    )
    wandb.config.update({"config_path": str(config_path)}, allow_val_change=True)
    return run


def configure_device(device: torch.device) -> None:
    print("Using", device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")


def resolve_data_settings(config: dict) -> DataSettings:
    data_cfg = get_section(config, "data")
    use_cache = data_cfg.get("use_cache", True)
    use_cache_train = data_cfg.get("use_cache_train", use_cache)
    manifest = resolve_path(ROOT, data_cfg.get("manifest") or "manifest.csv")
    validate_manifest(read_manifest_rows(manifest))
    cache_dir = resolve_optional_path(
        ROOT,
        (data_cfg.get("cache_dir") or "data/cache/mel_3s_16k") if use_cache else None,
    )
    train_cache_dir = cache_dir if use_cache_train else None
    eval_cache_dir = cache_dir if use_cache else None
    augment_cfg = data_cfg.get("augment")
    if augment_cfg is not None and not isinstance(augment_cfg, dict):
        raise ValueError("data.augment must be a mapping when provided.")
    augment_root = resolve_optional_path(
        ROOT, (augment_cfg or {}).get("root") or "data/augment"
    )
    return DataSettings(
        manifest=manifest,
        train_cache_dir=train_cache_dir,
        eval_cache_dir=eval_cache_dir,
        augment_cfg=augment_cfg,
        augment_root=augment_root,
    )


def build_dataset(
    manifest: Path,
    split: Literal["train", "val", "test"],
    cache_dir: Optional[Path],
    params: FeatureParams,
    augment_cfg: Optional[dict],
    augment_root: Optional[Path],
) -> AIDetectDataset:
    augment_enabled = bool(augment_cfg and augment_cfg.get("enabled", False))
    use_augment = augment_enabled and split == "train" and cache_dir is None
    return AIDetectDataset(
        str(manifest),
        split,
        cache_dir=(None if cache_dir is None else str(cache_dir)),
        chunk_duration=params.chunk_duration,
        target_sr=params.target_sr,
        win_ms=params.win_ms,
        hop_ms=params.hop_ms,
        n_mels=params.n_mels,
        augment=use_augment,
        augment_root=None if augment_root is None else str(augment_root),
        augment_cfg=augment_cfg,
    )


def _iter_labels(rows: list[dict]):
    for row in rows:
        try:
            label = int(row["label"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid label in manifest: {row.get('label')}") from exc
        if label not in (0, 1):
            raise ValueError(f"Unexpected label in training data: {label}")
        yield label


def _label_counts(rows: list[dict]) -> dict[int, int]:
    counts = {0: 0, 1: 0}
    for label in _iter_labels(rows):
        counts[label] += 1
    return counts


def _labels_and_counts(rows: list[dict]) -> tuple[list[int], dict[int, int]]:
    counts = {0: 0, 1: 0}
    labels: list[int] = []
    for label in _iter_labels(rows):
        counts[label] += 1
        labels.append(label)
    return labels, counts


def _build_balanced_sampler(
    labels: list[int], counts: dict[int, int]
) -> WeightedRandomSampler:
    if counts[0] == 0 or counts[1] == 0:
        raise ValueError(f"Balanced sampler requires both classes, got counts={counts}")
    weights = [1.0 / counts[label] for label in labels]
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )


def _compute_class_weights_from_counts(
    counts: dict[int, int], mode: str
) -> Optional[torch.Tensor]:
    total = counts[0] + counts[1]
    if total == 0 or counts[0] == 0 or counts[1] == 0:
        return None
    if mode == "balanced":
        w0 = total / (2.0 * counts[0])
        w1 = total / (2.0 * counts[1])
    elif mode == "inverse":
        w0 = 1.0 / counts[0]
        w1 = 1.0 / counts[1]
    else:
        return None
    return torch.tensor([w0, w1], dtype=torch.float32)


def build_loader(
    dataset: AIDetectDataset,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
    drop_last: bool,
    sampler: Optional[WeightedRandomSampler],
    shuffle: bool,
) -> DataLoader:
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers and num_workers > 0,
        "drop_last": drop_last,
        "collate_fn": collate_fn,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **loader_kwargs)


def build_split_loader(
    *,
    dataset: Optional[AIDetectDataset] = None,
    manifest: Path,
    split: Literal["train", "val", "test"],
    cache_dir: Optional[Path],
    params: FeatureParams,
    augment_cfg: Optional[dict],
    augment_root: Optional[Path],
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
    drop_last: bool,
    sampler: Optional[WeightedRandomSampler],
    shuffle: bool,
) -> tuple[AIDetectDataset, DataLoader]:
    if dataset is None:
        dataset = build_dataset(
            manifest=manifest,
            split=split,
            cache_dir=cache_dir,
            params=params,
            augment_cfg=augment_cfg,
            augment_root=augment_root,
        )
    loader = build_loader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        drop_last=drop_last,
        sampler=sampler,
        shuffle=shuffle,
    )
    return dataset, loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    crit: nn.Module,
    opt: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epoch: int,
    epochs: int,
    clip_grad_norm: float,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(loader, desc=f"Epoch {epoch}/{epochs} [train]")
    for x, y in progress:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            logits = model(x)
            loss = crit(logits, y)

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        if clip_grad_norm and clip_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        scaler.step(opt)
        scaler.update()

        batch_size = y.size(0)
        running_loss += loss.item() * batch_size
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += batch_size

    avg_loss = running_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    crit: nn.Module,
    device: torch.device,
    epoch: int,
    epochs: int,
    threshold_cfg: Optional[object] = None,
) -> dict:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    metrics_acc = BinaryMetricsAccumulator(track_pr_auc=True)
    use_threshold = threshold_cfg is not None
    scores_list = []
    labels_list = []

    for x, y in tqdm(loader, desc=f"Epoch {epoch}/{epochs} [val]"):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            logits = model(x)
            loss = crit(logits, y)

        batch_size = y.size(0)
        running_loss += loss.item() * batch_size
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += batch_size
        metrics_acc.update(logits, y)
        if use_threshold:
            probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu()
            scores_list.append(probs)
            labels_list.append(y.detach().cpu())

    avg_loss = running_loss / max(total, 1)
    acc = correct / max(total, 1)
    metrics = {"loss": avg_loss, "acc": acc}
    metrics.update(metrics_acc.compute())
    if use_threshold and scores_list:
        scores = torch.cat(scores_list).numpy().astype(float)
        labels = torch.cat(labels_list).numpy().astype(int)
        threshold = None
        if isinstance(threshold_cfg, str) and threshold_cfg.lower() == "auto":
            threshold, _ = _best_threshold_max_acc(labels, scores)
        else:
            threshold = float(threshold_cfg)
        threshold_metrics = _binary_metrics_from_scores(labels, scores, threshold)
        if threshold_metrics:
            metrics.update(threshold_metrics)
            metrics["threshold"] = float(threshold)
    return metrics


def log_epoch_metrics(metrics: dict, epoch: int, epochs: int) -> None:
    print(
        " | ".join(
            [
                f"epoch={epoch}/{epochs}",
                f"train_loss={metrics['train/loss']:.4f}",
                f"train_acc={metrics['train/acc']:.4f}",
                (
                    f"val_loss={metrics['val/loss']:.4f}"
                    if "val/loss" in metrics
                    else "val_loss=NA"
                ),
                (
                    f"val_acc={metrics['val/acc']:.4f}"
                    if "val/acc" in metrics
                    else "val_acc=NA"
                ),
            ]
        )
    )
    if "val/loss" in metrics:
        print(
            " | ".join(
                [
                    "val_metrics",
                    f"balanced_acc={metrics.get('val/balanced_acc', float('nan')):.4f}",
                    f"roc_auc={metrics.get('val/roc_auc', float('nan')):.4f}",
                    f"eer={metrics.get('val/eer', float('nan')):.4f}",
                ]
            )
        )
        print(
            " | ".join(
                [
                    "val_class0",
                    f"precision={metrics.get('val/precision0', float('nan')):.4f}",
                    f"recall={metrics.get('val/recall0', float('nan')):.4f}",
                    f"f1={metrics.get('val/f1_0', float('nan')):.4f}",
                ]
            )
        )
        print(
            " | ".join(
                [
                    "val_class1",
                    f"precision={metrics.get('val/precision1', float('nan')):.4f}",
                    f"recall={metrics.get('val/recall1', float('nan')):.4f}",
                    f"f1={metrics.get('val/f1_1', float('nan')):.4f}",
                ]
            )
        )
        print(
            " | ".join(
                [
                    "val_confusion",
                    f"tn={int(metrics.get('val/tn', 0))}",
                    f"fp={int(metrics.get('val/fp', 0))}",
                    f"fn={int(metrics.get('val/fn', 0))}",
                    f"tp={int(metrics.get('val/tp', 0))}",
                ]
            )
        )


def run_training(config: dict, config_path: Path) -> None:
    data_settings = resolve_data_settings(config)
    features_cfg = get_section(config, "features")
    loader_cfg = get_section(config, "loader")
    train_cfg = get_section(config, "train")
    optim_cfg = get_section(config, "optim")
    metrics_cfg = get_section(config, "metrics")
    ckpt_cfg = get_section(config, "checkpoints")
    wandb_cfg = get_section(config, "wandb")

    seed = train_cfg.get("seed", 1337)
    if seed is not None:
        torch.manual_seed(seed)
    device = choose_device(train_cfg.get("device", "auto"))
    configure_device(device)

    params = params_from_config(features_cfg)
    (
        batch_size,
        num_workers,
        pin_memory,
        persistent_workers,
        prefetch_factor,
        drop_last,
        balanced_sampler,
        allow_augment_workers,
    ) = (
        loader_cfg.get("batch_size", 256),
        loader_cfg.get("num_workers", 8),
        loader_cfg.get("pin_memory", True),
        loader_cfg.get("persistent_workers", True),
        loader_cfg.get("prefetch_factor", 4),
        loader_cfg.get("drop_last", True),
        loader_cfg.get("balanced_sampler", True),
        loader_cfg.get("allow_augment_workers", False),
    )
    augment_enabled = bool(
        data_settings.augment_cfg and data_settings.augment_cfg.get("enabled", False)
    )
    if augment_enabled and num_workers > 0:
        if allow_augment_workers:
            print(
                f"[info] augmentations enabled; keeping num_workers={num_workers} "
                "(allow_augment_workers=true)."
            )
        else:
            print(
                "[warn] augmentations enabled; forcing num_workers=0 to avoid audio backend crashes."
            )
            num_workers = 0
            persistent_workers = False
            prefetch_factor = 2

    epochs = train_cfg.get("epochs", 10)
    clip_grad_norm = train_cfg.get("clip_grad_norm", 1.0)
    class_weights_cfg = train_cfg.get("class_weights")
    metrics_every = metrics_cfg.get("every", 1)
    threshold_cfg = metrics_cfg.get("threshold")
    save_best = ckpt_cfg.get("save_best", True)
    save_last = ckpt_cfg.get("save_last", True)
    best_metric = ckpt_cfg.get("metric", "val_loss")
    ckpt_dir = resolve_path(ROOT, ckpt_cfg.get("dir", DEFAULT_CKPT_DIR))
    lr = optim_cfg.get("lr", 3e-4)
    weight_decay = optim_cfg.get("weight_decay", 1e-2)

    train_dataset = build_dataset(
        manifest=data_settings.manifest,
        split="train",
        cache_dir=data_settings.train_cache_dir,
        params=params,
        augment_cfg=data_settings.augment_cfg,
        augment_root=data_settings.augment_root,
    )
    labels = None
    label_counts = None
    if balanced_sampler:
        labels, label_counts = _labels_and_counts(train_dataset.rows)
    elif isinstance(class_weights_cfg, str):
        label_counts = _label_counts(train_dataset.rows)

    sampler = None
    shuffle = True
    if balanced_sampler:
        sampler = _build_balanced_sampler(labels, label_counts)
        shuffle = False

    train_dataset, train_loader = build_split_loader(
        dataset=train_dataset,
        manifest=data_settings.manifest,
        split="train",
        cache_dir=data_settings.train_cache_dir,
        params=params,
        augment_cfg=data_settings.augment_cfg,
        augment_root=data_settings.augment_root,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        drop_last=drop_last,
        sampler=sampler,
        shuffle=shuffle,
    )
    val_loader = None
    try:
        _, val_loader = build_split_loader(
            manifest=data_settings.manifest,
            split="val",
            cache_dir=data_settings.eval_cache_dir,
            params=params,
            augment_cfg=data_settings.augment_cfg,
            augment_root=data_settings.augment_root,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            drop_last=False,
            sampler=None,
            shuffle=False,
        )
        if len(val_loader) == 0:
            val_loader = None
    except Exception:
        val_loader = None

    model = BaselineCNN().to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    class_weights = None
    if class_weights_cfg:
        if isinstance(class_weights_cfg, str):
            mode = class_weights_cfg.lower()
            if label_counts is None:
                label_counts = _label_counts(train_dataset.rows)
            class_weights = _compute_class_weights_from_counts(label_counts, mode=mode)
            if class_weights is None:
                print("[warn] failed to infer class weights; skipping.")
        elif (
            isinstance(class_weights_cfg, (list, tuple)) and len(class_weights_cfg) == 2
        ):
            class_weights = torch.tensor(
                [float(class_weights_cfg[0]), float(class_weights_cfg[1])],
                dtype=torch.float32,
            )
        else:
            raise ValueError(
                "train.class_weights must be 'balanced', 'inverse', or [w0, w1]."
            )
    if class_weights is not None:
        class_weights = class_weights.to(device)
        print(f"[info] using class_weights: {class_weights.tolist()}")
    crit = nn.CrossEntropyLoss(weight=class_weights)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    wandb_run = init_wandb(wandb_cfg, config, config_path)
    best_metric_key = None
    best_metric_value = None

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            crit=crit,
            opt=opt,
            scaler=scaler,
            device=device,
            epoch=epoch,
            epochs=epochs,
            clip_grad_norm=clip_grad_norm,
        )

        metrics = {
            "train/loss": train_loss,
            "train/acc": train_acc,
            "epoch": epoch,
            "lr": opt.param_groups[0]["lr"],
        }
        should_log = metrics_every > 0 and (
            epoch % metrics_every == 0 or epoch == epochs
        )
        should_eval = val_loader is not None and (
            should_log
            or (save_best and best_metric.replace("/", "_").startswith("val_"))
        )
        if should_eval:
            val_metrics = evaluate(
                model=model,
                loader=val_loader,
                crit=crit,
                device=device,
                epoch=epoch,
                epochs=epochs,
                threshold_cfg=threshold_cfg,
            )
            for key, value in val_metrics.items():
                metrics[f"val/{key}"] = value

        if should_log:
            log_epoch_metrics(metrics, epoch, epochs)
            if wandb_run is not None:
                wandb_run.log(metrics, step=epoch)

        if save_best or save_last:
            checkpoint = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": opt.state_dict(),
                "scaler_state": scaler.state_dict(),
                "metrics": metrics,
                "config_path": str(config_path),
            }

            if save_last:
                save_checkpoint(ckpt_dir / "model_last.pt", checkpoint)

            if save_best:
                metric_key, metric_value = resolve_metric(metrics, best_metric)
                if is_better_metric(metric_key, metric_value, best_metric_value):
                    best_metric_key = metric_key
                    best_metric_value = metric_value
                    checkpoint["best_metric"] = {
                        "name": best_metric_key,
                        "value": best_metric_value,
                    }
                    save_checkpoint(ckpt_dir / "model_best.pt", checkpoint)

    if wandb_run is not None:
        wandb_run.finish()
