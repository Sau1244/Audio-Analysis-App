from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


def _safe_div(num: float, denom: float) -> float:
    return num / denom if denom else 0.0


def _pr_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if labels.size == 0:
        return float("nan")
    order = np.argsort(-scores)
    y = labels[order]
    total_pos = int((y == 1).sum())
    if total_pos == 0:
        return float("nan")
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / total_pos
    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])
    return float(np.trapz(precision, recall))


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if labels.size == 0:
        return float("nan")
    labels = labels.astype(np.int64)
    pos = labels == 1
    neg = labels == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores)
    sorted_scores = scores[order]
    ranks = np.empty_like(sorted_scores, dtype=np.float64)
    i = 0
    n = sorted_scores.size
    while i < n:
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        ranks[i : j + 1] = avg_rank
        i = j + 1
    ranks_full = np.empty_like(ranks)
    ranks_full[order] = ranks
    rank_sum_pos = ranks_full[pos].sum()
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _eer(labels: np.ndarray, scores: np.ndarray) -> float:
    if labels.size == 0:
        return float("nan")
    labels = labels.astype(np.int64)
    pos = labels == 1
    neg = labels == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels == 1)
    fp = np.cumsum(sorted_labels == 0)
    tpr = tp / max(n_pos, 1)
    fpr = fp / max(n_neg, 1)
    fnr = 1.0 - tpr
    idx = int(np.argmin(np.abs(fpr - fnr)))
    return float(0.5 * (fpr[idx] + fnr[idx]))


def _binary_metrics_from_scores(
    labels: np.ndarray, scores: np.ndarray, threshold: float
) -> dict:
    if labels.size == 0:
        return {}
    labels = labels.astype(np.int64)
    preds = scores >= threshold
    y1 = labels == 1
    y0 = labels == 0
    pred1 = preds
    pred0 = ~preds

    tp = int((pred1 & y1).sum())
    fp = int((pred1 & y0).sum())
    fn = int((pred0 & y1).sum())
    tn = int((pred0 & y0).sum())

    tp0 = int((pred0 & y0).sum())
    fn0 = int((pred1 & y0).sum())
    fp0 = int((pred0 & y1).sum())
    tn0 = int((pred1 & y1).sum())

    recall0 = _safe_div(tp0, tp0 + fn0)
    precision0 = _safe_div(tp0, tp0 + fp0)
    f1_0 = _safe_div(2 * precision0 * recall0, precision0 + recall0)

    recall1 = _safe_div(tp, tp + fn)
    precision1 = _safe_div(tp, tp + fp)
    f1_1 = _safe_div(2 * precision1 * recall1, precision1 + recall1)

    balanced_acc = 0.5 * (recall0 + recall1)
    acc = _safe_div(tp + tn, labels.size)

    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "tp0": tp0,
        "fn0": fn0,
        "fp0": fp0,
        "tn0": tn0,
        "tp1": tp,
        "fn1": fn,
        "fp1": fp,
        "tn1": tn,
        "recall0": recall0,
        "precision0": precision0,
        "f1_0": f1_0,
        "recall1": recall1,
        "precision1": precision1,
        "f1_1": f1_1,
        "balanced_acc": balanced_acc,
        "acc": acc,
    }


def _best_threshold_max_acc(
    labels: np.ndarray, scores: np.ndarray
) -> tuple[float, float]:
    if labels.size == 0:
        return 0.5, float("nan")
    labels = labels.astype(np.int64)
    order = np.argsort(-scores)
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    total_pos = int(sorted_labels.sum())
    total_neg = int(sorted_labels.size - total_pos)

    tp = np.cumsum(sorted_labels == 1)
    fp = np.cumsum(sorted_labels == 0)
    tn = total_neg - fp
    acc = (tp + tn) / max(sorted_labels.size, 1)

    acc0 = total_neg / max(sorted_labels.size, 1)
    best_idx = int(np.argmax(np.concatenate(([acc0], acc))))
    if best_idx == 0:
        threshold = min(float(sorted_scores.max()) + 1e-6, 1.0)
        return threshold, float(acc0)
    threshold = float(sorted_scores[best_idx - 1])
    return threshold, float(acc[best_idx - 1])


@dataclass
class BinaryMetricsAccumulator:
    track_pr_auc: bool = True
    tp0: int = 0
    fn0: int = 0
    fp0: int = 0
    tn0: int = 0
    tp1: int = 0
    fn1: int = 0
    fp1: int = 0
    tn1: int = 0
    _scores: list[torch.Tensor] = field(default_factory=list)
    _labels: list[torch.Tensor] = field(default_factory=list)
    _disabled: bool = False

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        if self._disabled:
            return
        if logits.dim() != 2 or logits.size(1) != 2:
            self._disabled = True
            return
        preds = logits.argmax(dim=1)
        pred0 = preds == 0
        pred1 = preds == 1
        y0 = labels == 0
        y1 = labels == 1
        self.tp0 += (pred0 & y0).sum().item()
        self.fn0 += (pred1 & y0).sum().item()
        self.fp0 += (pred0 & y1).sum().item()
        self.tn0 += (pred1 & y1).sum().item()
        self.tp1 += (pred1 & y1).sum().item()
        self.fn1 += (pred0 & y1).sum().item()
        self.fp1 += (pred1 & y0).sum().item()
        self.tn1 += (pred0 & y0).sum().item()
        if self.track_pr_auc:
            probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu()
            self._scores.append(probs)
            self._labels.append(labels.detach().cpu())

    def compute(self) -> dict:
        if self._disabled:
            return {}
        recall0 = _safe_div(self.tp0, self.tp0 + self.fn0)
        precision0 = _safe_div(self.tp0, self.tp0 + self.fp0)
        f1_0 = _safe_div(2 * precision0 * recall0, precision0 + recall0)
        recall1 = _safe_div(self.tp1, self.tp1 + self.fn1)
        precision1 = _safe_div(self.tp1, self.tp1 + self.fp1)
        f1_1 = _safe_div(2 * precision1 * recall1, precision1 + recall1)
        balanced_acc = 0.5 * (recall0 + recall1)
        if self.track_pr_auc and self._scores and self._labels:
            y_np = torch.cat(self._labels).numpy().astype(np.int64)
            s_np = torch.cat(self._scores).numpy().astype(np.float64)
            pr_auc_1 = _pr_auc(y_np, s_np)
            pr_auc_0 = _pr_auc(1 - y_np, 1 - s_np)
            roc_auc = _roc_auc(y_np, s_np)
            eer = _eer(y_np, s_np)
        else:
            pr_auc_1 = float("nan")
            pr_auc_0 = float("nan")
            roc_auc = float("nan")
            eer = float("nan")
        return {
            "tn": self.tn1,
            "fp": self.fp1,
            "fn": self.fn1,
            "tp": self.tp1,
            "tp0": self.tp0,
            "fn0": self.fn0,
            "fp0": self.fp0,
            "tn0": self.tn0,
            "tp1": self.tp1,
            "fn1": self.fn1,
            "fp1": self.fp1,
            "tn1": self.tn1,
            "recall0": recall0,
            "precision0": precision0,
            "f1_0": f1_0,
            "recall1": recall1,
            "precision1": precision1,
            "f1_1": f1_1,
            "balanced_acc": balanced_acc,
            "pr_auc_1": pr_auc_1,
            "pr_auc_0": pr_auc_0,
            "roc_auc": roc_auc,
            "eer": eer,
        }
