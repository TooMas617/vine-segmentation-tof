"""Segmentation metrics, accumulated over a whole split.

    * ConfusionMatrix -> one matrix per epoch, every overlap metric from it
    * compute_cldice  -> centreline Dice, on exact skeletons
    * MetricsTracker  -> history across epochs, early stopping, best epoch

Metrics are accumulated, not averaged. Scoring each batch and averaging the
scores is not the same measurement: a batch holding fifty pixels of a class
would weigh as much as one holding five thousand, and a batch where the class
is absent produces no score at all, so the average ends up conditioned on the
batches where the class happens to appear. With Node absent from a quarter of
the images, that bias is not small. Summing the counts first and dividing once
at the end gives the value over the split.

The same choice removes a cost: one integer histogram per batch on the device,
instead of a per-class Python loop with a host transfer each time.

Typical usage from notebook 03:

    from metrics import ConfusionMatrix, MetricsTracker

    matrix = ConfusionMatrix(device=config.DEVICE)
    for images, labels, skeletons in dataloader:
        matrix.update(outputs.argmax(dim=1), labels)
    epoch_metrics = matrix.compute()
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from skimage.morphology import skeletonize

import config

# Guards divisions whose denominator is a pixel count, zero for absent classes.
DEFAULT_EPSILON: float = 1e-6


# =====================================================
# 1 - CONFUSION MATRIX
# =====================================================


class ConfusionMatrix:
    """Accumulate predictions against targets, then derive every metric.

    Entry (i, j) counts the pixels whose true class is i and whose predicted
    class is j. Everything overlap-based follows from it: per class, the
    diagonal entry is the true positives, its column sum minus the diagonal the
    false positives, its row sum minus the diagonal the false negatives.

    A class absent from both the targets and the predictions has no meaningful
    score and comes back as NaN, so it is excluded from the means rather than
    counted as a perfect or a failed prediction.
    """

    def __init__(
        self,
        num_classes: int = config.NUM_CLASSES,
        class_names: Optional[Sequence[str]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """Prepare an empty matrix.

        Args:
            num_classes: Number of segmentation classes.
            class_names: Names used when reporting. Defaults to
                config.CLASS_NAMES.
            device: Device the matrix lives on. Keeping it beside the
                predictions avoids a host transfer per batch.
        """
        self.num_classes = num_classes
        self.class_names = list(config.CLASS_NAMES if class_names is None else class_names)

        if len(self.class_names) != num_classes:
            raise ValueError(
                f"Expected {num_classes} class names, got {len(self.class_names)}"
            )

        self.device = config.DEVICE if device is None else device
        self.reset()

    def reset(self) -> None:
        """Empty the matrix, to start a new epoch."""
        self.matrix = torch.zeros(
            (self.num_classes, self.num_classes), dtype=torch.int64, device=self.device
        )

    def update(self, pred_labels: torch.Tensor, targets: torch.Tensor) -> None:
        """Add one batch to the matrix.

        Args:
            pred_labels: Predicted class indices, shape (B, H, W). Pass
                logits.argmax(dim=1), not the logits.
            targets: Ground truth class indices, same shape.

        Raises:
            ValueError: If the shapes differ or a label falls outside the
                declared classes.
        """
        if pred_labels.shape != targets.shape:
            raise ValueError(
                f"Predictions and targets must have the same shape, got "
                f"{tuple(pred_labels.shape)} and {tuple(targets.shape)}"
            )

        pred_labels = pred_labels.detach().to(self.device).reshape(-1)
        targets = targets.detach().to(self.device).reshape(-1)

        if int(pred_labels.max()) >= self.num_classes or int(targets.max()) >= self.num_classes:
            raise ValueError(
                f"Labels outside the {self.num_classes} declared classes: "
                f"prediction max {int(pred_labels.max())}, target max {int(targets.max())}"
            )

        # One flat histogram over the pairs, reshaped into the matrix: a single
        # kernel launch, no synchronisation with the host.
        pair_index = targets * self.num_classes + pred_labels
        counts = torch.bincount(pair_index, minlength=self.num_classes**2)

        self.matrix += counts.reshape(self.num_classes, self.num_classes)

    def compute(self) -> Dict[str, Any]:
        """Derive every metric from the accumulated counts.

        Returns:
            Dictionary with the per-class arrays "class_iou", "class_dice",
            "class_precision", "class_recall" and "class_accuracy", their means
            "mean_iou", "mean_iou_no_bg", "mean_dice", "mean_precision",
            "mean_recall" and "mean_class_accuracy", and "pixel_accuracy".
            Per-class entries are NaN where the class appears in neither the
            targets nor the predictions.
        """
        matrix = self.matrix.double().cpu().numpy()

        true_positives = np.diag(matrix)
        predicted = matrix.sum(axis=0)
        actual = matrix.sum(axis=1)

        false_positives = predicted - true_positives
        false_negatives = actual - true_positives
        union = true_positives + false_positives + false_negatives

        # A class nobody predicted and nobody annotated has no score to give.
        absent = union == 0

        with np.errstate(invalid="ignore", divide="ignore"):
            class_iou = np.where(absent, np.nan, true_positives / union)
            class_dice = np.where(
                absent, np.nan, 2 * true_positives / (2 * true_positives + false_positives + false_negatives)
            )
            class_precision = np.where(
                predicted == 0, np.nan, true_positives / np.maximum(predicted, DEFAULT_EPSILON)
            )
            class_recall = np.where(
                actual == 0, np.nan, true_positives / np.maximum(actual, DEFAULT_EPSILON)
            )

        total = matrix.sum()

        return {
            "pixel_accuracy": float(true_positives.sum() / total) if total else float("nan"),
            "mean_iou": _nanmean(class_iou),
            "mean_iou_no_bg": _nanmean(class_iou[1:]),
            "mean_dice": _nanmean(class_dice),
            "mean_dice_no_bg": _nanmean(class_dice[1:]),
            "mean_precision": _nanmean(class_precision),
            "mean_recall": _nanmean(class_recall),
            "mean_class_accuracy": _nanmean(class_recall),
            "class_iou": class_iou.tolist(),
            "class_dice": class_dice.tolist(),
            "class_precision": class_precision.tolist(),
            "class_recall": class_recall.tolist(),
            "class_accuracy": class_recall.tolist(),
        }

    def print_report(self, title: str = "METRICS") -> None:
        """Print the per-class table and the summary lines.

        Args:
            title: Heading shown above the table.
        """
        metrics = self.compute()
        separator = "=" * 70

        print(separator)
        print(title)
        print(separator)
        print(f"  {'class':<12}{'IoU':>10}{'Dice':>10}{'precision':>12}{'recall':>10}")

        for index, name in enumerate(self.class_names):
            print(
                f"  {name:<12}"
                f"{_format_metric(metrics['class_iou'][index]):>10}"
                f"{_format_metric(metrics['class_dice'][index]):>10}"
                f"{_format_metric(metrics['class_precision'][index]):>12}"
                f"{_format_metric(metrics['class_recall'][index]):>10}"
            )

        print()
        print(f"  pixel accuracy   {metrics['pixel_accuracy']:.4f}")
        print(f"  mean IoU         {metrics['mean_iou']:.4f}")
        print(f"  mean IoU (no bg) {metrics['mean_iou_no_bg']:.4f}")
        print(separator)


def _nanmean(values: np.ndarray) -> float:
    """Average the entries that are not NaN, returning NaN if none are.

    np.nanmean warns on an all-NaN slice; here that case simply means no class
    was scored, which is information rather than an anomaly.

    Args:
        values: Array possibly containing NaN.

    Returns:
        The mean of the defined entries, or NaN.
    """
    finite = values[~np.isnan(values)]

    return float(finite.mean()) if finite.size else float("nan")


def _format_metric(value: float) -> str:
    """Render a metric for the report, marking undefined entries.

    Args:
        value: Metric value, possibly NaN.

    Returns:
        Four decimals, or a dash when the class was absent.
    """
    return "  -" if np.isnan(value) else f"{value:.4f}"


# =====================================================
# 2 - CENTRELINE DICE
# =====================================================


def exact_skeleton(mask: np.ndarray) -> np.ndarray:
    """Extract the one-pixel-wide skeleton of a binary mask.

    No dilation. clDice is defined on the morphological skeleton, and a
    thickened one turns the measure back into an area overlap — losing the
    connectivity information it exists to capture. The tubed skeleton written
    by notebook 01 belongs to the loss, where the extra pixels stabilise the
    signal, not to this metric.

    Args:
        mask: Binary mask of shape (H, W).

    Returns:
        Binary skeleton of the same shape.
    """
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=np.uint8)

    return skeletonize(mask.astype(bool)).astype(np.uint8)


def compute_cldice(
    pred_labels: np.ndarray,
    target_labels: np.ndarray,
    skeleton_classes: Optional[Sequence[int]] = None,
    epsilon: float = DEFAULT_EPSILON,
) -> Dict[str, Any]:
    """Compute the centreline Dice of every skeleton class.

    Two quantities per class, from the reference definition:

        topology precision = |skeleton(pred) & target| / |skeleton(pred)|
        topology recall    = |skeleton(target) & pred| / |skeleton(target)|
        clDice             = harmonic mean of the two

    The first is sensitive to false positives, the second to false negatives.
    Both skeletons are extracted here, from the masks as they are, so the two
    sides of the comparison are built the same way.

    Skeletonisation runs on the CPU and is not cheap, so this is meant for the
    final evaluation rather than for every epoch.

    Args:
        pred_labels: Predicted class indices, shape (N, H, W) or (H, W).
        target_labels: Ground truth class indices, same shape.
        skeleton_classes: Class indices to score. Defaults to
            config.SKELETON_LOSS_CONFIG.
        epsilon: Constant guarding the divisions.

    Returns:
        Dictionary with "cldice_per_class", "precision_per_class",
        "recall_per_class" — each keyed by class name — and "mean_cldice".

    Raises:
        ValueError: If the two arrays do not have the same shape.
    """
    if skeleton_classes is None:
        skeleton_classes = config.SKELETON_LOSS_CONFIG["skeleton_classes"]

    if pred_labels.shape != target_labels.shape:
        raise ValueError(
            f"Predictions and targets must have the same shape, got "
            f"{pred_labels.shape} and {target_labels.shape}"
        )

    if pred_labels.ndim == 2:
        pred_labels = pred_labels[np.newaxis]
        target_labels = target_labels[np.newaxis]

    cldice_per_class: Dict[str, float] = {}
    precision_per_class: Dict[str, float] = {}
    recall_per_class: Dict[str, float] = {}

    for class_index in skeleton_classes:
        class_name = config.CLASS_NAMES[class_index]

        # Accumulated over the whole set, for the same reason the confusion
        # matrix is: a per-image average would weigh a sliver of cordon as much
        # as a full one.
        precision_intersection = precision_total = 0.0
        recall_intersection = recall_total = 0.0

        for index in range(pred_labels.shape[0]):
            pred_mask = (pred_labels[index] == class_index).astype(np.uint8)
            target_mask = (target_labels[index] == class_index).astype(np.uint8)

            pred_skeleton = exact_skeleton(pred_mask)
            target_skeleton = exact_skeleton(target_mask)

            precision_intersection += float((pred_skeleton & target_mask).sum())
            precision_total += float(pred_skeleton.sum())
            recall_intersection += float((target_skeleton & pred_mask).sum())
            recall_total += float(target_skeleton.sum())

        if precision_total == 0 and recall_total == 0:
            cldice_per_class[class_name] = float("nan")
            precision_per_class[class_name] = float("nan")
            recall_per_class[class_name] = float("nan")
            continue

        topology_precision = precision_intersection / (precision_total + epsilon)
        topology_recall = recall_intersection / (recall_total + epsilon)

        precision_per_class[class_name] = topology_precision
        recall_per_class[class_name] = topology_recall
        cldice_per_class[class_name] = (
            2 * topology_precision * topology_recall
            / (topology_precision + topology_recall + epsilon)
        )

    return {
        "cldice_per_class": cldice_per_class,
        "precision_per_class": precision_per_class,
        "recall_per_class": recall_per_class,
        "mean_cldice": _nanmean(np.asarray(list(cldice_per_class.values()))),
    }


def print_cldice_report(results: Dict[str, Any]) -> None:
    """Print the clDice table.

    Args:
        results: Mapping returned by compute_cldice().
    """
    separator = "=" * 70

    print(separator)
    print("CENTRELINE DICE (exact skeletons)")
    print(separator)
    print(f"  {'class':<12}{'clDice':>10}{'top. prec.':>13}{'top. recall':>13}")

    for class_name, value in results["cldice_per_class"].items():
        print(
            f"  {class_name:<12}"
            f"{_format_metric(value):>10}"
            f"{_format_metric(results['precision_per_class'][class_name]):>13}"
            f"{_format_metric(results['recall_per_class'][class_name]):>13}"
        )

    print()
    print(f"  mean clDice      {results['mean_cldice']:.4f}")
    print(separator)


# =====================================================
# 3 - METRICS TRACKER
# =====================================================


class MetricsTracker:
    """Keep the per-epoch metrics of a run, for early stopping and plots."""

    def __init__(self, class_names: Optional[Sequence[str]] = None) -> None:
        """Prepare an empty history.

        Args:
            class_names: Names used when plotting. Defaults to
                config.CLASS_NAMES.
        """
        self.class_names = list(config.CLASS_NAMES if class_names is None else class_names)
        self.reset()

    def reset(self) -> None:
        """Drop the history of both phases."""
        self.history: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": []}

    def update(self, metrics: Dict[str, Any], phase: str = "train") -> None:
        """Record the metrics of one epoch.

        Args:
            metrics: Values for the epoch, as returned by
                ConfusionMatrix.compute() plus at least a "loss" entry.
            phase: Either "train" or "val".

        Raises:
            ValueError: If the phase is unknown.
        """
        if phase not in self.history:
            raise ValueError(f"Unknown phase '{phase}'. Expected 'train' or 'val'.")

        self.history[phase].append(dict(metrics))

    def _monitored_values(self, metric: str, phase: str) -> List[float]:
        """Return the recorded values of a metric, dropping undefined epochs.

        Args:
            metric: Metric name.
            phase: Either "train" or "val".

        Returns:
            The values, in epoch order.
        """
        return [
            float(epoch[metric])
            for epoch in self.history[phase]
            if metric in epoch and not np.isnan(epoch[metric])
        ]

    def get_best_epoch(
        self,
        metric: str = config.MONITOR_METRIC,
        phase: str = "val",
        mode: str = config.MONITOR_MODE,
    ) -> Tuple[Optional[int], Optional[float]]:
        """Find the best epoch recorded so far.

        Args:
            metric: Metric to rank by.
            phase: Either "train" or "val".
            mode: "max" if higher is better, "min" otherwise.

        Returns:
            Tuple of the epoch number, counted from 1, and its value. Both None
            if nothing has been recorded.
        """
        values = self._monitored_values(metric, phase)

        if not values:
            return None, None

        best_index = int(np.argmax(values)) if mode == "max" else int(np.argmin(values))

        return best_index + 1, values[best_index]

    def check_early_stopping(
        self,
        patience: int = config.EARLY_STOPPING_CONFIG["patience"],
        metric: str = config.MONITOR_METRIC,
        phase: str = "val",
        mode: str = config.MONITOR_MODE,
        min_delta: float = config.EARLY_STOPPING_CONFIG["min_delta"],
    ) -> bool:
        """Tell whether training should stop.

        Stops when no epoch among the last `patience` improved on the best
        value seen before them by at least `min_delta`.

        Args:
            patience: Number of epochs without improvement that ends the run.
            metric: Metric to monitor.
            phase: Either "train" or "val".
            mode: "max" if higher is better, "min" otherwise.
            min_delta: Improvement below this counts as none.

        Returns:
            True when the criterion is met.
        """
        values = self._monitored_values(metric, phase)

        if len(values) <= patience:
            return False

        earlier, recent = values[:-patience], values[-patience:]

        if mode == "max":
            return all(value <= max(earlier) + min_delta for value in recent)

        return all(value >= min(earlier) - min_delta for value in recent)

    def plot_history(
        self,
        metrics: Sequence[str] = ("loss", "mean_iou_no_bg", "mean_precision", "mean_recall"),
        save_path: Optional[Path] = None,
        figure_size: Tuple[float, float] = (12.0, 7.0),
    ) -> None:
        """Draw the training and validation curves.

        Precision and recall are included by default rather than only the
        overlap scores: a term that rewards recall without penalising false
        positives moves them apart, and the mean IoU alone would hide it.

        Args:
            metrics: Metric names, one panel each.
            save_path: Destination file, or None to only display the figure.
            figure_size: Size of the whole figure in inches.
        """
        columns = 2
        rows = -(-len(metrics) // columns)
        figure, axes = plt.subplots(rows, columns, figsize=figure_size, squeeze=False)
        flat_axes = axes.flatten()

        for position, metric in enumerate(metrics):
            axis = flat_axes[position]

            for phase, colour in (("train", "#2a78d6"), ("val", "#eb6834")):
                values = [
                    epoch.get(metric, float("nan")) for epoch in self.history[phase]
                ]
                if not values:
                    continue
                axis.plot(range(1, len(values) + 1), values, label=phase, color=colour)

            axis.set_title(metric, fontsize=10)
            axis.set_xlabel("epoch")
            axis.grid(color="#d8d7d2", linewidth=0.6)
            axis.set_axisbelow(True)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)

        for empty_axis in flat_axes[len(metrics):]:
            empty_axis.set_visible(False)

        flat_axes[0].legend(frameon=False, fontsize=9)
        figure.tight_layout()

        _finalise_figure(figure, save_path)


def _finalise_figure(figure: Figure, save_path: Optional[Path]) -> None:
    """Save a figure if a destination was given, then display it.

    Args:
        figure: Figure to write and show.
        save_path: Destination file, created together with its parent folder.
    """
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"[OK] Figure saved to: {save_path}")

    plt.show()
