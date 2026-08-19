"""Evaluation of finished runs: predictions, error maps and the comparison.

    * load_run          -> rebuild a model from its run folder
    * predict_split     -> predictions of a whole split, as arrays
    * show_predictions  -> image, truth, prediction and an error map
    * comparison_table  -> the runs side by side, from experiments.csv

The error map is the point of this module. Overlap scores say how much a
prediction is wrong by; they do not say *how*. Colouring false positives apart
from misses shows at a glance whether a model inflates the structures it finds
or loses them — two failures with the same IoU and very different consequences
for pruning.

Typical usage from notebook 04:

    from evaluation import load_run, predict_split, show_predictions

    model, run_config = load_run("20260819_1830_UNet-custom_combined_adamw")
    predictions, targets = predict_split(model, dataloaders["test"])
    show_predictions(predictions, targets, stems)
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib.figure import Figure
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from data_preparation import label_map_to_colored_mask, read_modality_image
from metrics import ConfusionMatrix, compute_cldice
from models import create_model
from training import CONFIG_FILENAME, BEST_CHECKPOINT_FILENAME, load_checkpoint

# Colours of the error map. Chosen so the two kinds of mistake are told apart
# without reading a legend: warm for predicting too much, cool for too little.
ERROR_COLORS: Dict[str, Tuple[int, int, int]] = {
    "correct_background": (0, 0, 0),
    "correct_foreground": (90, 90, 90),
    "false_positive": (214, 45, 45),   # predicted a structure that is not there
    "missed": (45, 110, 214),          # a structure the model did not find
    "wrong_class": (235, 180, 40),     # found, but called something else
}

# Metrics shown by default in the comparison, in the order they are read.
COMPARISON_COLUMNS: Tuple[str, ...] = (
    "architecture",
    "parameters",
    "skeleton_loss_applied",
    "test_mean_iou_no_bg",
    "test_mean_cldice",
    "test_mean_precision",
    "test_mean_recall",
    "epochs_run",
    "best_epoch",
)


# =====================================================
# 1 - LOADING A FINISHED RUN
# =====================================================


def list_runs(experiments_dir: Optional[Path] = None) -> List[str]:
    """List the runs that hold a usable checkpoint.

    Args:
        experiments_dir: Folder holding the runs. Defaults to
            config.EXPERIMENTS_DIR.

    Returns:
        Run identifiers, most recent first — the identifier starts with a
        timestamp, so sorting it is sorting by time.
    """
    if experiments_dir is None:
        experiments_dir = config.EXPERIMENTS_DIR

    if not experiments_dir.exists():
        return []

    runs = [
        path.name
        for path in experiments_dir.iterdir()
        if path.is_dir() and (path / "checkpoints" / BEST_CHECKPOINT_FILENAME).exists()
    ]

    return sorted(runs, reverse=True)


def load_run(
    run_id: str,
    device: Optional[torch.device] = None,
    experiments_dir: Optional[Path] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Rebuild the model of a finished run and load its best weights.

    The architecture and width come from the run's own config.json, not from
    the current config module: the point of comparing runs is that they differ,
    so reading the live configuration would rebuild the wrong network.

    Args:
        run_id: Identifier of the run.
        device: Device to place the model on. Defaults to config.DEVICE.
        experiments_dir: Folder holding the runs.

    Returns:
        Tuple of the model in evaluation mode and the run's configuration.

    Raises:
        FileNotFoundError: If the run folder, its config or its checkpoint is
            missing.
    """
    if device is None:
        device = config.DEVICE
    if experiments_dir is None:
        experiments_dir = config.EXPERIMENTS_DIR

    run_dir = experiments_dir / run_id
    config_path = run_dir / CONFIG_FILENAME
    checkpoint_path = run_dir / "checkpoints" / BEST_CHECKPOINT_FILENAME

    for path in (config_path, checkpoint_path):
        if not path.exists():
            raise FileNotFoundError(f"Run '{run_id}' is missing {path.name}: {path}")

    run_config = json.loads(config_path.read_text(encoding="utf-8"))
    model_config = run_config["model"]

    model = create_model(
        architecture=model_config["architecture"],
        in_channels=run_config["input"]["input_channels"],
        num_classes=model_config["num_classes"],
        base_channels=model_config.get("base_channels"),
    )

    load_checkpoint(
        checkpoint_path,
        model,
        device=device,
        expected_architecture=model_config["architecture"],
    )

    return model.to(device).eval(), run_config


# =====================================================
# 2 - PREDICTIONS
# =====================================================


def predict_split(
    model: nn.Module,
    dataloader: DataLoader,
    device: Optional[torch.device] = None,
    description: str = "Predicting",
) -> Tuple[np.ndarray, np.ndarray]:
    """Run a model over a split and return the predictions and the targets.

    Returned as arrays rather than tensors because everything downstream —
    error maps, clDice, figures — works on the CPU with numpy.

    Args:
        model: Model to run, already on the device.
        dataloader: Split to run over. Must not shuffle, or the predictions
            will not line up with the file stems.
        device: Device to run on. Defaults to config.DEVICE.
        description: Label shown on the progress bar.

    Returns:
        Tuple of predicted and true class indices, both shaped (N, H, W).
    """
    if device is None:
        device = config.DEVICE

    model.eval()
    predictions: List[np.ndarray] = []
    targets: List[np.ndarray] = []

    with torch.no_grad():
        for images, labels, _ in tqdm(dataloader, desc=description, leave=False):
            logits = model(images.to(device, non_blocking=True))
            predictions.append(logits.argmax(dim=1).cpu().numpy().astype(np.uint8))
            targets.append(labels.numpy().astype(np.uint8))

    return np.concatenate(predictions, axis=0), np.concatenate(targets, axis=0)


def evaluate_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Score a set of predictions with every metric at once.

    Args:
        predictions: Predicted class indices, shape (N, H, W).
        targets: True class indices, same shape.
        verbose: If True, print the tables.

    Returns:
        The confusion-matrix metrics with the clDice entries merged in.
    """
    matrix = ConfusionMatrix(device=torch.device("cpu"))
    matrix.update(torch.from_numpy(predictions.astype(np.int64)),
                  torch.from_numpy(targets.astype(np.int64)))

    metrics = matrix.compute()
    cldice = compute_cldice(predictions, targets)
    metrics.update(
        {
            "mean_cldice": cldice["mean_cldice"],
            "cldice_per_class": cldice["cldice_per_class"],
            "cldice_precision_per_class": cldice["precision_per_class"],
            "cldice_recall_per_class": cldice["recall_per_class"],
        }
    )

    if verbose:
        matrix.print_report(title="TEST METRICS")

    return metrics


# =====================================================
# 3 - ERROR MAPS AND PREDICTION FIGURES
# =====================================================


def build_error_map(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Paint each pixel by the kind of mistake made on it.

    Five outcomes, because "wrong" is not one thing:

        correct background   the easy majority, kept dark so it recedes
        correct foreground   a structure found and named right
        false positive       a structure predicted where there is none
        missed               a structure present and not found
        wrong class          found, but assigned to another class

    Separating the second and third is what the overlap scores cannot do. On
    this dataset the expected signature of over-segmentation is a red halo
    tracing the true structures rather than red blobs away from them.

    Args:
        prediction: Predicted class indices, shape (H, W).
        target: True class indices, same shape.

    Returns:
        RGB image of shape (H, W, 3).
    """
    predicted_foreground = prediction > 0
    true_foreground = target > 0
    agree = prediction == target

    error_map = np.zeros((*target.shape, 3), dtype=np.uint8)
    error_map[:] = ERROR_COLORS["correct_background"]

    error_map[agree & true_foreground] = ERROR_COLORS["correct_foreground"]
    error_map[predicted_foreground & ~true_foreground] = ERROR_COLORS["false_positive"]
    error_map[~predicted_foreground & true_foreground] = ERROR_COLORS["missed"]
    error_map[predicted_foreground & true_foreground & ~agree] = ERROR_COLORS["wrong_class"]

    return error_map


def show_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    stems: Sequence[str],
    split_name: str = "test",
    indices: Optional[Sequence[int]] = None,
    n_samples: int = 3,
    save_path: Optional[Path] = None,
) -> None:
    """Display samples as image, truth, prediction and error map.

    Args:
        predictions: Predicted class indices, shape (N, H, W).
        targets: True class indices, same shape.
        stems: File stems in the same order as the predictions.
        split_name: Split the samples come from, used to find the images.
        indices: Which samples to show. Defaults to the first n_samples.
        n_samples: How many to show when indices is not given.
        save_path: Destination file for the figure, or None to only display it.

    Raises:
        ValueError: If the arrays and the stems do not line up.
    """
    if len(stems) != predictions.shape[0]:
        raise ValueError(
            f"{len(stems)} stem(s) for {predictions.shape[0]} prediction(s): the "
            "dataloader must not shuffle, or they no longer correspond."
        )

    if indices is None:
        indices = range(min(n_samples, predictions.shape[0]))

    indices = list(indices)
    intensity_dir = config.PATHS["augmented"][split_name]["intensity"]

    figure, axes = plt.subplots(len(indices), 4, figsize=(15, 3.2 * len(indices)), squeeze=False)

    for row, index in enumerate(indices):
        intensity = read_modality_image(intensity_dir / f"{stems[index]}.png", "intensity")
        panels = (
            (np.repeat(intensity[:, :, np.newaxis], 3, axis=2), "intensity"),
            (label_map_to_colored_mask(targets[index], config.CLASS_COLORS), "ground truth"),
            (label_map_to_colored_mask(predictions[index], config.CLASS_COLORS), "prediction"),
            (build_error_map(predictions[index], targets[index]), "errors"),
        )

        for column, (image, title) in enumerate(panels):
            axis = axes[row][column]
            axis.imshow(image)
            axis.axis("off")
            if row == 0:
                axis.set_title(title, fontsize=10)

    legend = [
        mpatches.Patch(color=np.array(ERROR_COLORS[key]) / 255, label=label)
        for key, label in (
            ("correct_foreground", "correct"),
            ("false_positive", "false positive"),
            ("missed", "missed"),
            ("wrong_class", "wrong class"),
        )
    ]
    figure.legend(handles=legend, loc="lower center", ncol=4, frameon=False, fontsize=9)

    figure.suptitle(f"Predictions on {split_name}", fontsize=11)
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    _finalise_figure(figure, save_path)


def worst_samples(
    predictions: np.ndarray,
    targets: np.ndarray,
    class_index: Optional[int] = None,
    count: int = 3,
) -> List[int]:
    """Find the samples a model does worst on, by per-image IoU.

    Looking at the failures is worth more than looking at the average: they are
    where the systematic weaknesses show, and they are what a thesis figure
    should be honest about.

    Args:
        predictions: Predicted class indices, shape (N, H, W).
        targets: True class indices, same shape.
        class_index: Class to rank by, or None for all foreground together.
        count: How many indices to return.

    Returns:
        Sample indices, worst first. Images where the class is absent from both
        prediction and truth are left out, having no score.
    """
    scores: List[Tuple[float, int]] = []

    for index in range(predictions.shape[0]):
        if class_index is None:
            predicted = predictions[index] > 0
            actual = targets[index] > 0
        else:
            predicted = predictions[index] == class_index
            actual = targets[index] == class_index

        union = int((predicted | actual).sum())
        if union == 0:
            continue

        scores.append((int((predicted & actual).sum()) / union, index))

    scores.sort()

    return [index for _, index in scores[:count]]


# =====================================================
# 4 - COMPARISON ACROSS RUNS
# =====================================================


def load_results(results_csv: Optional[Path] = None) -> pd.DataFrame:
    """Read the results table.

    Args:
        results_csv: File to read. Defaults to config.RESULTS_CSV.

    Returns:
        The table, one row per run.

    Raises:
        FileNotFoundError: If no run has been recorded yet.
    """
    if results_csv is None:
        results_csv = config.RESULTS_CSV

    if not results_csv.exists():
        raise FileNotFoundError(
            f"No results table at {results_csv}. Run notebook 03 at least once."
        )

    return pd.read_csv(results_csv)


def comparison_table(
    results: Optional[pd.DataFrame] = None,
    columns: Sequence[str] = COMPARISON_COLUMNS,
    sort_by: str = "test_mean_iou_no_bg",
    only_current_split: bool = True,
) -> pd.DataFrame:
    """Put the runs side by side on the columns that distinguish them.

    Args:
        results: Table to read. Loaded from disk when omitted.
        columns: Columns to keep, besides the run identifier.
        sort_by: Column to sort on, descending.
        only_current_split: Whether to drop runs measured on another partition.
            Metrics from different splits are not comparable, so mixing them in
            one table invites exactly the wrong conclusion.

    Returns:
        The reduced table, best first.
    """
    if results is None:
        results = load_results()

    table = results.copy()

    if only_current_split and "splits_hash" in table.columns:
        from data_preparation import get_splits_hash

        current = get_splits_hash()
        dropped = int((table["splits_hash"] != current).sum())
        table = table[table["splits_hash"] == current]

        if dropped:
            print(
                f"[INFO] {dropped} run(s) measured on another split are hidden. "
                f"Current split: {current}."
            )

    keep = ["run_id"] + [column for column in columns if column in table.columns]

    if sort_by in table.columns:
        table = table.sort_values(sort_by, ascending=False)

    return table[keep].reset_index(drop=True)


def plot_class_comparison(
    results: Optional[pd.DataFrame] = None,
    metric: str = "iou",
    save_path: Optional[Path] = None,
    figure_size: Tuple[float, float] = (11.0, 4.5),
) -> None:
    """Compare the runs class by class, for one metric.

    The mean hides the classes that matter. A model can lead on mean IoU while
    losing the two thin classes the task is about, and this is the view that
    shows it.

    Args:
        results: Table to read. Loaded from disk when omitted.
        metric: Which per-class metric to plot: "iou", "dice", "precision",
            "recall" or "cldice".
        save_path: Destination file for the figure, or None to only display it.
        figure_size: Size of the figure in inches.

    Raises:
        ValueError: If no column holds that metric.
    """
    if results is None:
        results = load_results()

    columns = [
        (name, f"test_{metric}_{name}")
        for name in config.CLASS_NAMES
        if f"test_{metric}_{name}" in results.columns
    ]

    if not columns:
        raise ValueError(
            f"No per-class column for '{metric}'. Available prefixes: "
            f"{sorted({c.split('_')[1] for c in results.columns if c.startswith('test_')})}"
        )

    figure, axis = plt.subplots(figsize=figure_size)

    positions = np.arange(len(columns))
    run_count = len(results)
    bar_width = 0.8 / max(run_count, 1)

    for offset, (_, run) in enumerate(results.iterrows()):
        label = f"{run['run_id'][:13]} {run.get('architecture', '')}"
        axis.bar(
            positions + (offset - (run_count - 1) / 2) * bar_width,
            [run.get(column, np.nan) for _, column in columns],
            width=bar_width * 0.9,
            label=label,
        )

    axis.set_xticks(positions)
    axis.set_xticklabels([name for name, _ in columns])
    axis.set_ylabel(metric)
    axis.set_title(f"{metric} per class, by run", fontsize=10)
    axis.legend(frameon=False, fontsize=8)
    axis.grid(axis="y", color="#d8d7d2", linewidth=0.6)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

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
