"""Training loop, checkpointing and the experiment log.

    * create_optimizer / create_scheduler -> built from the configuration
    * run_epoch                           -> one pass, training or evaluation
    * train_model                         -> the full loop with early stopping
    * append_result                       -> one row per run in experiments.csv

Every run writes into its own folder under experiments/, and one line into
results/experiments.csv. The line records what was configured as well as what
came out, because a metric without the settings that produced it cannot be
compared with anything.

Typical usage from notebook 03:

    from training import train_model, append_result

    tracker, best_checkpoint = train_model(model, dataloaders, criterion)
"""

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from metrics import ConfusionMatrix, MetricsTracker, compute_cldice

# Written inside every run folder, so a checkpoint is never orphaned from the
# settings that produced it.
CONFIG_FILENAME: str = "config.json"
HISTORY_FILENAME: str = "history.json"
BEST_CHECKPOINT_FILENAME: str = "best.pt"
LAST_CHECKPOINT_FILENAME: str = "last.pt"


# =====================================================
# 1 - OPTIMIZER AND SCHEDULER
# =====================================================


def create_optimizer(
    model: nn.Module,
    optimizer_config: Optional[Dict[str, Any]] = None,
) -> torch.optim.Optimizer:
    """Build the optimiser described by the configuration.

    Args:
        model: Model whose parameters are optimised.
        optimizer_config: Settings. Defaults to config.OPTIMIZER_CONFIG.

    Returns:
        The optimiser.

    Raises:
        ValueError: If the optimiser type is unknown.
    """
    if optimizer_config is None:
        optimizer_config = config.OPTIMIZER_CONFIG

    optimizer_type = optimizer_config["type"]
    parameters = model.parameters()
    learning_rate = optimizer_config["lr"]
    weight_decay = optimizer_config["weight_decay"]

    if optimizer_type == "adamw":
        return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)

    if optimizer_type == "adam":
        return torch.optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)

    if optimizer_type == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=learning_rate,
            momentum=optimizer_config["momentum"],
            weight_decay=weight_decay,
        )

    raise ValueError(
        f"Unknown optimizer '{optimizer_type}'. Available: 'adamw', 'adam', 'sgd'"
    )


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_config: Optional[Dict[str, Any]] = None,
) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
    """Build the learning-rate schedule described by the configuration.

    Args:
        optimizer: Optimiser the schedule drives.
        scheduler_config: Settings. Defaults to config.SCHEDULER_CONFIG.

    Returns:
        The scheduler, or None when no schedule is requested.

    Raises:
        ValueError: If the scheduler type is unknown.
    """
    if scheduler_config is None:
        scheduler_config = config.SCHEDULER_CONFIG

    scheduler_type = scheduler_config["type"]

    if scheduler_type == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_config["mode"],
            factor=scheduler_config["factor"],
            patience=scheduler_config["patience"],
            min_lr=scheduler_config["min_lr"],
        )

    if scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=scheduler_config["t_max"], eta_min=scheduler_config["min_lr"]
        )

    if scheduler_type == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=scheduler_config["step_size"], gamma=scheduler_config["gamma"]
        )

    if scheduler_type == "none":
        return None

    raise ValueError(
        f"Unknown scheduler '{scheduler_type}'. "
        "Available: 'reduce_on_plateau', 'cosine', 'step', 'none'"
    )


# =====================================================
# 2 - CHECKPOINTS
# =====================================================


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    epoch: int,
    tracker: MetricsTracker,
    best_value: Optional[float],
    best_epoch: Optional[int],
) -> None:
    """Write everything needed to resume or to evaluate.

    The optimiser and scheduler states are included because resuming without
    them restarts the momentum and the schedule, which is not the same run.

    Args:
        path: Destination file.
        model: Model to save.
        optimizer: Optimiser whose state is saved.
        scheduler: Scheduler whose state is saved, if any.
        epoch: Epoch just finished, counted from 1.
        tracker: History of the run so far.
        best_value: Best value of the monitored metric so far.
        best_epoch: Epoch that produced it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
            "history": tracker.history,
            "best_value": best_value,
            "best_epoch": best_epoch,
            "architecture": config.ARCHITECTURE,
            "base_channels": config.MODEL_BASE_CHANNELS.get(config.ARCHITECTURE),
            "input_channels": config.INPUT_CHANNELS,
            "num_classes": config.NUM_CLASSES,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Restore a checkpoint into the objects given.

    Args:
        path: Checkpoint file.
        model: Model to restore the weights into.
        optimizer: Optimiser to restore, or None to skip.
        scheduler: Scheduler to restore, or None to skip.
        device: Device to map the tensors onto. Defaults to config.DEVICE.

    Returns:
        The checkpoint dictionary, so the caller can read the epoch and history.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the checkpoint was produced by a different architecture.
    """
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(
        path, map_location=config.DEVICE if device is None else device, weights_only=False
    )

    stored_architecture = checkpoint.get("architecture")
    if stored_architecture is not None and stored_architecture != config.ARCHITECTURE:
        raise ValueError(
            f"Checkpoint holds a '{stored_architecture}' but config.ARCHITECTURE "
            f"is '{config.ARCHITECTURE}'. Loading it would silently mismatch the weights."
        )

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint


# =====================================================
# 3 - ONE EPOCH
# =====================================================


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    skeleton_criterion: Optional[nn.Module] = None,
    skeleton_weight: float = 0.0,
    compute_centreline_dice: bool = False,
    description: str = "",
) -> Dict[str, Any]:
    """Run one pass over a dataloader, training or evaluating.

    Passing an optimiser makes it a training pass; leaving it out makes it an
    evaluation pass, with gradients disabled and the model in eval mode. Both
    share the same body so the two can never drift apart in what they measure.

    Metrics come from a confusion matrix accumulated over the whole pass rather
    than averaged per batch — see the module docstring of metrics.py.

    Args:
        model: Model to run.
        dataloader: Data to run it over.
        criterion: Main loss.
        device: Device to run on.
        optimizer: Optimiser. None means evaluation.
        skeleton_criterion: Skeleton recall loss, or None to leave it out.
        skeleton_weight: Weight of the skeleton term in the total.
        compute_centreline_dice: Whether to also compute clDice. It needs the
            predictions of the whole pass in memory and skeletonises on the
            CPU, so it is meant for validation, not for the training split.
        description: Label shown on the progress bar.

    Returns:
        The metrics of the pass, including "loss", "main_loss",
        "skeleton_loss", everything ConfusionMatrix.compute() returns, and the
        clDice entries when requested.
    """
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    matrix = ConfusionMatrix(device=device)
    total_loss = total_main = total_skeleton = 0.0
    batch_count = 0

    collected_predictions: List[np.ndarray] = []
    collected_targets: List[np.ndarray] = []

    progress = tqdm(dataloader, desc=description, leave=False)

    with torch.set_grad_enabled(is_training):
        for images, labels, skeletons in progress:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            skeletons = skeletons.to(device, non_blocking=True)

            logits = model(images)
            main_loss = criterion(logits, labels)

            if skeleton_criterion is not None:
                skeleton_loss = skeleton_criterion(logits, skeletons)
                loss = main_loss + skeleton_weight * skeleton_loss
            else:
                skeleton_loss = torch.zeros((), device=device)
                loss = main_loss

            if is_training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            predictions = logits.detach().argmax(dim=1)
            matrix.update(predictions, labels)

            if compute_centreline_dice:
                collected_predictions.append(predictions.cpu().numpy())
                collected_targets.append(labels.cpu().numpy())

            total_loss += loss.item()
            total_main += main_loss.item()
            total_skeleton += skeleton_loss.item()
            batch_count += 1

            progress.set_postfix(loss=f"{loss.item():.4f}")

    metrics: Dict[str, Any] = {
        "loss": total_loss / max(batch_count, 1),
        "main_loss": total_main / max(batch_count, 1),
        "skeleton_loss": total_skeleton / max(batch_count, 1),
    }
    metrics.update(matrix.compute())

    if compute_centreline_dice and collected_predictions:
        cldice = compute_cldice(
            np.concatenate(collected_predictions, axis=0),
            np.concatenate(collected_targets, axis=0),
        )
        metrics["mean_cldice"] = cldice["mean_cldice"]
        metrics["cldice_per_class"] = cldice["cldice_per_class"]
        metrics["cldice_precision_per_class"] = cldice["precision_per_class"]
        metrics["cldice_recall_per_class"] = cldice["recall_per_class"]

    return metrics


# =====================================================
# 4 - TRAINING LOOP
# =====================================================


def train_model(
    model: nn.Module,
    dataloaders: Dict[str, DataLoader],
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    skeleton_criterion: Optional[nn.Module] = None,
    run_id: Optional[str] = None,
    num_epochs: int = config.NUM_EPOCHS,
    device: Optional[torch.device] = None,
    track_cldice: bool = True,
    verbose: bool = True,
) -> Tuple[MetricsTracker, Path]:
    """Train a model, keeping the best checkpoint by the monitored metric.

    Each epoch runs training then validation, records both, updates the
    schedule and saves a checkpoint whenever the monitored metric improves. The
    last epoch is saved as well, so an interrupted run can resume.

    Args:
        model: Model to train, already on the device.
        dataloaders: Mapping with at least "train" and "val".
        criterion: Main loss.
        optimizer: Optimiser. Built from the configuration when omitted.
        scheduler: Learning-rate schedule. Built from the configuration when
            omitted.
        skeleton_criterion: Skeleton recall loss, or None. Running once with
            and once without is the ablation described in section 7.6 of the
            notes.
        run_id: Identifier of the run. Generated when omitted.
        num_epochs: Maximum number of epochs.
        device: Device to train on. Defaults to config.DEVICE.
        track_cldice: Whether clDice is computed on validation each epoch.
        verbose: Whether progress is printed.

    Returns:
        Tuple of the history and the path of the best checkpoint.
    """
    if device is None:
        device = config.DEVICE
    if run_id is None:
        run_id = config.build_run_id()
    if optimizer is None:
        optimizer = create_optimizer(model)
    if scheduler is None:
        scheduler = create_scheduler(optimizer)

    run_paths = config.get_run_dir(run_id)
    checkpoints_dir = run_paths["checkpoints"]

    # The settings are written before the first epoch: a run interrupted after
    # five minutes still leaves behind what it was trying to do.
    run_config = config.build_config(run_id)
    run_config["model"]["base_channels"] = config.MODEL_BASE_CHANNELS.get(config.ARCHITECTURE)
    run_config["model"]["parameters"] = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    run_config["skeleton_loss"]["applied"] = skeleton_criterion is not None
    run_config["git_commit"] = git_commit_hash()
    (run_paths["root"] / CONFIG_FILENAME).write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )

    tracker = MetricsTracker()
    monitor_metric = config.MONITOR_METRIC
    monitor_mode = config.MONITOR_MODE
    best_value: Optional[float] = None
    best_epoch: Optional[int] = None
    best_checkpoint = checkpoints_dir / BEST_CHECKPOINT_FILENAME
    skeleton_weight = config.SKELETON_LOSS_CONFIG["weight"]
    early_stopping = config.EARLY_STOPPING_CONFIG
    stopped_early = False

    if verbose:
        _print_run_header(run_id, model, num_epochs, skeleton_criterion, device)

    training_started = time.perf_counter()

    for epoch in range(1, num_epochs + 1):
        epoch_started = time.perf_counter()

        train_metrics = run_epoch(
            model=model,
            dataloader=dataloaders["train"],
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            skeleton_criterion=skeleton_criterion,
            skeleton_weight=skeleton_weight,
            description=f"epoch {epoch}/{num_epochs} train",
        )

        val_metrics = run_epoch(
            model=model,
            dataloader=dataloaders["val"],
            criterion=criterion,
            device=device,
            skeleton_criterion=skeleton_criterion,
            skeleton_weight=skeleton_weight,
            compute_centreline_dice=track_cldice,
            description=f"epoch {epoch}/{num_epochs} val",
        )

        train_metrics["epoch_seconds"] = time.perf_counter() - epoch_started
        train_metrics["learning_rate"] = optimizer.param_groups[0]["lr"]
        val_metrics["epoch_seconds"] = train_metrics["epoch_seconds"]

        tracker.update(train_metrics, phase="train")
        tracker.update(val_metrics, phase="val")

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # This scheduler needs the quantity it is watching; the others
                # only need to be told an epoch went by.
                scheduler.step(val_metrics[monitor_metric])
            else:
                scheduler.step()

        current = val_metrics.get(monitor_metric, float("nan"))
        improved = _is_improvement(current, best_value, monitor_mode)

        if improved:
            best_value, best_epoch = current, epoch
            save_checkpoint(
                best_checkpoint, model, optimizer, scheduler, epoch, tracker, best_value, best_epoch
            )

        save_checkpoint(
            checkpoints_dir / LAST_CHECKPOINT_FILENAME,
            model, optimizer, scheduler, epoch, tracker, best_value, best_epoch,
        )

        if verbose:
            _print_epoch_line(epoch, num_epochs, train_metrics, val_metrics, monitor_metric, improved)

        if early_stopping["enabled"] and tracker.check_early_stopping(
            patience=early_stopping["patience"],
            metric=monitor_metric,
            mode=monitor_mode,
            min_delta=early_stopping["min_delta"],
        ):
            stopped_early = True
            if verbose:
                print(f"\n[STOP] {monitor_metric} has not improved for "
                      f"{early_stopping['patience']} epochs.")
            break

    total_seconds = time.perf_counter() - training_started

    history_payload = {
        "run_id": run_id,
        "epochs_run": len(tracker.history["val"]),
        "stopped_early": stopped_early,
        "best_epoch": best_epoch,
        "best_value": best_value,
        "monitor_metric": monitor_metric,
        "total_seconds": total_seconds,
        "history": tracker.history,
    }
    (run_paths["root"] / HISTORY_FILENAME).write_text(
        json.dumps(history_payload, indent=2, default=float), encoding="utf-8"
    )

    if verbose:
        print()
        print(f"[OK] Finished in {total_seconds / 60:.1f} min. "
              f"Best {monitor_metric} {best_value:.4f} at epoch {best_epoch}.")
        print(f"     Run folder: {run_paths['root']}")

    return tracker, best_checkpoint


def _is_improvement(current: float, best: Optional[float], mode: str) -> bool:
    """Tell whether a value beats the best seen so far.

    Args:
        current: Value just measured.
        best: Best value so far, or None if none yet.
        mode: "max" if higher is better, "min" otherwise.

    Returns:
        True when the value is an improvement. A NaN never is.
    """
    if np.isnan(current):
        return False
    if best is None:
        return True

    return current > best if mode == "max" else current < best


def _print_run_header(
    run_id: str,
    model: nn.Module,
    num_epochs: int,
    skeleton_criterion: Optional[nn.Module],
    device: torch.device,
) -> None:
    """Print what is about to run.

    Args:
        run_id: Identifier of the run.
        model: Model being trained.
        num_epochs: Maximum number of epochs.
        skeleton_criterion: Skeleton loss, or None.
        device: Device in use.
    """
    separator = "=" * 70
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(separator)
    print("TRAINING")
    print(separator)
    print(f"Run id:        {run_id}")
    print(f"Architecture:  {config.ARCHITECTURE} "
          f"(base {config.MODEL_BASE_CHANNELS.get(config.ARCHITECTURE)}, {parameters:,} parameters)")
    print(f"Loss:          {config.LOSS_CONFIG['type']}")
    print(f"Skeleton loss: "
          f"{'on, weight ' + str(config.SKELETON_LOSS_CONFIG['weight']) if skeleton_criterion else 'off'}")
    print(f"Optimizer:     {config.OPTIMIZER_CONFIG['type']} lr={config.OPTIMIZER_CONFIG['lr']}")
    print(f"Monitoring:    {config.MONITOR_METRIC} ({config.MONITOR_MODE})")
    print(f"Epochs:        up to {num_epochs}")
    print(f"Device:        {device}")
    print(separator)


def _print_epoch_line(
    epoch: int,
    num_epochs: int,
    train_metrics: Dict[str, Any],
    val_metrics: Dict[str, Any],
    monitor_metric: str,
    improved: bool,
) -> None:
    """Print one line summarising an epoch.

    Precision and recall are shown alongside IoU because a loss that rewards
    recall without penalising false positives moves them apart, and IoU alone
    would hide it.

    Args:
        epoch: Epoch just finished.
        num_epochs: Maximum number of epochs.
        train_metrics: Metrics of the training pass.
        val_metrics: Metrics of the validation pass.
        monitor_metric: Metric driving checkpoint saving.
        improved: Whether this epoch produced the best value so far.
    """
    cldice = val_metrics.get("mean_cldice")
    cldice_text = "" if cldice is None or np.isnan(cldice) else f"  clDice {cldice:.4f}"

    print(
        f"  {epoch:>3}/{num_epochs}  "
        f"loss {train_metrics['loss']:.4f}/{val_metrics['loss']:.4f}  "
        f"{monitor_metric} {val_metrics.get(monitor_metric, float('nan')):.4f}  "
        f"P {val_metrics['mean_precision']:.3f}  R {val_metrics['mean_recall']:.3f}"
        f"{cldice_text}  "
        f"{train_metrics['epoch_seconds']:.0f}s"
        f"{'  *' if improved else ''}"
    )


# =====================================================
# 5 - EXPERIMENT LOG
# =====================================================


def git_commit_hash() -> str:
    """Return the short hash of the commit the code is at.

    Recorded with every run because the modules have changed substantially and
    will change again: a metric produced two weeks ago is only interpretable
    next to the code that produced it. A dirty working tree is marked, since
    the hash alone would then be misleading.

    Returns:
        The short hash, with "-dirty" appended when there are uncommitted
        changes, or "unknown" outside a repository.
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=config.BASE_PATH,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()

        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=config.BASE_PATH,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()

        return f"{commit}-dirty" if dirty else commit

    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return "unknown"


def build_result_row(
    run_id: str,
    tracker: MetricsTracker,
    test_metrics: Optional[Dict[str, Any]] = None,
    splits_hash: Optional[str] = None,
    parameters: Optional[int] = None,
    skeleton_applied: Optional[bool] = None,
    notes: str = "",
) -> Dict[str, Any]:
    """Assemble the row describing one run.

    Everything is flattened into one level: a CSV read back into a dataframe is
    only sortable and filterable if the columns are scalars.

    Args:
        run_id: Identifier of the run.
        tracker: History of the run.
        test_metrics: Metrics on the test split, as returned by run_epoch()
            plus optionally the clDice entries. Omitted for a run not yet
            evaluated.
        splits_hash: Fingerprint of the partition used.
        parameters: Trainable parameter count of the model.
        skeleton_applied: Whether the skeleton loss actually contributed.
        notes: Free text, for whatever distinguishes this run by hand.

    Returns:
        A flat dictionary, one entry per column.
    """
    best_epoch, best_value = tracker.get_best_epoch()
    val_history = tracker.history["val"]
    last_val = val_history[-1] if val_history else {}

    row: Dict[str, Any] = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit_hash(),
        "splits_hash": splits_hash,
        # --- configuration ---
        "architecture": config.ARCHITECTURE,
        "base_channels": config.MODEL_BASE_CHANNELS.get(config.ARCHITECTURE),
        "parameters": parameters,
        "input_channels": config.INPUT_CHANNELS,
        "img_size": f"{config.IMG_SIZE[0]}x{config.IMG_SIZE[1]}",
        "batch_size": config.BATCH_SIZE,
        "loss_type": config.LOSS_CONFIG["type"],
        "ce_weight": config.LOSS_CONFIG["ce_weight"],
        "dice_weight": config.LOSS_CONFIG["dice_weight"],
        "class_weight_method": config.CLASS_WEIGHT_CONFIG["method"],
        "class_weights_applied": config.CLASS_WEIGHT_CONFIG["apply_weights"],
        "skeleton_loss_applied": skeleton_applied,
        "skeleton_loss_weight": config.SKELETON_LOSS_CONFIG["weight"],
        "skeleton_kernel": config.SKELETON_LOSS_CONFIG["dilation_kernel_size"],
        "optimizer": config.OPTIMIZER_CONFIG["type"],
        "learning_rate": config.OPTIMIZER_CONFIG["lr"],
        "weight_decay": config.OPTIMIZER_CONFIG["weight_decay"],
        "scheduler": config.SCHEDULER_CONFIG["type"],
        # --- course of the run ---
        "epochs_run": len(val_history),
        "best_epoch": best_epoch,
        "monitor_metric": config.MONITOR_METRIC,
        "best_monitor_value": best_value,
        "final_val_loss": last_val.get("loss"),
        "seconds_per_epoch": last_val.get("epoch_seconds"),
        "notes": notes,
    }

    if test_metrics is not None:
        row.update(_flatten_metrics(test_metrics, prefix="test"))

    return row


def _flatten_metrics(metrics: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """Turn a metrics dictionary into scalar columns.

    Per-class lists become one column per class, named after the class, so the
    CSV can be sorted on any of them.

    Args:
        metrics: Metrics as returned by run_epoch().
        prefix: Prefix given to every column.

    Returns:
        A flat dictionary.
    """
    flat: Dict[str, Any] = {}

    for key, value in metrics.items():
        if isinstance(value, (int, float, np.floating, np.integer)):
            flat[f"{prefix}_{key}"] = float(value)
        elif isinstance(value, list) and key.startswith("class_"):
            for class_index, entry in enumerate(value):
                flat[f"{prefix}_{key[len('class_'):]}_{config.CLASS_NAMES[class_index]}"] = entry
        elif isinstance(value, dict):
            for class_name, entry in value.items():
                flat[f"{prefix}_{key.replace('_per_class', '')}_{class_name}"] = entry

    return flat


def append_result(row: Dict[str, Any], results_csv: Optional[Path] = None) -> Path:
    """Append one run to the results table.

    New columns are tolerated: the file is read, concatenated and rewritten, so
    a run recording a metric earlier runs did not have leaves those cells empty
    rather than failing or silently dropping the column.

    Args:
        row: Row as returned by build_result_row().
        results_csv: Destination file. Defaults to config.RESULTS_CSV.

    Returns:
        The path written.
    """
    if results_csv is None:
        results_csv = config.RESULTS_CSV

    results_csv.parent.mkdir(parents=True, exist_ok=True)
    new_row = pd.DataFrame([row])

    if results_csv.exists():
        existing = pd.read_csv(results_csv)
        table = pd.concat([existing, new_row], ignore_index=True)
    else:
        table = new_row

    table.to_csv(results_csv, index=False)
    print(f"[OK] Result appended to: {results_csv} ({len(table)} run(s))")

    return results_csv
