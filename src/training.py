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

import hashlib
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
    monitor_mode: Optional[str] = None,
) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
    """Build the learning-rate schedule described by the configuration.

    ReduceLROnPlateau is stepped with config.MONITOR_METRIC, so its direction
    is taken from config.MONITOR_MODE rather than declared separately. The two
    describe the same quantity: letting them be set independently is what
    allowed a schedule watching for a *decrease* to be fed a metric that grows,
    halving the learning rate every time the model improved.

    Args:
        optimizer: Optimiser the schedule drives.
        scheduler_config: Settings. Defaults to config.SCHEDULER_CONFIG.
        monitor_mode: Direction of the monitored metric. Defaults to
            config.MONITOR_MODE.

    Returns:
        The scheduler, or None when no schedule is requested.

    Raises:
        ValueError: If the scheduler type is unknown, or the direction is not
            "max" or "min".
    """
    if scheduler_config is None:
        scheduler_config = config.SCHEDULER_CONFIG
    if monitor_mode is None:
        monitor_mode = config.MONITOR_MODE

    if monitor_mode not in ("max", "min"):
        raise ValueError(f"monitor_mode must be 'max' or 'min', got '{monitor_mode}'")

    scheduler_type = scheduler_config["type"]

    if scheduler_type == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=monitor_mode,
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
    elapsed_seconds: float = 0.0,
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
        elapsed_seconds: Training time accumulated so far, across resumes, so
            the reported duration of a run interrupted twice is still its total.
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
            "elapsed_seconds": elapsed_seconds,
            "architecture": config.ARCHITECTURE,
            "base_channels": config.MODEL_BASE_CHANNELS.get(config.ARCHITECTURE),
            "input_channels": config.INPUT_CHANNELS,
            "num_classes": config.NUM_CLASSES,
        },
        path,
    )


def find_resume_point(run_id: str, experiments_dir: Optional[Path] = None) -> Optional[Path]:
    """Return the checkpoint a run should be resumed from, if there is one.

    Args:
        run_id: Identifier of the run.
        experiments_dir: Folder holding the runs. Defaults to
            config.EXPERIMENTS_DIR.

    Returns:
        Path of the resumable checkpoint, or None if the run has none.
    """
    if experiments_dir is None:
        experiments_dir = config.EXPERIMENTS_DIR

    candidate = experiments_dir / run_id / "checkpoints" / LAST_CHECKPOINT_FILENAME

    return candidate if candidate.exists() else None


def _save_history(
    run_root: Path,
    run_id: str,
    tracker: MetricsTracker,
    best_epoch: Optional[int],
    best_value: Optional[float],
    elapsed_seconds: float,
    stopped_early: bool,
    completed: bool,
    resumed_from_epoch: Optional[int],
) -> None:
    """Write the metrics history, overwriting the previous version.

    Called after every epoch rather than once at the end. It weighs a few
    kilobytes against the checkpoint's tens of megabytes, and it is the only
    place the numbers of an interrupted run survive in readable form.

    Args:
        run_root: Folder of the run.
        run_id: Identifier of the run.
        tracker: History so far.
        best_epoch: Epoch with the best monitored value.
        best_value: That value.
        elapsed_seconds: Training time accumulated across resumes.
        stopped_early: Whether early stopping triggered.
        completed: Whether the loop reached its last epoch.
        resumed_from_epoch: Epoch this invocation started at, when resuming.
    """
    payload = {
        "run_id": run_id,
        "epochs_run": len(tracker.history["val"]),
        "completed": completed,
        "stopped_early": stopped_early,
        "resumed_from_epoch": resumed_from_epoch,
        "best_epoch": best_epoch,
        "best_value": best_value,
        "monitor_metric": config.MONITOR_METRIC,
        "total_seconds": elapsed_seconds,
        "history": tracker.history,
    }

    (run_root / HISTORY_FILENAME).write_text(
        json.dumps(payload, indent=2, default=float), encoding="utf-8"
    )


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    device: Optional[torch.device] = None,
    expected_architecture: Optional[str] = None,
) -> Dict[str, Any]:
    """Restore a checkpoint into the objects given.

    Args:
        path: Checkpoint file.
        model: Model to restore the weights into.
        optimizer: Optimiser to restore, or None to skip.
        scheduler: Scheduler to restore, or None to skip.
        device: Device to map the tensors onto. Defaults to config.DEVICE.
        expected_architecture: Architecture the model was built as. Defaults to
            config.ARCHITECTURE, which is what a training run wants. Evaluation
            compares runs of several architectures, so it passes the one the
            run itself recorded.

    Returns:
        The checkpoint dictionary, so the caller can read the epoch and history.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the checkpoint was produced by a different architecture.
    """
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    if expected_architecture is None:
        expected_architecture = config.ARCHITECTURE

    checkpoint = torch.load(
        path, map_location=config.DEVICE if device is None else device, weights_only=False
    )

    stored_architecture = checkpoint.get("architecture")
    if stored_architecture is not None and stored_architecture != expected_architecture:
        raise ValueError(
            f"Checkpoint holds a '{stored_architecture}' but the model given is a "
            f"'{expected_architecture}'. Loading it would silently mismatch the weights."
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
    num_epochs: Optional[int] = None,
    device: Optional[torch.device] = None,
    track_cldice: bool = True,
    resume_from: Optional[Path] = None,
    checkpoint_interval: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[MetricsTracker, Path]:
    """Train a model, keeping the best checkpoint by the monitored metric.

    Each epoch runs training then validation, records both and updates the
    schedule. Two artefacts are written on different schedules, because they
    cost three orders of magnitude apart:

        history.json  every epoch, a few kilobytes — the numbers of an
                      interrupted run survive in readable form
        last.pt       every checkpoint_interval epochs and at the end, tens of
                      megabytes — the point training resumes from

    The best checkpoint is written whenever the monitored metric improves,
    whatever the interval.

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
        run_id: Identifier of the run. Generated when omitted, or taken from
            the checkpoint's folder when resuming, so a resumed run keeps
            writing where it started.
        num_epochs: Epoch to stop at, counted from the start of the run and not
            from the resume point: resuming a run interrupted at 31 with
            num_epochs=70 trains epochs 32 to 70.
        device: Device to train on. Defaults to config.DEVICE.
        track_cldice: Whether clDice is computed on validation each epoch.
        resume_from: Checkpoint to continue from, as returned by
            find_resume_point(). Restores the weights, the optimiser and
            scheduler state, the history, the best value and the epoch counter,
            so the run continues as though it had never stopped.
        checkpoint_interval: How often the resumable checkpoint is rewritten.
        verbose: Whether progress is printed.

    Returns:
        Tuple of the history and the path of the best checkpoint.
    """
    if num_epochs is None:
        num_epochs = config.NUM_EPOCHS
    if checkpoint_interval is None:
        checkpoint_interval = config.CHECKPOINT_INTERVAL

    if device is None:
        device = config.DEVICE
    if optimizer is None:
        optimizer = create_optimizer(model)
    if scheduler is None:
        scheduler = create_scheduler(optimizer)

    tracker = MetricsTracker()
    monitor_metric = config.MONITOR_METRIC
    monitor_mode = config.MONITOR_MODE
    best_value: Optional[float] = None
    best_epoch: Optional[int] = None
    start_epoch = 1
    elapsed_before = 0.0
    resumed_from_epoch: Optional[int] = None

    if resume_from is not None:
        checkpoint = load_checkpoint(resume_from, model, optimizer, scheduler, device=device)
        tracker.history = checkpoint["history"]
        best_value = checkpoint.get("best_value")
        best_epoch = checkpoint.get("best_epoch")
        elapsed_before = float(checkpoint.get("elapsed_seconds", 0.0))
        start_epoch = int(checkpoint["epoch"]) + 1
        resumed_from_epoch = start_epoch
        # The run continues in the folder it started in, so its history and its
        # checkpoints stay together.
        if run_id is None:
            run_id = resume_from.parent.parent.name

    if run_id is None:
        run_id = config.build_run_id()

    run_paths = config.get_run_dir(run_id)
    checkpoints_dir = run_paths["checkpoints"]
    config_path = run_paths["root"] / CONFIG_FILENAME

    # Written before the first epoch: a run interrupted after five minutes
    # still leaves behind what it was trying to do. On a resume the original is
    # left in place — it describes the settings the earlier epochs actually
    # used, which is what makes the run interpretable.
    if not config_path.exists():
        # build_config returns a deep copy, so adding to it here cannot reach
        # back into the config module.
        run_config = config.build_config(run_id)
        run_config["model"]["parameters"] = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        # "enabled" says what was configured; "applied" says what actually ran.
        run_config["skeleton_loss"]["applied"] = skeleton_criterion is not None
        run_config["git_commit"] = git_commit_hash()
        run_config["code_fingerprint"] = code_fingerprint()
        config_path.write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    best_checkpoint = checkpoints_dir / BEST_CHECKPOINT_FILENAME
    last_checkpoint = checkpoints_dir / LAST_CHECKPOINT_FILENAME
    skeleton_weight = config.SKELETON_LOSS_CONFIG["weight"]
    early_stopping = config.EARLY_STOPPING_CONFIG
    stopped_early = False

    if verbose:
        _print_run_header(run_id, model, num_epochs, skeleton_criterion, device)
        if resumed_from_epoch is not None:
            best = "none yet" if best_value is None else f"{best_value:.4f} at epoch {best_epoch}"
            print(f"[RESUME] Continuing at epoch {resumed_from_epoch}, "
                  f"{len(tracker.history['val'])} epoch(s) already recorded, best {best}.")
            print("=" * 70)

    if start_epoch > num_epochs:
        if verbose:
            print(f"[INFO] Nothing to do: the run already reached epoch "
                  f"{start_epoch - 1} of {num_epochs}.")
        return tracker, best_checkpoint

    training_started = time.perf_counter()
    epoch = start_epoch - 1

    for epoch in range(start_epoch, num_epochs + 1):
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

        elapsed = elapsed_before + (time.perf_counter() - training_started)

        if improved:
            best_value, best_epoch = current, epoch
            save_checkpoint(
                best_checkpoint, model, optimizer, scheduler, epoch, tracker,
                best_value, best_epoch, elapsed,
            )

        if verbose:
            _print_epoch_line(epoch, num_epochs, train_metrics, val_metrics, monitor_metric, improved)

        stopped_early = early_stopping["enabled"] and tracker.check_early_stopping(
            patience=early_stopping["patience"],
            metric=monitor_metric,
            mode=monitor_mode,
            min_delta=early_stopping["min_delta"],
        )
        is_last_epoch = epoch == num_epochs or stopped_early

        # The history is small enough to rewrite every epoch, and it is the
        # only readable record an interrupted run leaves behind.
        _save_history(
            run_root=run_paths["root"], run_id=run_id, tracker=tracker,
            best_epoch=best_epoch, best_value=best_value, elapsed_seconds=elapsed,
            stopped_early=stopped_early, completed=is_last_epoch,
            resumed_from_epoch=resumed_from_epoch,
        )

        # The resumable checkpoint is not: it carries the weights and the
        # optimiser state, so it is rewritten on an interval — and always at
        # the end, so the final state is never a stale multiple of it.
        if epoch % checkpoint_interval == 0 or is_last_epoch:
            save_checkpoint(
                last_checkpoint, model, optimizer, scheduler, epoch, tracker,
                best_value, best_epoch, elapsed,
            )

        if stopped_early:
            if verbose:
                print(f"\n[STOP] {monitor_metric} has not improved for "
                      f"{early_stopping['patience']} epochs.")
            break

    total_seconds = elapsed_before + (time.perf_counter() - training_started)

    if verbose:
        print()
        # best_value stays None only if every epoch produced NaN for the
        # monitored metric — a broken run, but one that should be reported
        # rather than crash the cell after the training already finished.
        if best_value is None:
            print(f"[WARNING] Finished in {total_seconds / 60:.1f} min, but "
                  f"{monitor_metric} was never defined: no checkpoint was kept as best.")
        else:
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


def code_fingerprint() -> str:
    """Return a hash of the source the run is executing.

    git_commit_hash() answers "unknown" on Colab, where the project is a copy
    of the Drive folder and carries no .git directory — so every run of the
    campaign was recorded without any link to a version of the code. That is
    not survivable when the modules change between runs, which they have.

    Hashing the sources themselves needs no repository and no discipline from
    the caller: two runs share a fingerprint exactly when they ran the same
    code. The value can be matched back to a commit afterwards by computing it
    again on a checkout.

    Returns:
        First 8 hex characters of the SHA-256 over every module in src/, taken
        in name order, or "unknown" if the sources cannot be read.
    """
    digest = hashlib.sha256()

    try:
        for path in sorted(Path(__file__).resolve().parent.glob("*.py")):
            # The name matters as much as the content: renaming a module is a
            # change to the code that ran.
            digest.update(path.name.encode("utf-8"))
            digest.update(path.read_bytes())
    except OSError:
        return "unknown"

    return digest.hexdigest()[:8]


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
        "code_fingerprint": code_fingerprint(),
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
