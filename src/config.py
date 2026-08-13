"""Central configuration for the Vine Segmentation (ToF) project.

Every notebook imports this module, so paths, hyperparameters and the compute
device stay consistent across the whole pipeline. Nothing here is executed
automatically except the definition of constants: side effects such as seeding
or folder creation must be triggered explicitly by the caller.

Typical usage inside a notebook:

    import config
    from config import PATHS, DEVICE, CLASS_NAMES

    config.print_environment_info()
    config.set_seed(config.RANDOM_SEED)
    config.create_project_structure()
"""

import importlib
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

# =====================================================
# 1 - ENVIRONMENT
# =====================================================


def detect_platform() -> str:
    """Identify the runtime environment.

    Returns:
        One of "colab", "kaggle" or "local".
    """
    if Path("/content").exists():
        return "colab"
    if Path("/kaggle").exists():
        return "kaggle"
    return "local"


PLATFORM: str = detect_platform()

# BASE_PATH is inferred from this file's position (src/config.py -> project root),
# so the project can be moved or cloned without editing any path by hand.
# Set the VINE_PROJECT_ROOT environment variable to override it (useful on Kaggle,
# where the source code may be mounted read-only under /kaggle/input).
BASE_PATH: Path = Path(
    os.environ.get("VINE_PROJECT_ROOT", Path(__file__).resolve().parents[1])
)

DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _library_version(module_name: str) -> str:
    """Return the version of a library without failing if it is missing.

    Optional dependencies are reported rather than raising, so a notebook that
    does not need them can still print the environment summary.

    Args:
        module_name: Importable name of the library.

    Returns:
        The version string, or "not installed".
    """
    try:
        return importlib.import_module(module_name).__version__
    except ImportError:
        return "not installed"


def print_environment_info() -> None:
    """Print platform, device and library versions for reproducibility."""
    separator = "=" * 70

    print(separator)
    print("ENVIRONMENT INFO")
    print(separator)
    print(f"Platform:       {PLATFORM}")
    print(f"Base path:      {BASE_PATH}")
    print(f"Device:         {DEVICE}")

    if torch.cuda.is_available():
        total_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU:            {torch.cuda.get_device_name(0)}")
        print(f"CUDA version:   {torch.version.cuda}")
        print(f"GPU memory:     {total_memory_gb:.2f} GB")

    print(separator)
    print(f"PyTorch:        {torch.__version__}")
    print(f"NumPy:          {np.__version__}")
    # Optional dependencies: OpenCV is needed from notebook 01 on, albumentations
    # only for the online augmentation used during training.
    print(f"OpenCV:         {_library_version('cv2')}")
    print(f"Albumentations: {_library_version('albumentations')}")
    print(separator)


# =====================================================
# 2 - PROJECT PATHS
# =====================================================

DATA_DIR: Path = BASE_PATH / "data"
EXPERIMENTS_DIR: Path = BASE_PATH / "experiments"
RESULTS_DIR: Path = BASE_PATH / "results"

# Single CSV indexing every training run, used to compare configurations.
RESULTS_CSV: Path = RESULTS_DIR / "experiments.csv"

# Class weights are computed once during EDA (notebook 02) and reloaded by the
# training notebook. Storing them on disk keeps the notebooks independent.
CLASS_WEIGHTS_JSON: Path = DATA_DIR / "processed" / "class_weights.json"

PATHS: Dict[str, Any] = {
    "base": BASE_PATH,
    "raw": {
        "intensity": DATA_DIR / "raw" / "intensity",
        "depth": DATA_DIR / "raw" / "depth",
        "confidence": DATA_DIR / "raw" / "confidence",
        "annotations": DATA_DIR / "raw" / "annotations",
    },
    "processed": {
        "labels_multi": DATA_DIR / "processed" / "labels" / "multiclass",
        "labels_binary": DATA_DIR / "processed" / "labels" / "cordon_binary",
        "colored_multi": DATA_DIR / "processed" / "colored_masks" / "multiclass",
        "colored_binary": DATA_DIR / "processed" / "colored_masks" / "cordon_binary",
        "skeleton": DATA_DIR / "processed" / "skeleton",
        "colored_skeleton": DATA_DIR / "processed" / "colored_masks" / "skeleton_overlay",
    },
    # Train/val/test file lists, so the split is reproducible across notebooks.
    "splits": DATA_DIR / "splits",
    "augmented": {
        split: {
            "intensity": DATA_DIR / "augmented" / split / "intensity",
            "depth": DATA_DIR / "augmented" / split / "depth",
            "confidence": DATA_DIR / "augmented" / split / "confidence",
            "labels_multi": DATA_DIR / "augmented" / split / "labels_multi",
            "labels_binary": DATA_DIR / "augmented" / split / "labels_binary",
            "skeleton": DATA_DIR / "augmented" / split / "skeleton",
        }
        for split in ("train", "val", "test")
    },
    "experiments": EXPERIMENTS_DIR,
    "results": RESULTS_DIR,
}


def _collect_paths(structure: Any) -> List[Path]:
    """Recursively collect every Path contained in a nested dictionary.

    Args:
        structure: Nested dictionary potentially containing Path objects.

    Returns:
        Flat list of all Path objects found.
    """
    collected: List[Path] = []

    if isinstance(structure, Path):
        collected.append(structure)
    elif isinstance(structure, dict):
        for value in structure.values():
            collected.extend(_collect_paths(value))

    return collected


def create_project_structure(verbose: bool = True) -> None:
    """Create every folder declared in PATHS if it does not already exist.

    Safe to call repeatedly: existing folders are left untouched.

    Args:
        verbose: If True, print the list of folders that were created.
    """
    created: List[Path] = []

    for path in _collect_paths(PATHS):
        if not path.exists():
            created.append(path)
        path.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Project structure ready at: {BASE_PATH}")
        if created:
            print(f"Created {len(created)} new folder(s):")
            for path in created:
                print(f"  + {path.relative_to(BASE_PATH)}")
        else:
            print("All folders already existed.")


# =====================================================
# 3 - INPUT CONFIGURATION
# =====================================================

USE_INTENSITY: bool = True
USE_DEPTH: bool = True
USE_CONFIDENCE: bool = True

# Number of channels contributed by each modality. Depth is stored as a
# 3-channel colorised image, hence 3 instead of 1.
MODALITY_CHANNELS: Dict[str, int] = {
    "intensity": 1,
    "depth": 3,
    "confidence": 1,
}

ENABLED_MODALITIES: Dict[str, bool] = {
    "intensity": USE_INTENSITY,
    "depth": USE_DEPTH,
    "confidence": USE_CONFIDENCE,
}

INPUT_CHANNELS: int = sum(
    MODALITY_CHANNELS[modality]
    for modality, is_enabled in ENABLED_MODALITIES.items()
    if is_enabled
)

# =====================================================
# 4 - CLASS INFORMATION
# =====================================================

CLASS_NAMES: List[str] = ["Background", "Cane", "Cordon", "Node", "Other"]
CLASS_COLORS: List[tuple] = [
    (0, 0, 0),      # Background
    (0, 128, 0),    # Cane
    (128, 0, 0),    # Cordon
    (128, 128, 0),  # Node
    (0, 0, 128),    # Other
]
BINARY_COLORS: List[tuple] = [(0, 0, 0), (255, 0, 0)]  # Background, Cordon

# Derived from CLASS_NAMES so the two can never fall out of sync.
NUM_CLASSES: int = len(CLASS_NAMES)

# =====================================================
# 5 - ANNOTATION PARSING
# =====================================================
# How the LabelMe annotations map onto the canonical classes above.
# Editing these constants is enough to adapt the pipeline to a new
# annotation convention: no function needs to be modified.

# Alias -> canonical class name. Lets annotators use Italian labels and
# lets several source labels be merged into a single class.
LABEL_ALIASES: Dict[str, str] = {
    "tralcio": "cane",
    "tralcio_secco": "cane",
    "cordone": "cordon",
    "curva": "cordon",
    "nodo": "node",
    "altro": "other",
}

# Labels present in the JSON files but deliberately excluded from the masks.
# They are skipped silently, without being reported as unknown labels.
LABELS_TO_IGNORE: List[str] = ["ferro", "bud"]

# Drawing order, from lowest to highest priority: later shapes overwrite
# earlier ones, so the last class in the list wins on overlapping pixels.
PRIORITY_MULTICLASS: List[str] = ["cordon", "node", "other", "cane"]
PRIORITY_BINARY: List[str] = ["cane", "other", "node", "cordon"]

# Classes whose linestrip annotations stay thin lines (later dilated) instead
# of being closed into filled polygons. Canes are annotated as open curves.
LINESTRIP_AS_LINE_CLASSES: List[str] = ["cane"]

# Kernel size used to dilate those thin lines so they gain area. Must be odd.
LINESTRIP_DILATION_KERNEL: int = 3

# =====================================================
# 6 - TRAINING CONFIGURATION
# =====================================================

RANDOM_SEED: int = 42
BATCH_SIZE: int = 4
IMG_SIZE: tuple = (512, 512)  # (height, width)
NUM_EPOCHS: int = 70
NUM_WORKERS: int = 2  # 0 while debugging, 2-4 for training

# Dataset split ratios; the test ratio is the remainder.
TRAIN_RATIO: float = 0.70
VAL_RATIO: float = 0.15

# Offline augmentation applied to the training split only.
# Available options: "flipH", "flipV", "flipHV". Empty list disables it.
OFFLINE_AUGMENTATIONS: List[str] = ["flipH", "flipV", "flipHV"]

# Normalisation strategy.
#   True  -> the 3 depth channels use ImageNet statistics, intensity and
#            confidence use dataset statistics. Use with pre-trained encoders.
#   False -> every channel uses statistics computed from the dataset.
#            Use with CustomUNet and SegNet, which train from scratch.
USE_IMAGENET_NORM_FOR_RGB: bool = False

# Metric driving checkpoint saving and early stopping.
MONITOR_METRIC: str = "mean_iou_no_bg"
MONITOR_MODE: str = "max"  # "max" or "min"

EARLY_STOPPING_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "patience": 20,
    "min_delta": 1e-4,
}

# =====================================================
# 7 - MODEL CONFIGURATION
# =====================================================

# Architecture family: ["UNet", "UNetPlusPlus", "SegNet", "PSPNet"]
ARCHITECTURE: str = "UNet"

# Encoder: "custom" for architectures trained from scratch, or an SMP encoder
# name such as "resnet34" / "efficientnet-b0" for pre-trained backbones.
ENCODER: str = "custom"

# Pre-trained encoder weights: "imagenet" or None.
ENCODER_WEIGHTS: Optional[str] = None

# =====================================================
# 8 - CLASS WEIGHTING
# =====================================================

CLASS_WEIGHT_CONFIG: Dict[str, Any] = {
    # ["sqrt_inverse", "inverse", "median_freq", "effective_num", "none"]
    "method": "sqrt_inverse",
    "normalize": True,
    "beta": 0.9999,        # Used only when method="effective_num"
    "apply_weights": True,  # Whether the loss actually uses the weights
}


def load_class_weights() -> Optional[torch.Tensor]:
    """Load the class weights computed during EDA (notebook 02).

    Returns:
        A tensor of shape (NUM_CLASSES,) on DEVICE, or None if the file does
        not exist yet or weighting is disabled in CLASS_WEIGHT_CONFIG.
    """
    if not CLASS_WEIGHT_CONFIG["apply_weights"]:
        return None

    if not CLASS_WEIGHTS_JSON.exists():
        print(
            f"[WARNING] Class weights file not found: {CLASS_WEIGHTS_JSON}\n"
            "          Run notebook 02 (EDA) first, or set "
            "CLASS_WEIGHT_CONFIG['apply_weights'] = False."
        )
        return None

    with open(CLASS_WEIGHTS_JSON, "r", encoding="utf-8") as file:
        payload = json.load(file)

    return torch.tensor(payload["weights"], dtype=torch.float32, device=DEVICE)


def save_class_weights(weights: np.ndarray, method: str) -> None:
    """Persist the class weights so the training notebook can reload them.

    Args:
        weights: Array of shape (NUM_CLASSES,) with one weight per class.
        method: Name of the weighting method used to compute them.
    """
    CLASS_WEIGHTS_JSON.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "method": method,
        "class_names": CLASS_NAMES,
        "weights": [float(weight) for weight in weights],
        "computed_at": datetime.now().isoformat(timespec="seconds"),
    }

    with open(CLASS_WEIGHTS_JSON, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)

    print(f"[OK] Class weights saved to: {CLASS_WEIGHTS_JSON}")


# =====================================================
# 9 - LOSS, OPTIMIZER AND SCHEDULER
# =====================================================

LOSS_CONFIG: Dict[str, Any] = {
    "type": "combined",         # ["combined", "ce", "dice", "focal"]
    # Used only when type="combined"
    "ce_weight": 0.7,           # Typical: 0.5-0.8
    "dice_weight": 0.3,         # Typical: 0.2-0.5
    "use_weighted_dice": True,  # Usually True for imbalanced segmentation
    # Used only when type="focal"
    "focal_gamma": 2.0,         # Typical: 1.5-3.0
}

OPTIMIZER_CONFIG: Dict[str, Any] = {
    "type": "adamw",      # ["adamw", "adam", "sgd"]
    "lr": 1e-4,           # Typical: AdamW 1e-4 to 1e-3, SGD 1e-2 to 1e-1
    "weight_decay": 1e-4,  # Typical: 1e-5 to 1e-2
    "momentum": 0.9,      # Used only when type="sgd"
}

SCHEDULER_CONFIG: Dict[str, Any] = {
    "type": "reduce_on_plateau",  # ["reduce_on_plateau", "cosine", "step", "none"]
    # Used only for ReduceLROnPlateau
    "mode": "min",
    "factor": 0.5,
    "patience": 5,
    "min_lr": 1e-7,
    # Used only for CosineAnnealingLR
    "t_max": NUM_EPOCHS,
    # Used only for StepLR
    "step_size": 10,
    "gamma": 0.1,
}

SKELETON_LOSS_CONFIG: Dict[str, Any] = {
    "enabled": False,       # Whether to add the skeleton recall loss
    "load_skeleton": True,  # Skeletons are always loaded for clDice and plots
    # total_loss = main_loss + weight * skeleton_loss
    "weight": 0.2,
    "dilation_kernel_size": 2,
    "skeleton_classes": [CLASS_NAMES.index("Cane"), CLASS_NAMES.index("Cordon")],
}

# =====================================================
# 10 - RUN IDENTIFICATION
# =====================================================


def build_run_id(suffix: Optional[str] = None) -> str:
    """Build a unique, self-describing identifier for a training run.

    The identifier is used as the experiment folder name and as the primary
    key in the results CSV, e.g. "20260807_1432_UNet-custom_combined_adamw".

    Args:
        suffix: Optional short tag appended at the end, e.g. "lr3e-4".

    Returns:
        The run identifier string.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    parts = [
        timestamp,
        f"{ARCHITECTURE}-{ENCODER}",
        LOSS_CONFIG["type"],
        OPTIMIZER_CONFIG["type"],
    ]

    if suffix:
        parts.append(suffix)

    return "_".join(parts)


def get_run_dir(run_id: str, create: bool = True) -> Dict[str, Path]:
    """Return (and optionally create) the folder layout of a single run.

    Args:
        run_id: Identifier produced by build_run_id().
        create: If True, create the folders on disk.

    Returns:
        Dictionary with the "root", "checkpoints" and "figures" paths.
    """
    root = EXPERIMENTS_DIR / run_id
    run_paths = {
        "root": root,
        "checkpoints": root / "checkpoints",
        "figures": root / "figures",
    }

    if create:
        for path in run_paths.values():
            path.mkdir(parents=True, exist_ok=True)

    return run_paths


def build_config(run_id: str) -> Dict[str, Any]:
    """Assemble the full configuration of a run into a serialisable dictionary.

    Saved as config.json inside the run folder, it is the complete record of
    the settings used, complementing the summary row written to the CSV.

    Args:
        run_id: Identifier produced by build_run_id().

    Returns:
        Nested dictionary describing the entire configuration.
    """
    return {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input": {
            "use_intensity": USE_INTENSITY,
            "use_depth": USE_DEPTH,
            "use_confidence": USE_CONFIDENCE,
            "modality_channels": MODALITY_CHANNELS,
            "input_channels": INPUT_CHANNELS,
        },
        "model": {
            "architecture": ARCHITECTURE,
            "encoder": ENCODER,
            "encoder_weights": ENCODER_WEIGHTS,
            "num_classes": NUM_CLASSES,
            "use_imagenet_norm_for_rgb": USE_IMAGENET_NORM_FOR_RGB,
        },
        "training": {
            "batch_size": BATCH_SIZE,
            "img_size": list(IMG_SIZE),
            "num_epochs": NUM_EPOCHS,
            "train_ratio": TRAIN_RATIO,
            "val_ratio": VAL_RATIO,
            "offline_augmentations": OFFLINE_AUGMENTATIONS,
            "monitor_metric": MONITOR_METRIC,
            "monitor_mode": MONITOR_MODE,
            "early_stopping": EARLY_STOPPING_CONFIG,
        },
        "class_info": {
            "class_names": CLASS_NAMES,
            "class_colors": [list(color) for color in CLASS_COLORS],
        },
        "class_weights": CLASS_WEIGHT_CONFIG,
        "loss": LOSS_CONFIG,
        "optimizer": OPTIMIZER_CONFIG,
        "scheduler": SCHEDULER_CONFIG,
        "skeleton_loss": SKELETON_LOSS_CONFIG,
        "reproducibility": {"random_seed": RANDOM_SEED},
    }


# =====================================================
# 11 - REPRODUCIBILITY
# =====================================================


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Seed every random number generator used in the project.

    Args:
        seed: Seed value applied to Python, NumPy and PyTorch.

    Note:
        cuDNN determinism is disabled to prioritise speed, so results are
        reproducible only up to small GPU non-determinism. Never change the
        seed between runs you intend to compare.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    os.environ["PYTHONHASHSEED"] = str(seed)

    print(f"[OK] Random seed set to {seed} (cuDNN deterministic: False)")


def seed_worker(worker_id: int) -> None:
    """Seed a DataLoader worker process so augmentations are reproducible.

    Passed to DataLoader via the worker_init_fn argument.

    Args:
        worker_id: Worker index assigned by the DataLoader.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
