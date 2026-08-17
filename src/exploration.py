"""Exploratory analysis of the dataset prepared by notebook 01.

The module answers the questions that must be settled before any training
starts, and produces the two sets of numbers the training notebook reloads
from disk:

    * dataset inventory      -> is every sample complete, and what format is it
    * class distribution     -> how much of each class the dataset contains
    * channel statistics     -> per-channel mean and std used to normalise
    * class weights          -> derived from the training-set frequencies
    * skeleton coverage      -> how much of a class its tubed skeleton covers

Statistics meant purely to describe the data are reported for every split;
statistics that become training parameters are measured on the training split
alone, so no information from validation or test reaches the model.

This module reads what notebook 01 wrote and never modifies it.

Typical usage from notebook 02:

    from exploration import check_dataset_consistency, describe_image_formats

    check_dataset_consistency()
    describe_image_formats()
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from tqdm import tqdm

import config
from data_preparation import SPLIT_NAMES, get_source_dirs, load_splits, read_modality_image

# Groups every distribution is reported for: the three partitions and the
# dataset as a whole. Descriptive figures cover all of them; the numbers that
# become training parameters are taken from "train" alone.
DISTRIBUTION_GROUPS: Tuple[str, ...] = SPLIT_NAMES + ("overall",)

# Categorical colours identifying the three splits in grouped charts. Taken
# unchanged from the first three slots of the reference palette, the subset
# documented as clearing the colourblind-separation floors on every pair.
SPLIT_COLORS: Dict[str, str] = {
    "train": "#2a78d6",
    "val": "#eb6834",
    "test": "#1baf7a",
}

# Resolution of saved figures. High enough to stay sharp when a figure is
# scaled down into the thesis, without producing files of pointless size.
FIGURE_DPI: int = 150


def _finalise_figure(figure: Figure, save_path: Optional[Path]) -> None:
    """Save a figure if a destination was given, then display it.

    Shared by every plotting function, so saving behaves identically wherever
    it is offered. The white face colour is set explicitly: figures inherit a
    transparent background otherwise, which turns into black when the file is
    viewed or embedded in a dark context.

    Args:
        figure: Figure to write and show.
        save_path: Destination file, created together with its parent folder.
            None displays the figure without writing anything.
    """
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
        print(f"[OK] Figure saved to: {save_path}")

    plt.show()

# =====================================================
# 1 - DATASET INVENTORY
# =====================================================


def _collect_stems_per_source(source_dirs: Dict[str, Path]) -> Dict[str, Set[str]]:
    """Collect the file stems present in each source folder.

    Args:
        source_dirs: Mapping from source name to the folder holding its images.

    Returns:
        Mapping from source name to the set of stems found in it.
    """
    return {
        source: {path.stem for path in directory.glob("*.png")}
        for source, directory in source_dirs.items()
    }


def check_dataset_consistency(verbose: bool = True) -> Dict[str, List[str]]:
    """Verify that every sample is complete across all sources.

    A sample is complete when the same file stem exists in every folder the
    pipeline feeds on: the three raw modalities that make up the input
    (intensity, depth, confidence) and the three maps derived from the
    annotations (multiclass labels, binary cordon mask, skeleton). Only the
    multiclass map is the training target — the binary mask serves the cordon
    experiment and the skeleton the clDice metric — but a sample is unusable
    unless all six files are there.

    A stem missing from one of them means either that notebook 01 did not
    finish, or that a raw file was added afterwards without regenerating the
    derived maps. Both would surface much later, as a training run that
    silently skips samples.

    The list of sources comes from data_preparation.get_source_dirs(), so the
    definition of a complete sample cannot drift apart from the one used to
    build the augmented dataset.

    Args:
        verbose: If True, print a per-source table.

    Returns:
        Mapping from source name to the sorted stems missing from it. A source
        with nothing missing maps to an empty list.

    Raises:
        FileNotFoundError: If a source folder does not exist, or no image is
            found in any of them.
    """
    source_dirs = get_source_dirs()

    absent_dirs = [str(directory) for directory in source_dirs.values() if not directory.exists()]
    if absent_dirs:
        raise FileNotFoundError(
            f"{len(absent_dirs)} source folder(s) do not exist, e.g. {absent_dirs[:3]}. "
            "Run notebook 01 first."
        )

    stems_per_source = _collect_stems_per_source(source_dirs)

    # The union, rather than one chosen reference folder, so a stem present
    # only in the raw modalities is reported as a missing label and not
    # ignored altogether.
    all_stems: Set[str] = set().union(*stems_per_source.values())

    if not all_stems:
        raise FileNotFoundError(
            f"No image found in any source folder under {config.DATA_DIR}. "
            "Run notebook 01 first."
        )

    missing_per_source = {
        source: sorted(all_stems - stems) for source, stems in stems_per_source.items()
    }

    if verbose:
        _print_consistency_report(stems_per_source, missing_per_source, len(all_stems))

    return missing_per_source


def _print_consistency_report(
    stems_per_source: Dict[str, Set[str]],
    missing_per_source: Dict[str, List[str]],
    total_stems: int,
) -> None:
    """Print the outcome of the consistency check.

    Args:
        stems_per_source: Stems found in each source folder.
        missing_per_source: Stems missing from each source folder.
        total_stems: Number of distinct stems across all sources.
    """
    separator = "=" * 70

    print(separator)
    print("DATASET CONSISTENCY")
    print(separator)
    print(f"Distinct samples: {total_stems}")
    print()
    print(f"  {'source':<15} {'files':>7} {'missing':>9}")

    for source, stems in stems_per_source.items():
        print(f"  {source:<15} {len(stems):>7} {len(missing_per_source[source]):>9}")

    incomplete = {source: stems for source, stems in missing_per_source.items() if stems}

    print()
    if not incomplete:
        print(f"[OK] Every sample is complete across all {len(stems_per_source)} sources.")
    else:
        print(f"[WARNING] {len(incomplete)} source(s) are missing files:")
        for source, stems in incomplete.items():
            print(f"  {source:<15} {len(stems):>4} missing, e.g. {stems[:3]}")

    print(separator)


def _count_image_channels(image: np.ndarray) -> int:
    """Return the number of channels of a decoded image.

    Args:
        image: Image as returned by cv2.imread().

    Returns:
        1 for a single-channel image, otherwise the size of the last axis.
    """
    return 1 if image.ndim == 2 else image.shape[2]


def describe_image_formats(verbose: bool = True) -> Dict[str, Dict[str, Any]]:
    """Report the format of every source, as the pipeline loads it.

    Files are read through data_preparation.read_modality_image(), so the
    report describes the arrays the Dataset will actually receive rather than
    the bytes on disk. The two differ for the acquisition batch that stored
    single-channel modalities as several identical planes: that difference is
    reconciled at read time, and a file that cannot be reconciled raises there
    instead of being reported here.

    Every file is inspected rather than a random sample: on this dataset the
    full pass takes a few seconds, and a single odd file is precisely what the
    check exists to find. Reporting one arbitrary format and warning about the
    rest — as the original notebook did — is how the mixed channel counts went
    unnoticed in the first place.

    The observed value range doubles as a validity check on the label maps.
    They hold class indices, so the multiclass maps must stay within
    0..NUM_CLASSES-1, the binary masks within 0..1, and the skeleton maps
    within the class indices being skeletonised.

    Args:
        verbose: If True, print one row per source.

    Returns:
        Mapping from source name to a dictionary with keys "file_count",
        "shapes", "dtypes", "channels", "min_value" and "max_value".

    Raises:
        FileNotFoundError: If a source folder holds no image.
        ValueError: Propagated from read_modality_image() when a file cannot be
            decoded or its channel count contradicts config.MODALITY_CHANNELS.
    """
    source_dirs = get_source_dirs()

    files_per_source: Dict[str, List[Path]] = {
        source: sorted(directory.glob("*.png")) for source, directory in source_dirs.items()
    }

    empty_sources = [source for source, files in files_per_source.items() if not files]
    if empty_sources:
        raise FileNotFoundError(
            f"No image found for source(s) {empty_sources}. Run notebook 01 first."
        )

    inspection: Dict[str, Dict[str, Any]] = {}
    total_files = sum(len(files) for files in files_per_source.values())

    with tqdm(total=total_files, desc="Inspecting formats") as progress:
        for source, files in files_per_source.items():
            shapes: Set[Tuple[int, int]] = set()
            dtypes: Set[str] = set()
            channel_counts: Set[int] = set()
            observed_minimums: List[int] = []
            observed_maximums: List[int] = []

            for path in files:
                image = read_modality_image(path, source)

                shapes.add(image.shape[:2])
                dtypes.add(str(image.dtype))
                channel_counts.add(_count_image_channels(image))
                observed_minimums.append(int(image.min()))
                observed_maximums.append(int(image.max()))

                progress.update(1)

            inspection[source] = {
                "file_count": len(files),
                "shapes": sorted(shapes),
                "dtypes": sorted(dtypes),
                "channels": sorted(channel_counts),
                "min_value": min(observed_minimums),
                "max_value": max(observed_maximums),
            }

    if verbose:
        _print_format_report(inspection)

    return inspection


def _format_uniform_value(values: List[Any]) -> str:
    """Render a set of observed values as a single cell of the report.

    Args:
        values: Distinct values observed across the files of one source.

    Returns:
        The value itself when the source is uniform, otherwise a marker
        stating how many different values were found.
    """
    if len(values) == 1:
        return str(values[0])

    return f"MIXED ({len(values)})"


def _print_format_report(inspection: Dict[str, Dict[str, Any]]) -> None:
    """Print the format of each source, one row per source.

    Args:
        inspection: Mapping returned by describe_image_formats().
    """
    separator = "=" * 70

    print(separator)
    print("IMAGE FORMATS")
    print(separator)
    print(
        f"  {'source':<15} {'files':>5}  {'shape':<12} {'dtype':<8} "
        f"{'chans':>9}  {'observed range':<15}"
    )

    for source, description in inspection.items():
        shape = _format_uniform_value(description["shapes"])
        dtype = _format_uniform_value(description["dtypes"])
        channels = _format_uniform_value(description["channels"])
        value_range = f"{description['min_value']} - {description['max_value']}"

        print(
            f"  {source:<15} {description['file_count']:>5}  {shape:<12} {dtype:<8} "
            f"{channels:>9}  {value_range:<15}"
        )

    # A source mixing shapes, dtypes or channel counts breaks the assumption
    # that one loading path fits every file of that modality.
    mixed_sources = [
        source
        for source, description in inspection.items()
        if len(description["shapes"]) > 1
        or len(description["dtypes"]) > 1
        or len(description["channels"]) > 1
    ]

    print()
    if mixed_sources:
        print(f"[WARNING] Non-uniform format in: {mixed_sources}")
        print("          The Dataset assumes one loading path per modality.")
    else:
        print("[OK] Every source is uniform in shape, dtype and channel count.")

    print(separator)


# =====================================================
# 2 - CLASS DISTRIBUTION
# =====================================================


def _count_class_pixels(label_path: Path) -> np.ndarray:
    """Count the pixels of each class in one multiclass label map.

    np.bincount walks the array once, where testing the map against each class
    index in turn would walk it once per class.

    Args:
        label_path: Path to the multiclass label map.

    Returns:
        Array of shape (NUM_CLASSES,) holding the pixel count of each class.

    Raises:
        ValueError: If the map holds a value outside the declared classes.
    """
    label_map = read_modality_image(label_path, "labels_multi")
    counts = np.bincount(label_map.ravel(), minlength=config.NUM_CLASSES)

    if counts.size > config.NUM_CLASSES:
        illegal = [index for index in range(config.NUM_CLASSES, counts.size) if counts[index]]
        raise ValueError(
            f"{label_path.name} holds class index/indices {illegal}, outside the "
            f"{config.NUM_CLASSES} classes declared in config.CLASS_NAMES"
        )

    return counts


def _summarise_counts(
    pixel_counts: np.ndarray,
    image_presence: np.ndarray,
    image_count: int,
) -> Dict[str, Any]:
    """Turn raw counts into the summary reported for one group.

    Args:
        pixel_counts: Pixels of each class, shape (NUM_CLASSES,).
        image_presence: Number of images each class appears in.
        image_count: Number of images the group is made of.

    Returns:
        Dictionary with "pixel_counts", "pixel_frequencies", "image_presence"
        and "image_count".
    """
    total_pixels = int(pixel_counts.sum())

    return {
        "pixel_counts": pixel_counts,
        "pixel_frequencies": pixel_counts / total_pixels if total_pixels else pixel_counts * 0.0,
        "image_presence": image_presence,
        "image_count": image_count,
    }


def compute_class_distribution(
    splits: Optional[Dict[str, List[str]]] = None,
    verbose: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Measure how much of each class the dataset holds, split by split.

    Two quantities are reported per group. The pixel share drives the class
    weights and shows how imbalanced the problem is. The image presence — how
    many images a class appears in at all — answers a question the share
    cannot: a class missing entirely from a partition leaves its IoU undefined
    on that partition, which silently makes the mean IoU incomparable between
    runs. With fourteen images in validation and test, that is a real risk.

    Label maps are read from data/processed, filtered by the split, rather
    than from the augmented folders. A flip permutes pixels without changing
    their values, so the augmented copies carry exactly the same counts as the
    originals four times over: reading them would cost four times as long and
    return the same frequencies.

    Args:
        splits: Mapping from split name to file stems. Defaults to the split
            saved by notebook 01.
        verbose: If True, print the distribution tables.

    Returns:
        Mapping from group name — the three splits plus "overall" — to the
        summary returned by _summarise_counts().

    Raises:
        FileNotFoundError: If a label map named by the split is missing.
    """
    if splits is None:
        splits = load_splits()

    labels_dir = config.PATHS["processed"]["labels_multi"]

    missing = [
        f"{stem}.png"
        for stems in splits.values()
        for stem in stems
        if not (labels_dir / f"{stem}.png").exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} label map(s) named by the split are missing from "
            f"{labels_dir}, e.g. {missing[:3]}. Rerun notebook 01."
        )

    distribution: Dict[str, Dict[str, Any]] = {}

    for split_name in SPLIT_NAMES:
        stems = splits[split_name]
        pixel_counts = np.zeros(config.NUM_CLASSES, dtype=np.int64)
        image_presence = np.zeros(config.NUM_CLASSES, dtype=np.int64)

        for stem in tqdm(stems, desc=f"Counting {split_name}"):
            counts = _count_class_pixels(labels_dir / f"{stem}.png")
            pixel_counts += counts
            image_presence += counts > 0

        distribution[split_name] = _summarise_counts(pixel_counts, image_presence, len(stems))

    # Summed from the partitions rather than recomputed, so no file is read
    # twice and the parts are guaranteed to add up to the whole.
    distribution["overall"] = _summarise_counts(
        pixel_counts=sum(distribution[name]["pixel_counts"] for name in SPLIT_NAMES),
        image_presence=sum(distribution[name]["image_presence"] for name in SPLIT_NAMES),
        image_count=sum(distribution[name]["image_count"] for name in SPLIT_NAMES),
    )

    if verbose:
        _print_distribution_report(distribution)

    return distribution


def _print_distribution_report(distribution: Dict[str, Dict[str, Any]]) -> None:
    """Print the pixel share and the image presence of every class.

    Args:
        distribution: Mapping returned by compute_class_distribution().
    """
    separator = "=" * 70
    header = "".join(f"{group:>11}" for group in DISTRIBUTION_GROUPS)

    print(separator)
    print("CLASS DISTRIBUTION")
    print(separator)
    print("Measured on the original images of each split, before augmentation.")
    print()

    print(f"  Pixel share (%)\n  {'class':<12}{header}")
    for class_index, class_name in enumerate(config.CLASS_NAMES):
        shares = "".join(
            f"{100 * distribution[group]['pixel_frequencies'][class_index]:>11.2f}"
            for group in DISTRIBUTION_GROUPS
        )
        print(f"  {class_name:<12}{shares}")

    print()
    print(f"  Images containing the class\n  {'class':<12}{header}")
    for class_index, class_name in enumerate(config.CLASS_NAMES):
        cells = []
        for group in DISTRIBUTION_GROUPS:
            present = distribution[group]["image_presence"][class_index]
            total = distribution[group]["image_count"]
            cells.append(f"{present}/{total}".rjust(11))
        print(f"  {class_name:<12}{''.join(cells)}")

    absent = [
        (group, config.CLASS_NAMES[class_index])
        for group in SPLIT_NAMES
        for class_index in range(config.NUM_CLASSES)
        if distribution[group]["image_presence"][class_index] == 0
    ]

    print()
    if absent:
        print("[WARNING] Class(es) absent from a partition:")
        for group, class_name in absent:
            print(f"  '{class_name}' never appears in '{group}'")
        print("          Their IoU is undefined there, which makes the mean IoU")
        print("          incomparable with runs where the class is present.")
    else:
        print("[OK] Every class appears in every partition.")

    print(separator)


def plot_class_distribution(
    distribution: Dict[str, Dict[str, Any]],
    save_path: Optional[Path] = None,
    figure_size: Tuple[float, float] = (13.0, 4.5),
) -> None:
    """Draw the class distribution as two panels.

    The imbalance spans two orders of magnitude, so a single linear chart would
    render the rare classes as invisible slivers next to Background. The two
    panels answer two different questions instead: the left one shows how much
    Background dominates, which is what motivates weighting the loss at all;
    the right one drops it and compares the foreground classes across splits on
    a scale where the differences are legible.

    Bars in the left panel carry the colours the masks are painted with, so a
    bar can be matched to the structure it counts. Identity never rests on
    colour alone: the class names label the axis in both panels.

    Args:
        distribution: Mapping returned by compute_class_distribution().
        save_path: Destination file for the figure, or None to only display it.
        figure_size: Size of the whole figure in inches.
    """
    class_names = config.CLASS_NAMES
    foreground_indices = list(range(1, config.NUM_CLASSES))

    figure, (axis_overall, axis_by_split) = plt.subplots(1, 2, figsize=figure_size)

    # --- Left: the whole dataset, every class ---
    overall_shares = 100 * distribution["overall"]["pixel_frequencies"]
    bar_colors = [tuple(channel / 255 for channel in color) for color in config.CLASS_COLORS]

    bars = axis_overall.bar(class_names, overall_shares, color=bar_colors, width=0.7)

    for bar, share in zip(bars, overall_shares):
        axis_overall.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(overall_shares) * 0.02,
            f"{share:.2f}%",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#52514e",
        )

    axis_overall.set_title("Whole dataset", fontsize=10)
    axis_overall.set_ylabel("Pixel share (%)")
    axis_overall.set_ylim(0, max(overall_shares) * 1.18)

    # --- Right: foreground classes only, one bar per split ---
    bar_width = 0.26
    positions = np.arange(len(foreground_indices))

    for offset, split_name in enumerate(SPLIT_NAMES):
        shares = [
            100 * distribution[split_name]["pixel_frequencies"][index]
            for index in foreground_indices
        ]
        axis_by_split.bar(
            positions + (offset - 1) * bar_width,
            shares,
            width=bar_width * 0.92,  # the gap keeps adjacent bars from touching
            label=split_name,
            color=SPLIT_COLORS[split_name],
        )

    axis_by_split.set_xticks(positions)
    axis_by_split.set_xticklabels([class_names[index] for index in foreground_indices])
    axis_by_split.set_title("Foreground classes, by split", fontsize=10)
    axis_by_split.set_ylabel("Pixel share (%)")
    axis_by_split.legend(frameon=False, fontsize=9)

    for axis in (axis_overall, axis_by_split):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", color="#d8d7d2", linewidth=0.6)
        axis.set_axisbelow(True)

    figure.tight_layout()
    _finalise_figure(figure, save_path)
