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


# =====================================================
# 3 - CHANNEL STATISTICS
# =====================================================

# Number of levels an 8-bit channel can take. The histograms are accumulated
# exactly, one bin per level, rather than binned into ranges.
_INTENSITY_LEVELS: int = 256


def _accumulate_channel_sums(
    stems: List[str],
    source_dirs: Dict[str, Path],
    description: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Walk a set of images once, accumulating what the statistics need.

    Only running totals are kept, never the pixel values themselves: the
    original implementation collected every pixel into Python lists, which for
    this dataset means well over a hundred million objects and several
    gigabytes. Sums, sums of squares and a 256-bin histogram per channel are a
    few hundred numbers regardless of how many images are read.

    Args:
        stems: File stems to walk.
        source_dirs: Mapping returned by get_source_dirs().
        description: Label shown on the progress bar.

    Returns:
        Tuple of (totals, squared_totals, histograms, pixels_per_channel).
        Histograms hold one row per channel and are all zero when the data is
        not 8-bit, since the accumulator assumes one bin per level.

    Raises:
        ValueError: If the channels read do not add up to the declared ones.
    """
    channel_names = config.build_channel_names()
    channel_count = len(channel_names)

    totals = np.zeros(channel_count, dtype=np.float64)
    squared_totals = np.zeros(channel_count, dtype=np.float64)
    histograms = np.zeros((channel_count, _INTENSITY_LEVELS), dtype=np.int64)
    pixels_per_channel = 0
    histograms_valid = True

    for stem in tqdm(stems, desc=description):
        channel_index = 0

        for modality in config.MODALITY_ORDER:
            # read_modality_image() raises on a missing or unreadable file, so
            # a gap cannot leave channel_index behind and shift every later
            # channel into the wrong accumulator.
            image = read_modality_image(source_dirs[modality] / f"{stem}.png", modality)
            planes = image[:, :, np.newaxis] if image.ndim == 2 else image

            if modality == config.MODALITY_ORDER[0]:
                pixels_per_channel += planes.shape[0] * planes.shape[1]

            for plane_index in range(planes.shape[2]):
                plane = planes[:, :, plane_index]
                as_float = plane.astype(np.float64)

                totals[channel_index] += as_float.sum()
                squared_totals[channel_index] += np.square(as_float).sum()

                if plane.dtype == np.uint8:
                    histograms[channel_index] += np.bincount(
                        plane.ravel(), minlength=_INTENSITY_LEVELS
                    )
                else:
                    histograms_valid = False

                channel_index += 1

        if channel_index != channel_count:
            raise ValueError(
                f"{stem} yielded {channel_index} channel(s) but "
                f"{channel_count} are declared: {channel_names}"
            )

    if not histograms_valid:
        histograms = np.zeros_like(histograms)

    return totals, squared_totals, histograms, pixels_per_channel


def _summarise_channel_sums(
    totals: np.ndarray,
    squared_totals: np.ndarray,
    histograms: np.ndarray,
    pixels_per_channel: int,
) -> Dict[str, Any]:
    """Turn running totals into per-channel mean and standard deviation.

    The variance comes from E[x²] - E[x]², which needs a single pass where the
    textbook formula needs two. The cancellation that makes this form risky
    only bites when the standard deviation is minute next to the mean; with
    8-bit values accumulated in float64 there are ten orders of magnitude to
    spare. The clamp at zero guards the remaining rounding noise.

    Args:
        totals: Sum of the values of each channel.
        squared_totals: Sum of the squared values of each channel.
        histograms: Value histogram of each channel, or zeros if unavailable.
        pixels_per_channel: Pixels each channel contributed.

    Returns:
        Dictionary with "means", "stds", "histograms" and "pixels_per_channel".
    """
    means = totals / pixels_per_channel
    variances = np.maximum(squared_totals / pixels_per_channel - np.square(means), 0.0)

    return {
        "means": means,
        "stds": np.sqrt(variances),
        "histograms": histograms,
        "pixels_per_channel": pixels_per_channel,
    }


def compute_channel_statistics(
    splits: Optional[Dict[str, List[str]]] = None,
    verbose: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Measure the per-channel mean and standard deviation used to normalise.

    Every modality declared in config.MODALITY_CHANNELS is measured, whether or
    not it is currently enabled: the statistics describe the data, not the
    experiment, so turning a modality back on later can never find a file that
    has no values for it. Channels follow config.MODALITY_ORDER, the order the
    Dataset will stack them in.

    Reported for all four groups. Only the training values are meant to be
    saved, through config.save_channel_stats(): normalisation is applied to the
    model's input, so statistics drawn from validation or test would carry
    information about them into training. The other groups are descriptive, and
    a marked difference between them would point at drift between acquisition
    batches rather than at a coding mistake.

    Args:
        splits: Mapping from split name to file stems. Defaults to the split
            saved by notebook 01.
        verbose: If True, print the mean and standard deviation tables.

    Returns:
        Mapping from group name to the summary returned by
        _summarise_channel_sums().

    Raises:
        ValueError: Propagated from read_modality_image() when a file is
            unreadable or contradicts config.MODALITY_CHANNELS.
    """
    if splits is None:
        splits = load_splits()

    source_dirs = get_source_dirs()
    statistics: Dict[str, Dict[str, Any]] = {}

    accumulated: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, int]] = {}

    for split_name in SPLIT_NAMES:
        accumulated[split_name] = _accumulate_channel_sums(
            stems=splits[split_name],
            source_dirs=source_dirs,
            description=f"Measuring {split_name}",
        )
        statistics[split_name] = _summarise_channel_sums(*accumulated[split_name])

    # Summed from the partitions rather than measured again, so the images are
    # read once and the whole is guaranteed consistent with its parts.
    statistics["overall"] = _summarise_channel_sums(
        totals=np.sum([accumulated[name][0] for name in SPLIT_NAMES], axis=0),
        squared_totals=np.sum([accumulated[name][1] for name in SPLIT_NAMES], axis=0),
        histograms=np.sum([accumulated[name][2] for name in SPLIT_NAMES], axis=0),
        pixels_per_channel=sum(accumulated[name][3] for name in SPLIT_NAMES),
    )

    if verbose:
        _print_channel_statistics(statistics)

    return statistics


def _print_channel_statistics(statistics: Dict[str, Dict[str, Any]]) -> None:
    """Print the mean and standard deviation of every channel, group by group.

    Args:
        statistics: Mapping returned by compute_channel_statistics().
    """
    separator = "=" * 70
    channel_names = config.build_channel_names()
    header = "".join(f"{group:>11}" for group in DISTRIBUTION_GROUPS)

    print(separator)
    print("CHANNEL STATISTICS")
    print(separator)
    print("Every modality is measured, enabled or not. Channel order follows")
    print(f"config.MODALITY_ORDER: {list(config.MODALITY_ORDER)}")
    print()

    for statistic, label in (("means", "Mean"), ("stds", "Standard deviation")):
        print(f"  {label}\n  {'channel':<12}{header}")
        for channel_index, channel_name in enumerate(channel_names):
            values = "".join(
                f"{statistics[group][statistic][channel_index]:>11.2f}"
                for group in DISTRIBUTION_GROUPS
            )
            print(f"  {channel_name:<12}{values}")
        print()

    print("Only the 'train' column is meant to be saved: normalisation feeds the")
    print("model, so validation and test must not contribute to it.")
    print(separator)


def plot_channel_histograms(
    statistics: Dict[str, Dict[str, Any]],
    save_path: Optional[Path] = None,
    columns: int = 3,
) -> None:
    """Draw the value distribution of every channel, one panel per channel.

    Each panel overlays the three splits. The counts are turned into densities
    first, since the splits hold very different numbers of images and the raw
    counts would differ by a factor of five for reasons that say nothing about
    the data.

    Mean and standard deviation summarise a distribution only as far as it is
    unimodal; these panels show what the two numbers hide — clipping at the
    ends of the range, or a second mode, both plausible in a confidence channel.

    Args:
        statistics: Mapping returned by compute_channel_statistics().
        save_path: Destination file for the figure, or None to only display it.
        columns: Panels per row.

    Raises:
        ValueError: If the histograms were not accumulated, which happens when
            the data is not 8-bit.
    """
    channel_names = config.build_channel_names()

    if not statistics["overall"]["histograms"].any():
        raise ValueError(
            "No histogram available: the accumulator assumes 8-bit channels. "
            "Mean and standard deviation are unaffected."
        )

    rows = -(-len(channel_names) // columns)  # ceiling division
    figure, axes = plt.subplots(
        rows, columns, figsize=(4.2 * columns, 2.8 * rows), squeeze=False
    )
    flat_axes = axes.flatten()

    levels = np.arange(_INTENSITY_LEVELS)

    for channel_index, channel_name in enumerate(channel_names):
        axis = flat_axes[channel_index]

        for split_name in SPLIT_NAMES:
            counts = statistics[split_name]["histograms"][channel_index]
            density = counts / counts.sum() if counts.sum() else counts
            axis.plot(
                levels,
                density,
                label=split_name,
                color=SPLIT_COLORS[split_name],
                linewidth=1.2,
            )

        mean = statistics["train"]["means"][channel_index]
        axis.axvline(mean, color="#52514e", linewidth=0.9, linestyle="--")

        axis.set_title(channel_name, fontsize=10)
        axis.set_xlim(0, _INTENSITY_LEVELS - 1)
        axis.set_yticks([])
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_visible(False)

    # Panels left over when the channels do not fill the grid.
    for empty_axis in flat_axes[len(channel_names):]:
        empty_axis.set_visible(False)

    flat_axes[0].legend(frameon=False, fontsize=8)
    figure.suptitle(
        "Channel value distribution (dashed line: training mean)", fontsize=11
    )
    figure.tight_layout()
    _finalise_figure(figure, save_path)


# =====================================================
# 4 - CLASS WEIGHTS
# =====================================================

# Methods worth putting side by side. "none" is flat by construction, and
# "median_freq" is excluded because under normalisation it is not a distinct
# method: dividing by the smallest weight cancels the median factor, so it
# returns exactly what "inverse" returns. Plotting both would show two
# identical series and read as a bug.
_COMPARABLE_METHODS: Tuple[str, ...] = ("sqrt_inverse", "inverse")

# Categorical colours for the method comparison, from the same validated
# subset used for the splits.
_METHOD_COLORS: Dict[str, str] = {
    "sqrt_inverse": "#2a78d6",
    "inverse": "#eb6834",
    "median_freq": "#1baf7a",
}


def compute_class_weights(
    class_frequencies: np.ndarray,
    method: Optional[str] = None,
    normalize: Optional[bool] = None,
    verbose: bool = True,
) -> np.ndarray:
    """Turn class frequencies into the weights the loss applies to each class.

    Weighting exists because the classes differ by more than two orders of
    magnitude: without it the loss is dominated by Background, and a model that
    predicts Background everywhere already scores well. A weight raises the cost
    of an error on a rare class in proportion to how rare it is.

    The frequencies must come from the training split alone. They describe the
    data the model is fitted on, and drawing them from the whole dataset would
    carry validation and test information into the loss.

    Available methods, from mildest to harshest:

        sqrt_inverse  1 / sqrt(f)      the default
        inverse       1 / f
        median_freq   median(f) / f
        none          uniform

    Note:
        With normalize=True, "median_freq" and "inverse" return the same
        weights: the median is a constant factor and cancels when every weight
        is divided by the smallest one. The two differ only when normalisation
        is switched off.

    Args:
        class_frequencies: Pixel share of each class, shape (NUM_CLASSES,).
        method: Weighting method. Defaults to config.CLASS_WEIGHT_CONFIG.
        normalize: If True, divide by the smallest weight so the mildest class
            weighs exactly 1 and the others are read as multiples of it.
            Defaults to config.CLASS_WEIGHT_CONFIG.
        verbose: If True, print the frequency and weight of every class.

    Returns:
        Array of shape (NUM_CLASSES,) with one weight per class.

    Raises:
        ValueError: If the method is unknown, the frequencies do not hold one
            value per class, or a class has zero frequency.
    """
    if method is None:
        method = config.CLASS_WEIGHT_CONFIG["method"]
    if normalize is None:
        normalize = config.CLASS_WEIGHT_CONFIG["normalize"]

    if len(class_frequencies) != config.NUM_CLASSES:
        raise ValueError(
            f"Expected {config.NUM_CLASSES} frequencies, one per class, "
            f"got {len(class_frequencies)}"
        )

    # Every method except "none" divides by the frequency, so an absent class
    # would produce an infinite weight and poison the loss on the first batch.
    absent = [
        config.CLASS_NAMES[index]
        for index, frequency in enumerate(class_frequencies)
        if frequency <= 0
    ]
    if absent and method != "none":
        raise ValueError(
            f"Class(es) {absent} have zero frequency in the split the weights "
            f"are computed from, so method '{method}' would divide by zero."
        )

    frequencies = np.asarray(class_frequencies, dtype=np.float64)

    if method == "sqrt_inverse":
        weights = 1.0 / np.sqrt(frequencies)
    elif method == "inverse":
        weights = 1.0 / frequencies
    elif method == "median_freq":
        weights = np.median(frequencies) / frequencies
    elif method == "none":
        weights = np.ones_like(frequencies)
    else:
        raise ValueError(
            f"Unknown class weighting method '{method}'. Available: "
            "'sqrt_inverse', 'inverse', 'median_freq', 'none'"
        )

    if normalize:
        weights = weights / weights.min()

    if verbose:
        _print_class_weights(frequencies, weights, method, bool(normalize))

    return weights


def _print_class_weights(
    class_frequencies: np.ndarray,
    weights: np.ndarray,
    method: str,
    normalize: bool,
) -> None:
    """Print the frequency and the resulting weight of every class.

    Args:
        class_frequencies: Pixel share of each class.
        weights: Weight computed for each class.
        method: Weighting method used.
        normalize: Whether the weights were divided by the smallest one.
    """
    separator = "=" * 70

    print(separator)
    print("CLASS WEIGHTS")
    print(separator)
    print(f"Method:    {method}")
    print(f"Normalise: {normalize}")
    print("Computed on the training split only.")
    print()
    print(f"  {'class':<12}{'pixel share':>14}{'weight':>10}")

    for class_index, class_name in enumerate(config.CLASS_NAMES):
        share = 100 * class_frequencies[class_index]
        print(f"  {class_name:<12}{share:>13.2f}%{weights[class_index]:>10.2f}")

    print()
    print(f"Ratio between the heaviest and the lightest class: {weights.max() / weights.min():.1f}x")
    print(separator)


def plot_class_weight_methods(
    class_frequencies: np.ndarray,
    save_path: Optional[Path] = None,
    figure_size: Tuple[float, float] = (8.0, 4.0),
) -> None:
    """Compare what each weighting method would produce on this dataset.

    The figure exists to justify the choice of method rather than to explore:
    the three curves differ by more than an order of magnitude on the rarest
    class, and the scale of that gap is the whole argument.

    The vertical axis is logarithmic. On a linear axis the `inverse` bars would
    flatten everything else into the baseline, which is exactly the property
    being illustrated and would therefore become invisible.

    Args:
        class_frequencies: Pixel share of each class, shape (NUM_CLASSES,).
        save_path: Destination file for the figure, or None to only display it.
        figure_size: Size of the figure in inches.

    Raises:
        ValueError: Propagated from compute_class_weights().
    """
    figure, axis = plt.subplots(figsize=figure_size)

    bar_width = 0.26
    positions = np.arange(config.NUM_CLASSES)

    for offset, method in enumerate(_COMPARABLE_METHODS):
        weights = compute_class_weights(class_frequencies, method=method, verbose=False)
        axis.bar(
            positions + (offset - 1) * bar_width,
            weights,
            width=bar_width * 0.92,
            label=method,
            color=_METHOD_COLORS[method],
        )

    axis.set_yscale("log")
    axis.set_xticks(positions)
    axis.set_xticklabels(config.CLASS_NAMES)
    axis.set_ylabel("Weight (log scale, lightest class = 1)")
    axis.set_title("Class weight by method", fontsize=10)
    axis.legend(frameon=False, fontsize=9)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color="#d8d7d2", linewidth=0.6)
    axis.set_axisbelow(True)

    figure.tight_layout()
    _finalise_figure(figure, save_path)
