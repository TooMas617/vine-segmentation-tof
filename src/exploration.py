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
from typing import Any, Dict, List, Set, Tuple

import numpy as np
from tqdm import tqdm

import config
from data_preparation import get_source_dirs, read_modality_image

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
