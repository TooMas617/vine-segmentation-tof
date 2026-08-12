"""Generation of segmentation label maps from LabelMe annotations.

The module turns the JSON files produced by LabelMe into the label maps used
for training:

    * multiclass label maps  -> one integer per pixel, see config.CLASS_NAMES
    * binary label maps      -> cordon (1) against everything else (0)
    * skeleton maps          -> dilated centrelines of the elongated classes
    * colored masks          -> RGB versions of the above, for visual checks

All the settings describing how annotations map onto classes (aliases,
ignored labels, drawing priority, linestrip handling) live in config.py.
This module only implements the logic.

Typical usage from notebook 01:

    from data_preparation import generate_label_maps, generate_skeleton_maps

    generate_label_maps(overwrite=False)
    generate_skeleton_maps(overwrite=False)
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw
from skimage.morphology import skeletonize
from tqdm import tqdm

import config

# LabelMe shape types this module knows how to rasterise.
SUPPORTED_SHAPE_TYPES = ("polygon", "linestrip")


# =====================================================
# 1 - LABEL MAPPING
# =====================================================


def build_label_index_map() -> Dict[str, int]:
    """Build the mapping from annotation label to class index.

    Starts from config.CLASS_NAMES, adds the aliases declared in
    config.LABEL_ALIASES and removes the labels listed in
    config.LABELS_TO_IGNORE. All keys are lowercase.

    Returns:
        Dictionary mapping every accepted label to its class index.

    Raises:
        ValueError: If an alias points to a class that does not exist.
    """
    label_to_index = {name.lower(): index for index, name in enumerate(config.CLASS_NAMES)}

    for alias, target_class in config.LABEL_ALIASES.items():
        target_lower = target_class.lower()

        if target_lower not in label_to_index:
            raise ValueError(
                f"Alias '{alias}' points to unknown class '{target_class}'. "
                f"Available classes: {list(label_to_index)}"
            )

        label_to_index[alias.lower()] = label_to_index[target_lower]

    for label in config.LABELS_TO_IGNORE:
        label_to_index.pop(label.lower(), None)

    return label_to_index


def _validate_priority(priority_order: Sequence[str], label_to_index: Dict[str, int]) -> None:
    """Check that every label in a priority list is a known label.

    Args:
        priority_order: Labels ordered from lowest to highest priority.
        label_to_index: Mapping returned by build_label_index_map().

    Raises:
        ValueError: If the list contains labels that cannot be drawn.
    """
    unknown = {label.lower() for label in priority_order} - set(label_to_index)

    if unknown:
        raise ValueError(f"Priority list contains unknown labels: {sorted(unknown)}")


def _sort_shapes_by_priority(
    shapes: List[Dict],
    label_to_index: Dict[str, int],
    priority_order: Optional[Sequence[str]],
) -> List[Dict]:
    """Order shapes so that higher-priority classes are drawn last.

    Shapes drawn later overwrite earlier ones, so the class appearing last in
    priority_order wins on overlapping pixels. Shapes whose class is not listed
    are drawn first.

    Args:
        shapes: Shape dictionaries from a LabelMe JSON file.
        label_to_index: Mapping returned by build_label_index_map().
        priority_order: Labels from lowest to highest priority, or None to keep
            the original file order.

    Returns:
        The shapes, reordered.
    """
    if priority_order is None:
        return list(shapes)

    # Priority is resolved at class-index level, so aliases inherit the
    # priority of the class they point to.
    index_to_priority: Dict[int, int] = {}
    for position, label in enumerate(priority_order):
        label_lower = label.lower()
        if label_lower in label_to_index:
            index_to_priority[label_to_index[label_lower]] = position

    lowest_priority = len(priority_order)

    def priority_of(shape: Dict) -> int:
        class_index = label_to_index.get(shape.get("label", "").lower(), -1)
        return index_to_priority.get(class_index, lowest_priority)

    return sorted(shapes, key=priority_of)


# =====================================================
# 2 - SHAPE RASTERISATION
# =====================================================


def parse_labelme_json(json_path: Path) -> Dict:
    """Read a LabelMe JSON annotation file.

    Args:
        json_path: Path to the JSON file.

    Returns:
        Dictionary with keys "image_path", "image_height", "image_width"
        and "shapes".
    """
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return {
        "image_path": data.get("imagePath", ""),
        "image_height": data.get("imageHeight", 0),
        "image_width": data.get("imageWidth", 0),
        "shapes": data.get("shapes", []),
    }


def _shape_points(shape: Dict) -> List[Tuple[int, int]]:
    """Extract the vertices of a shape as integer pixel coordinates.

    Args:
        shape: Shape dictionary from a LabelMe JSON file.

    Returns:
        List of (x, y) integer tuples.
    """
    return [tuple(int(round(value)) for value in point) for point in shape.get("points", [])]


def _draw_shape(
    draw: ImageDraw.ImageDraw,
    shape: Dict,
    class_index: int,
    as_thin_line: bool,
    source_file: Optional[str] = None,
    report_warnings: bool = True,
) -> None:
    """Rasterise a single shape onto a drawing surface.

    Polygons are filled. Linestrips are either drawn as a one-pixel line
    (when as_thin_line is True, for open curves such as canes) or closed into
    a filled polygon.

    Args:
        draw: PIL drawing surface bound to the target mask.
        shape: Shape dictionary from a LabelMe JSON file.
        class_index: Value written into the mask for this shape.
        as_thin_line: Whether a linestrip stays a thin line instead of being
            closed into a polygon.
        source_file: File name used in warning messages.
        report_warnings: Whether malformed shapes are reported. Disable it when
            the same shapes have already been reported by a previous pass.
    """
    label = shape.get("label", "")
    shape_type = shape.get("shape_type", "polygon")
    points = _shape_points(shape)
    location = f" in {source_file}" if source_file else ""

    if shape_type not in SUPPORTED_SHAPE_TYPES:
        if report_warnings:
            tqdm.write(f"Warning: unsupported shape type '{shape_type}' for '{label}'{location}")
        return

    if shape_type == "linestrip" and as_thin_line:
        if len(points) < 2:
            if report_warnings:
                tqdm.write(
                    f"Warning: linestrip '{label}' has {len(points)} point(s), "
                    f"at least 2 required — skipped{location}"
                )
            return
        draw.line(points, fill=class_index, width=1)
        return

    if len(points) < 3:
        if report_warnings:
            tqdm.write(
                f"Warning: {shape_type} '{label}' has {len(points)} point(s), "
                f"at least 3 required to fill an area — skipped{location}"
            )
        return

    draw.polygon(points, fill=class_index, outline=class_index)


def _is_thin_line_shape(shape: Dict, thin_line_indices: Set[int], class_index: int) -> bool:
    """Tell whether a shape must be rasterised as a thin, dilatable line.

    Args:
        shape: Shape dictionary from a LabelMe JSON file.
        thin_line_indices: Class indices configured as thin-line classes.
        class_index: Class index of the shape.

    Returns:
        True if the shape is a linestrip of a thin-line class.
    """
    return shape.get("shape_type", "") == "linestrip" and class_index in thin_line_indices


def _dilate_thin_lines(
    mask: np.ndarray,
    shapes: List[Dict],
    label_to_index: Dict[str, int],
    thin_line_indices: Set[int],
    kernel_size: int,
) -> np.ndarray:
    """Give thin lines some area by dilating them into the background.

    Each thin-line class is dilated separately and written only where the mask
    is still background, so dilation never overwrites another class.

    Args:
        mask: Label map produced by the main drawing pass.
        shapes: Shapes of the annotation file, already ordered.
        label_to_index: Mapping returned by build_label_index_map().
        thin_line_indices: Class indices configured as thin-line classes.
        kernel_size: Size of the square dilation kernel.

    Returns:
        The label map with dilated lines merged in.
    """
    height, width = mask.shape

    # Redraw the thin lines in isolation, so dilation cannot pick up pixels
    # belonging to filled polygons of the same class.
    lines_image = Image.fromarray(np.zeros((height, width), dtype=np.uint8))
    lines_draw = ImageDraw.Draw(lines_image)
    drawn_indices: Set[int] = set()

    for shape in shapes:
        class_index = label_to_index.get(shape.get("label", "").lower(), -1)

        if not _is_thin_line_shape(shape, thin_line_indices, class_index):
            continue

        _draw_shape(lines_draw, shape, class_index, as_thin_line=True, report_warnings=False)
        drawn_indices.add(class_index)

    if not drawn_indices:
        return mask

    lines_mask = np.array(lines_image)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    for class_index in sorted(drawn_indices):
        class_lines = (lines_mask == class_index).astype(np.uint8)

        if class_lines.sum() == 0:
            continue

        dilated = cv2.dilate(class_lines, kernel, iterations=1)
        mask = np.where((dilated > 0) & (mask == 0), class_index, mask)

    return mask


def labelme_shapes_to_mask(
    shapes: List[Dict],
    image_height: int,
    image_width: int,
    label_to_index: Dict[str, int],
    priority_order: Optional[Sequence[str]] = None,
    thin_line_indices: Optional[Set[int]] = None,
    dilation_kernel_size: int = config.LINESTRIP_DILATION_KERNEL,
    source_file: Optional[str] = None,
    report_warnings: bool = True,
) -> np.ndarray:
    """Convert LabelMe shapes into an integer label map.

    Args:
        shapes: Shape dictionaries from a LabelMe JSON file.
        image_height: Height of the annotated image, in pixels.
        image_width: Width of the annotated image, in pixels.
        label_to_index: Mapping returned by build_label_index_map().
        priority_order: Labels from lowest to highest drawing priority.
        thin_line_indices: Class indices whose linestrips stay thin lines.
            Defaults to the classes listed in config.LINESTRIP_AS_LINE_CLASSES.
        dilation_kernel_size: Square kernel size used to dilate thin lines.
        source_file: File name used in warning messages.
        report_warnings: Whether to report unknown labels and malformed shapes.
            Disable it on the second pass over the same file, so each problem
            is reported only once.

    Returns:
        Label map of shape (image_height, image_width) with class indices,
        background being 0.
    """
    if thin_line_indices is None:
        thin_line_indices = {
            label_to_index[name.lower()]
            for name in config.LINESTRIP_AS_LINE_CLASSES
            if name.lower() in label_to_index
        }

    ordered_shapes = _sort_shapes_by_priority(shapes, label_to_index, priority_order)

    mask_image = Image.fromarray(np.zeros((image_height, image_width), dtype=np.uint8))
    draw = ImageDraw.Draw(mask_image)
    unknown_labels: Set[str] = set()

    for shape in ordered_shapes:
        label = shape.get("label", "").lower()

        if label not in label_to_index:
            unknown_labels.add(label)
            continue

        class_index = label_to_index[label]
        as_thin_line = _is_thin_line_shape(shape, thin_line_indices, class_index)
        _draw_shape(draw, shape, class_index, as_thin_line, source_file, report_warnings)

    mask = np.array(mask_image)

    mask = _dilate_thin_lines(
        mask=mask,
        shapes=ordered_shapes,
        label_to_index=label_to_index,
        thin_line_indices=thin_line_indices,
        kernel_size=dilation_kernel_size,
    )

    if report_warnings:
        # Labels dropped on purpose must not be reported as annotation errors.
        ignored = {label.lower() for label in config.LABELS_TO_IGNORE}
        unexpected = unknown_labels - ignored

        if unexpected:
            location = f" in {source_file}" if source_file else ""
            tqdm.write(f"Warning: unknown labels {sorted(unexpected)}{location}")

    return mask


def label_map_to_colored_mask(
    label_map: np.ndarray,
    class_colors: Sequence[Tuple[int, int, int]],
) -> np.ndarray:
    """Turn a label map into an RGB image for visual inspection.

    Args:
        label_map: Array of shape (H, W) containing class indices.
        class_colors: One RGB triplet per class index.

    Returns:
        RGB array of shape (H, W, 3).
    """
    # A lookup table maps every class index to its color in a single pass.
    lookup_table = np.zeros((256, 3), dtype=np.uint8)

    for class_index, color in enumerate(class_colors):
        lookup_table[class_index] = color

    return lookup_table[label_map]


# =====================================================
# 3 - PIPELINE
# =====================================================


def _prepare_output_dir(path: Path, overwrite: bool) -> None:
    """Make sure an output directory exists and is in the requested state.

    Args:
        path: Directory to prepare.
        overwrite: If True, existing content is deleted. If False, an existing
            non-empty directory is left untouched.
    """
    if path.exists() and any(path.iterdir()):
        if overwrite:
            shutil.rmtree(path)
            path.mkdir(parents=True, exist_ok=True)
        else:
            print(f"  Directory '{path.name}' is not empty — keeping existing files.")
    else:
        path.mkdir(parents=True, exist_ok=True)


def _save_mask(mask: np.ndarray, directory: Path, filename: str) -> None:
    """Write a single-channel label map to disk as PNG.

    Args:
        mask: Array of shape (H, W) with class indices.
        directory: Destination folder.
        filename: File name including the extension.
    """
    cv2.imwrite(str(directory / filename), mask)


def _save_colored_mask(
    label_map: np.ndarray,
    class_colors: Sequence[Tuple[int, int, int]],
    directory: Path,
    filename: str,
) -> None:
    """Write the RGB version of a label map to disk.

    Args:
        label_map: Array of shape (H, W) with class indices.
        class_colors: One RGB triplet per class index.
        directory: Destination folder.
        filename: File name including the extension.
    """
    colored_rgb = label_map_to_colored_mask(label_map, class_colors)
    # OpenCV writes images in BGR order.
    cv2.imwrite(str(directory / filename), cv2.cvtColor(colored_rgb, cv2.COLOR_RGB2BGR))


def generate_label_maps(
    overwrite: bool = False,
    save_colored: bool = True,
    annotations_dir: Optional[Path] = None,
) -> None:
    """Convert every LabelMe annotation into multiclass and binary label maps.

    Reads the JSON files in the annotations folder and writes, for each of
    them, a multiclass label map and a binary cordon mask (plus their colored
    versions). Classes, aliases and drawing priorities come from config.py.

    Args:
        overwrite: If True, existing outputs are deleted and regenerated.
            If False and all output folders already contain files, the whole
            step is skipped.
        save_colored: Whether to also write the RGB masks used for visual
            inspection.
        annotations_dir: Folder containing the JSON files. Defaults to the
            raw annotations folder declared in config.PATHS.

    Raises:
        ValueError: If the configuration is inconsistent (mismatched colors,
            missing cordon class, invalid priority lists).
    """
    if annotations_dir is None:
        annotations_dir = config.PATHS["raw"]["annotations"]

    processed = config.PATHS["processed"]
    multiclass_dir = processed["labels_multi"]
    binary_dir = processed["labels_binary"]
    colored_multi_dir = processed["colored_multi"]
    colored_binary_dir = processed["colored_binary"]

    # --- Validate the configuration before touching the filesystem ---
    if len(config.CLASS_COLORS) != len(config.CLASS_NAMES):
        raise ValueError(
            f"CLASS_COLORS has {len(config.CLASS_COLORS)} entries but "
            f"CLASS_NAMES has {len(config.CLASS_NAMES)}"
        )

    if len(config.BINARY_COLORS) != 2:
        raise ValueError(
            f"BINARY_COLORS must contain exactly 2 colors, got {len(config.BINARY_COLORS)}"
        )

    label_to_index = build_label_index_map()
    _validate_priority(config.PRIORITY_MULTICLASS, label_to_index)
    _validate_priority(config.PRIORITY_BINARY, label_to_index)

    if "cordon" not in label_to_index:
        raise ValueError("Class 'cordon' is required to build the binary masks")

    cordon_index = label_to_index["cordon"]

    # --- Skip the whole step if everything is already generated ---
    output_dirs = [multiclass_dir, binary_dir]
    if save_colored:
        output_dirs += [colored_multi_dir, colored_binary_dir]

    already_generated = all(
        directory.exists() and any(directory.iterdir()) for directory in output_dirs
    )

    if already_generated and not overwrite:
        print("[OK] Label maps already exist — skipping generation.")
        print("     Call generate_label_maps(overwrite=True) to rebuild them.")
        return

    for directory in output_dirs:
        _prepare_output_dir(directory, overwrite)

    # Sorted for reproducibility.
    json_files = sorted(annotations_dir.glob("*.json"))

    if not json_files:
        raise FileNotFoundError(f"No JSON annotation found in: {annotations_dir}")

    separator = "=" * 70
    print(separator)
    print("LABEL MAPS GENERATION")
    print(separator)
    print(f"Annotations:     {annotations_dir}")
    print(f"Files to process:{len(json_files):>4}")
    print(f"Classes:         {config.CLASS_NAMES}")
    print(f"Aliases:         {config.LABEL_ALIASES}")
    print(f"Ignored labels:  {config.LABELS_TO_IGNORE}")
    print(f"Priority multi:  {config.PRIORITY_MULTICLASS}")
    print(f"Priority binary: {config.PRIORITY_BINARY}")
    print(f"Overwrite:       {overwrite}")
    print(separator)

    for json_path in tqdm(json_files, desc="Generating label maps"):
        annotation = parse_labelme_json(json_path)
        height = annotation["image_height"]
        width = annotation["image_width"]

        if height <= 0 or width <= 0:
            tqdm.write(f"Warning: invalid image size in {json_path.name} — skipped")
            continue

        multiclass_map = labelme_shapes_to_mask(
            shapes=annotation["shapes"],
            image_height=height,
            image_width=width,
            label_to_index=label_to_index,
            priority_order=config.PRIORITY_MULTICLASS,
            source_file=json_path.name,
            report_warnings=True,
        )

        # The binary mask is rasterised again with its own priority order, so
        # that cordon pixels are never hidden by overlapping classes.
        binary_source_map = labelme_shapes_to_mask(
            shapes=annotation["shapes"],
            image_height=height,
            image_width=width,
            label_to_index=label_to_index,
            priority_order=config.PRIORITY_BINARY,
            source_file=json_path.name,
            report_warnings=False,
        )
        binary_map = (binary_source_map == cordon_index).astype(np.uint8)

        filename = f"{json_path.stem}.png"
        _save_mask(multiclass_map, multiclass_dir, filename)
        _save_mask(binary_map, binary_dir, filename)

        if save_colored:
            _save_colored_mask(multiclass_map, config.CLASS_COLORS, colored_multi_dir, filename)
            _save_colored_mask(binary_map, config.BINARY_COLORS, colored_binary_dir, filename)

    print(separator)
    print("LABEL MAPS GENERATION COMPLETED")
    print(separator)
    print(f"  Multiclass label maps: {multiclass_dir}")
    print(f"  Binary label maps:     {binary_dir}")
    if save_colored:
        print(f"  Colored multiclass:    {colored_multi_dir}")
        print(f"  Colored binary:        {colored_binary_dir}")
    print(separator)


# =====================================================
# 4 - SKELETON MAPS
# =====================================================

# Color used to paint skeleton pixels in the overlay, and the darker outline
# drawn around them so they stay visible on light classes.
SKELETON_OVERLAY_COLOR = (255, 255, 255)
SKELETON_OUTLINE_COLOR = (128, 128, 128)


def _class_binary_mask(
    label_multi: np.ndarray,
    label_binary: np.ndarray,
    class_index: int,
    cordon_index: int,
) -> np.ndarray:
    """Build the binary mask a skeleton is extracted from.

    The cordon is annotated as a single continuous structure but is partially
    hidden by canes crossing in front of it. Skeletonising the visible pixels
    alone would produce a broken centreline, so the unified binary mask is used
    instead and the occluded pixels are removed afterwards.

    Args:
        label_multi: Multiclass label map of shape (H, W).
        label_binary: Unified cordon mask of shape (H, W), values 0 or 1.
        class_index: Class the mask is built for.
        cordon_index: Class index of the cordon.

    Returns:
        Binary mask of shape (H, W), values 0 or 1.
    """
    if class_index == cordon_index:
        return (label_binary > 0).astype(np.uint8)

    return (label_multi == class_index).astype(np.uint8)


def compute_tubed_skeleton(
    binary_mask: np.ndarray,
    dilation_kernel: np.ndarray,
) -> np.ndarray:
    """Extract the centreline of a binary mask and give it a small thickness.

    The Zhang-Suen algorithm thins the mask down to a one-pixel-wide centreline,
    which is then dilated into a thin tube. Dilation is clipped to the input
    mask, so the tube never grows past the boundary of the annotated structure.

    Args:
        binary_mask: Binary mask of shape (H, W), values 0 or 1.
        dilation_kernel: Square kernel used to thicken the centreline.

    Returns:
        Binary tubed skeleton of shape (H, W), values 0 or 1.
    """
    skeleton = skeletonize(binary_mask.astype(bool)).astype(np.uint8)
    tubed = cv2.dilate(skeleton, dilation_kernel, iterations=1)

    # Clipping matters because dilation is isotropic: without it the tube would
    # bleed into neighbouring classes wherever the structure is thinner than
    # the kernel.
    return tubed & binary_mask


def build_skeleton_map(
    label_multi: np.ndarray,
    label_binary: np.ndarray,
    skeleton_class_indices: Sequence[int],
    cordon_index: int,
    dilation_kernel_size: int,
) -> np.ndarray:
    """Build the skeleton map of a single annotated image.

    Each class is skeletonised independently and the results are merged. The
    merge cannot lose pixels: after occlusion masking every skeleton lives only
    on pixels where label_multi holds its own class, so the per-class skeletons
    are disjoint and the priority already resolved in the label map is
    inherited here.

    Args:
        label_multi: Multiclass label map of shape (H, W).
        label_binary: Unified cordon mask of shape (H, W), values 0 or 1.
        skeleton_class_indices: Class indices to skeletonise.
        cordon_index: Class index of the cordon.
        dilation_kernel_size: Square kernel size used to thicken centrelines.

    Returns:
        Skeleton map of shape (H, W) holding class indices, background 0.
    """
    skeleton_map = np.zeros_like(label_multi, dtype=np.uint8)
    kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)

    # Pixels where the cordon is annotated but hidden behind another class.
    # The model cannot be expected to predict cordon there, so they are dropped
    # from the skeleton rather than counted as missed centreline.
    occlusion_mask = (label_binary > 0) & (label_multi != cordon_index)

    for class_index in skeleton_class_indices:
        binary_mask = _class_binary_mask(
            label_multi=label_multi,
            label_binary=label_binary,
            class_index=class_index,
            cordon_index=cordon_index,
        )

        if binary_mask.sum() == 0:
            continue

        tubed_skeleton = compute_tubed_skeleton(binary_mask, kernel)
        class_skeleton = tubed_skeleton * class_index

        if class_index == cordon_index:
            class_skeleton[occlusion_mask] = 0

        skeleton_map = np.where(class_skeleton > 0, class_skeleton, skeleton_map)

    return skeleton_map


def build_skeleton_overlay(
    label_multi: np.ndarray,
    skeleton_map: np.ndarray,
) -> np.ndarray:
    """Draw the skeleton on top of the colored multiclass mask.

    Meant for visual inspection only: skeleton pixels are painted white and
    outlined in grey so they remain readable over any class color.

    Args:
        label_multi: Multiclass label map of shape (H, W).
        skeleton_map: Skeleton map of shape (H, W) holding class indices.

    Returns:
        RGB overlay of shape (H, W, 3).
    """
    overlay = label_map_to_colored_mask(label_multi, config.CLASS_COLORS)
    skeleton_pixels = skeleton_map > 0

    # The outline is the ring of pixels just outside the skeleton, obtained by
    # dilating it and subtracting the skeleton itself.
    dilated = cv2.dilate(skeleton_pixels.astype(np.uint8), np.ones((3, 3), np.uint8))
    outline_pixels = (dilated > 0) & ~skeleton_pixels

    overlay[outline_pixels] = SKELETON_OUTLINE_COLOR
    overlay[skeleton_pixels] = SKELETON_OVERLAY_COLOR

    return overlay


def generate_skeleton_maps(
    overwrite: bool = False,
    save_colored: bool = True,
) -> None:
    """Precompute the skeleton map of every processed label map.

    Skeletons are needed by the clDice metric and by the skeleton recall loss.
    Computing them once offline keeps them out of the training loop, where
    Zhang-Suen thinning would dominate the data loading time.

    Which classes are skeletonised and how much the centrelines are thickened
    is set in config.SKELETON_LOSS_CONFIG.

    Args:
        overwrite: If True, existing skeletons are deleted and recomputed.
            If False and the output folders already contain files, the step is
            skipped.
        save_colored: Whether to also write the overlay images used for visual
            inspection.

    Raises:
        FileNotFoundError: If the label maps have not been generated yet, or if
            a multiclass label map has no matching binary mask.
    """
    processed = config.PATHS["processed"]
    multiclass_dir = processed["labels_multi"]
    binary_dir = processed["labels_binary"]
    skeleton_dir = processed["skeleton"]
    colored_skeleton_dir = processed["colored_skeleton"]

    for directory in (multiclass_dir, binary_dir):
        if not directory.exists() or not any(directory.iterdir()):
            raise FileNotFoundError(
                f"Label maps not found in {directory}. "
                "Run generate_label_maps() first."
            )

    skeleton_config = config.SKELETON_LOSS_CONFIG
    skeleton_class_indices = skeleton_config["skeleton_classes"]
    dilation_kernel_size = skeleton_config["dilation_kernel_size"]
    cordon_index = config.CLASS_NAMES.index("Cordon")

    output_dirs = [skeleton_dir]
    if save_colored:
        output_dirs.append(colored_skeleton_dir)

    already_generated = all(
        directory.exists() and any(directory.iterdir()) for directory in output_dirs
    )

    if already_generated and not overwrite:
        print("[OK] Skeleton maps already exist — skipping generation.")
        print("     Call generate_skeleton_maps(overwrite=True) to rebuild them.")
        return

    label_files = sorted(multiclass_dir.glob("*.png"))

    # Fail before writing anything, rather than halfway through the loop.
    missing_binary = [path.name for path in label_files if not (binary_dir / path.name).exists()]
    if missing_binary:
        raise FileNotFoundError(
            f"{len(missing_binary)} multiclass label map(s) have no binary "
            f"counterpart in {binary_dir}, e.g. {missing_binary[:3]}"
        )

    for directory in output_dirs:
        _prepare_output_dir(directory, overwrite)

    skeleton_class_names = [config.CLASS_NAMES[index] for index in skeleton_class_indices]

    separator = "=" * 70
    print(separator)
    print("SKELETON MAPS GENERATION")
    print(separator)
    print(f"Input labels:     {multiclass_dir}")
    print(f"Files to process: {len(label_files):>4}")
    print(f"Skeleton classes: {skeleton_class_names}")
    print(f"Dilation kernel:  {dilation_kernel_size}x{dilation_kernel_size}")
    print(f"Overwrite:        {overwrite}")
    print(separator)

    for label_path in tqdm(label_files, desc="Generating skeleton maps"):
        filename = label_path.name

        label_multi = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
        label_binary = cv2.imread(str(binary_dir / filename), cv2.IMREAD_GRAYSCALE)

        if label_multi is None or label_binary is None:
            tqdm.write(f"Warning: could not read label pair for {filename} — skipped")
            continue

        if label_multi.shape != label_binary.shape:
            tqdm.write(
                f"Warning: shape mismatch for {filename} "
                f"({label_multi.shape} vs {label_binary.shape}) — skipped"
            )
            continue

        skeleton_map = build_skeleton_map(
            label_multi=label_multi,
            label_binary=label_binary,
            skeleton_class_indices=skeleton_class_indices,
            cordon_index=cordon_index,
            dilation_kernel_size=dilation_kernel_size,
        )

        cv2.imwrite(str(skeleton_dir / filename), skeleton_map)

        if save_colored:
            overlay = build_skeleton_overlay(label_multi, skeleton_map)
            cv2.imwrite(
                str(colored_skeleton_dir / filename),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
            )

    print(separator)
    print("SKELETON MAPS GENERATION COMPLETED")
    print(separator)
    print(f"  Skeleton maps:    {skeleton_dir}")
    if save_colored:
        print(f"  Colored overlays: {colored_skeleton_dir}")
    print(separator)
