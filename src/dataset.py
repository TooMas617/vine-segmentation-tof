"""PyTorch dataset, augmentation pipeline and dataloaders.

The module turns the folders materialised by notebook 01 into batches:

    * MultiChannelNormalise   -> per-channel standardisation of the input
    * MultiModalTransform     -> geometry applied to image and masks alike
    * VineSegmentationDataset -> one sample as (image, label, skeleton)
    * create_dataloaders      -> the three loaders, seeded and reproducible

Images are read through data_preparation.read_modality_image(), so the channel
count of every modality matches config.MODALITY_CHANNELS whatever the source
files look like, and modalities are stacked in config.MODALITY_ORDER — the same
order the exploratory analysis measured the normalisation statistics in.

Augmentation is purely geometric, by decision: photometric transforms would
change what a depth value and a confidence value mean, and applying them to
intensity alone would break the physical agreement between channels describing
the same pixel. Appearance variation comes from the offline flips instead.

albumentations is imported lazily. It cannot be installed on Windows without a
C++ compiler, and online augmentation only runs during training on Colab, so
everything except build_transforms() works locally.

Typical usage from notebook 03:

    from dataset import build_transforms, create_dataloaders

    transforms = build_transforms()
    dataloaders = create_dataloaders(transforms)
"""

import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import config
from data_preparation import SPLIT_NAMES, read_modality_image

# albumentations needs a C++ compiler on Windows and is only used for the
# online augmentation, which runs on Colab. Importing it lazily keeps the
# Dataset and the dataloaders usable on the local machine.
try:  # pragma: no cover - depends on the environment, not on the logic
    import albumentations as albumentations_module
except ImportError:  # pragma: no cover
    albumentations_module = None

# Statistics of the ImageNet corpus, used only when a pre-trained encoder
# expects inputs standardised the way it was trained.
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

# Value an 8-bit channel is divided by before ImageNet statistics apply, which
# are defined on the unit interval.
_UINT8_MAX: float = 255.0

# The training image is enlarged before the random geometry, so the centre crop
# that follows always lands on real pixels: rotation and distortion push
# content in from outside the final window instead of reflecting a border into
# it. A rotation of 15 degrees on a square of side L leaves an inscribed
# axis-aligned square of L / (cos15 + sin15) = L / 1.225, so 1.5 leaves ample
# margin for the 512-pixel crop.
GEOMETRIC_MARGIN_FACTOR: float = 1.5

# Random scale applied on top of that margin, as albumentations expects it:
# the sampled factor is 1 + limit, so this range means s in [0.82, 1.30].
#
# The lower bound is what makes the reflected border unreachable instead of
# merely rare, and it is a geometric condition rather than a preference.
# Rotating a square of side L by ROTATION_LIMIT degrees leaves an inscribed
# axis-aligned square of L / (cos15 + sin15) made of real pixels only. The
# crop stays inside it while target_size * f * s >= target_size * 1.2247,
# that is f * s >= 1.2247, which at f = 1.5 means s >= 0.8165.
#
# That threshold is measured, not just derived: at s = 0.8165 a sentinel run
# through cv2.warpAffine still shows 0.002 per cent of the crop touched by the
# border, because bilinear interpolation samples a 2x2 neighbourhood and bleeds
# one pixel past the analytic bound. s = 0.82 gives 0.000 per cent at every
# angle in the range. Below it the guarantee degrades fast: the previous bound
# of 0.68 put 8.6 per cent reflected pixels inside the crop.
#
# A scale change is also the only variation this camera can genuinely produce:
# it never frames somewhere without vines, it frames the same scene from
# slightly nearer or further, which is a field of view and not an arbitrary
# crop. That is why the geometry zooms rather than translating.
RANDOM_SCALE_LIMIT: Tuple[float, float] = (-0.18, 0.30)

# Rotation range in degrees, and the distortion strength applied after it.
ROTATION_LIMIT: int = 15
DISTORTION_LIMIT: Tuple[float, float] = (-0.15, 0.15)

# Probability of each random geometric transform.
GEOMETRIC_PROBABILITY: float = 0.5

# Probability that a training sample skips the geometry entirely and is only
# resized, which is exactly what validation and test receive.
#
# Enlarging and cropping keeps every pixel real, but it cannot reach the
# evaluation scale: a sample rotated by up to 15 degrees and cropped without
# fabricated pixels is magnified by at least cos15 + sin15 = 1.2247, whatever
# the margin factor. The floor comes from the rotation, so no choice of
# GEOMETRIC_MARGIN_FACTOR removes it. Without this branch the model would
# never once see the framing it is evaluated on.
#
# The two branches are exclusive: a sample is either the evaluation view or a
# geometrically perturbed one, never a compromise between them.
IDENTITY_PROBABILITY: float = 0.5


# =====================================================
# 1 - NORMALISATION
# =====================================================


def enabled_modality_names() -> List[str]:
    """Return the enabled modalities, in the order they are stacked.

    Returns:
        Modality names taken from config.MODALITY_ORDER, keeping only those
        switched on in config.ENABLED_MODALITIES.
    """
    return [
        modality
        for modality in config.MODALITY_ORDER
        if config.ENABLED_MODALITIES.get(modality, False)
    ]


def select_enabled_channel_stats(
    means: Sequence[float],
    stds: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Keep the statistics of the channels the model actually receives.

    The EDA measures every modality, enabled or not, so that switching one back
    on never finds a file without values for it. The Dataset stacks only the
    enabled ones, so the saved vectors have to be cut down to the same subset —
    by name rather than by position, since a disabled modality in the middle
    would otherwise shift every later channel onto the wrong statistics.

    Args:
        means: Per-channel means for every declared channel.
        stds: Per-channel standard deviations, same order.

    Returns:
        Tuple of (means, stds) restricted to the enabled channels.

    Raises:
        ValueError: If the input does not cover every declared channel.
    """
    all_channels = config.build_channel_names()

    if len(means) != len(all_channels) or len(stds) != len(all_channels):
        raise ValueError(
            f"Expected statistics for {len(all_channels)} channels "
            f"{all_channels}, got {len(means)} mean(s) and {len(stds)} std(s)"
        )

    enabled_channels = config.build_channel_names(enabled_modality_names())
    keep = [all_channels.index(name) for name in enabled_channels]

    return np.asarray(means, dtype=np.float32)[keep], np.asarray(stds, dtype=np.float32)[keep]


class MultiChannelNormalise:
    """Standardise each input channel with the statistics measured in EDA.

    Two strategies, chosen by config.USE_IMAGENET_NORM_FOR_RGB:

        False  every channel uses dataset statistics. The right choice for
               architectures trained from scratch.
        True   the three depth channels use ImageNet statistics, intensity and
               confidence keep dataset ones. Only worth it with a pre-trained
               encoder, which expects inputs distributed as its training set.
    """

    def __init__(
        self,
        channel_means: Sequence[float],
        channel_stds: Sequence[float],
        use_imagenet_for_depth: Optional[bool] = None,
    ) -> None:
        """Prepare the normaliser for the currently enabled modalities.

        Args:
            channel_means: Per-channel means of the enabled channels.
            channel_stds: Per-channel standard deviations, same order.
            use_imagenet_for_depth: Override for
                config.USE_IMAGENET_NORM_FOR_RGB.

        Raises:
            ValueError: If the statistics do not match the enabled channels, or
                a standard deviation is not strictly positive.
        """
        self.channel_names = config.build_channel_names(enabled_modality_names())
        self.channel_count = len(self.channel_names)

        if len(channel_means) != self.channel_count or len(channel_stds) != self.channel_count:
            raise ValueError(
                f"Expected {self.channel_count} values per statistic, one per enabled "
                f"channel {self.channel_names}, got {len(channel_means)} and {len(channel_stds)}"
            )

        self.channel_means = np.asarray(channel_means, dtype=np.float32)
        self.channel_stds = np.asarray(channel_stds, dtype=np.float32)

        if (self.channel_stds <= 0).any():
            degenerate = [
                name for name, std in zip(self.channel_names, self.channel_stds) if std <= 0
            ]
            raise ValueError(f"Non-positive standard deviation for channel(s) {degenerate}")

        self.use_imagenet_for_depth = (
            config.USE_IMAGENET_NORM_FOR_RGB
            if use_imagenet_for_depth is None
            else use_imagenet_for_depth
        )

        # Position of the depth block inside the stacked tensor, needed only by
        # the ImageNet branch. None when depth is disabled.
        self._depth_slice: Optional[slice] = None
        if config.ENABLED_MODALITIES.get("depth", False):
            depth_channels = config.build_channel_names(["depth"])
            start = self.channel_names.index(depth_channels[0])
            self._depth_slice = slice(start, start + len(depth_channels))

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Standardise one stacked image.

        Args:
            image: Array of shape (H, W, C) holding the stacked modalities.

        Returns:
            Float32 array of the same shape, standardised channel by channel.

        Raises:
            ValueError: If the image does not hold the expected channel count.
        """
        if image.ndim != 3 or image.shape[2] != self.channel_count:
            raise ValueError(
                f"Expected an image with {self.channel_count} channel(s) "
                f"{self.channel_names}, got shape {image.shape}"
            )

        normalised = image.astype(np.float32)

        if self.use_imagenet_for_depth and self._depth_slice is not None:
            for index in range(self.channel_count):
                if self._depth_slice.start <= index < self._depth_slice.stop:
                    continue
                normalised[:, :, index] = (
                    normalised[:, :, index] - self.channel_means[index]
                ) / self.channel_stds[index]

            # ImageNet statistics are defined on [0, 1], so the block is scaled
            # before they are applied.
            depth_block = normalised[:, :, self._depth_slice] / _UINT8_MAX
            for offset, (mean, std) in enumerate(zip(IMAGENET_MEAN, IMAGENET_STD)):
                depth_block[:, :, offset] = (depth_block[:, :, offset] - mean) / std
            normalised[:, :, self._depth_slice] = depth_block
        else:
            normalised = (normalised - self.channel_means) / self.channel_stds

        return normalised


def build_normaliser(
    splits_hash: Optional[str] = None,
    channel_means: Optional[Sequence[float]] = None,
    channel_stds: Optional[Sequence[float]] = None,
) -> MultiChannelNormalise:
    """Build the normaliser from the statistics notebook 02 saved.

    Args:
        splits_hash: Fingerprint of the split in use. When given, a mismatch
            with the one recorded in the statistics file is reported.
        channel_means: Statistics for every declared channel, overriding the
            saved file. Mostly useful in tests.
        channel_stds: Standard deviations matching channel_means.

    Returns:
        A normaliser restricted to the enabled channels.

    Raises:
        FileNotFoundError: If the statistics have not been computed yet and no
            explicit values were passed.
    """
    if channel_means is None or channel_stds is None:
        loaded = config.load_channel_stats(expected_splits_hash=splits_hash)

        if loaded is None:
            raise FileNotFoundError(
                f"Channel statistics not found at {config.CHANNEL_STATS_JSON}. "
                "Run notebook 02 (EDA) before training."
            )

        channel_means, channel_stds = loaded

    enabled_means, enabled_stds = select_enabled_channel_stats(channel_means, channel_stds)

    return MultiChannelNormalise(enabled_means, enabled_stds)


# =====================================================
# 2 - AUGMENTATION PIPELINE
# =====================================================


def _require_albumentations() -> Any:
    """Return the albumentations module, or explain why it is missing.

    Returns:
        The imported albumentations module.

    Raises:
        ImportError: If albumentations is not installed.
    """
    if albumentations_module is None:
        raise ImportError(
            "albumentations is not installed. On Windows its stringzilla "
            "dependency needs a C++ compiler; online augmentation is meant to "
            "run on Colab. The Dataset works without it: pass transform=None."
        )

    return albumentations_module


class MultiModalTransform:
    """Apply one geometric transformation to the image and every mask.

    Geometry must reach the image *and* all its masks, or the labels stop
    describing the pixels they sit on. Masks are handled as a list so the label
    map and the skeleton map receive exactly the same transformation.

    There is deliberately no photometric stage. Brightness, contrast or noise
    transforms would change what a depth value and a confidence value mean, and
    applying them to intensity alone would break the physical agreement between
    channels that describe the same pixel.

    A second pipeline may be supplied as `identity`. When it is, each sample
    takes one branch or the other, never both: the geometry enlarges and crops,
    so it cannot produce the plain resize that validation and test receive, and
    a model that never sees that framing is evaluated on one it was not trained
    on. Drawing the branch per sample keeps both framings in the training
    distribution while every pixel of both stays real.
    """

    def __init__(
        self,
        geometric: Any,
        normaliser: MultiChannelNormalise,
        identity: Optional[Any] = None,
        identity_probability: float = 0.0,
    ) -> None:
        """Assemble the pipeline.

        Args:
            geometric: albumentations pipeline applied to image and masks.
            normaliser: Per-channel standardisation applied after the geometry.
            identity: Pipeline used instead of `geometric` on the sampled
                fraction of calls. Omitted for validation and test, which must
                stay deterministic.
            identity_probability: Probability of taking that branch. Ignored
                when `identity` is None.
        """
        self.geometric = geometric
        self.normaliser = normaliser
        self.identity = identity
        self.identity_probability = identity_probability if identity is not None else 0.0

    def __call__(
        self,
        image: np.ndarray,
        masks: List[np.ndarray],
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Transform one sample.

        Args:
            image: Stacked image of shape (H, W, C), 8-bit.
            masks: Masks receiving the same geometry as the image.

        Returns:
            Tuple of the image tensor, shaped (C, H, W) and float32, and the
            mask tensors, each shaped (H, W) and int64.
        """
        # random is seeded per worker by config.seed_worker, the same mechanism
        # albumentations relies on for its own probabilities.
        take_identity = (
            self.identity_probability > 0.0
            and random.random() < self.identity_probability
        )
        pipeline = self.identity if take_identity else self.geometric

        transformed = pipeline(image=image, masks=masks)

        return to_tensors(self.normaliser(transformed["image"]), transformed["masks"])


def to_tensors(
    image: np.ndarray,
    masks: Sequence[np.ndarray],
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Convert one transformed sample into the tensors the model consumes.

    Args:
        image: Normalised image of shape (H, W, C).
        masks: Masks of shape (H, W) holding class indices.

    Returns:
        Tuple of the image tensor (C, H, W) as float32 and the mask tensors
        (H, W) as int64, which is what CrossEntropyLoss requires.
    """
    image_tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float()
    mask_tensors = [torch.from_numpy(np.ascontiguousarray(mask)).long() for mask in masks]

    return image_tensor, mask_tensors


def build_transforms(
    img_size: Tuple[int, int] = config.IMG_SIZE,
    normaliser: Optional[MultiChannelNormalise] = None,
    splits_hash: Optional[str] = None,
) -> Dict[str, MultiModalTransform]:
    """Build the training and evaluation pipelines.

    Training takes one of two exclusive branches per sample. The geometric one
    enlarges by GEOMETRIC_MARGIN_FACTOR, perturbs the geometry and crops the
    centre back to the target size, so the crop lands on real pixels instead of
    on a reflected border — at the cost of a magnification of at least 1.2247,
    which the rotation makes unavoidable. The other branch, taken with
    probability IDENTITY_PROBABILITY, is the evaluation pipeline itself, so the
    framing the model is scored on is also one it has trained on.

    Validation and test only resize: their samples must stay identical across
    epochs and across runs, or the numbers they produce cannot be compared.

    Masks are resampled with nearest-neighbour interpolation, which keeps class
    indices intact; the image is resampled bilinearly.

    Args:
        img_size: Target size as (height, width).
        normaliser: Normalisation to apply. Built from the saved statistics
            when omitted.
        splits_hash: Fingerprint of the split, forwarded to the normaliser so a
            statistics file from another partition is reported.

    Returns:
        Mapping with the "train" and "val_test" pipelines.

    Raises:
        ImportError: If albumentations is not installed.
    """
    albumentations = _require_albumentations()

    if normaliser is None:
        normaliser = build_normaliser(splits_hash=splits_hash)

    height, width = img_size
    margin_height = int(height * GEOMETRIC_MARGIN_FACTOR)
    margin_width = int(width * GEOMETRIC_MARGIN_FACTOR)

    geometric_train = albumentations.Compose(
        [
            albumentations.Resize(
                height=margin_height, width=margin_width, interpolation=cv2.INTER_LINEAR
            ),
            albumentations.RandomScale(
                scale_limit=RANDOM_SCALE_LIMIT, p=GEOMETRIC_PROBABILITY
            ),
            albumentations.Rotate(
                limit=ROTATION_LIMIT,
                border_mode=cv2.BORDER_REFLECT,
                p=GEOMETRIC_PROBABILITY,
            ),
            albumentations.OpticalDistortion(
                distort_limit=DISTORTION_LIMIT,
                border_mode=cv2.BORDER_REFLECT,
                p=GEOMETRIC_PROBABILITY,
            ),
            albumentations.CenterCrop(height=height, width=width),
        ]
    )

    geometric_val_test = albumentations.Compose(
        [albumentations.Resize(height=height, width=width, interpolation=cv2.INTER_LINEAR)]
    )

    return {
        "train": MultiModalTransform(
            geometric_train,
            normaliser,
            identity=geometric_val_test,
            identity_probability=IDENTITY_PROBABILITY,
        ),
        "val_test": MultiModalTransform(geometric_val_test, normaliser),
    }


# =====================================================
# 3 - DATASET
# =====================================================


class VineSegmentationDataset(Dataset):
    """One sample of the augmented dataset, as the model expects it.

    Every enabled modality is read and concatenated along the channel axis in
    config.MODALITY_ORDER, giving an image of config.INPUT_CHANNELS channels.
    The label map and the skeleton map come back as separate tensors, both
    carrying class indices.

    Reading goes through data_preparation.read_modality_image(), so a modality
    always has the channel count config.MODALITY_CHANNELS declares, whatever
    the source file holds.
    """

    def __init__(
        self,
        split_name: str,
        transform: Optional[MultiModalTransform] = None,
        normaliser: Optional[MultiChannelNormalise] = None,
        load_skeleton: Optional[bool] = None,
    ) -> None:
        """Open one split of the augmented dataset.

        Args:
            split_name: One of "train", "val" or "test".
            transform: Pipeline applied to every sample. When omitted the
                sample is normalised and converted to tensors without being
                resized, which is what a local check without albumentations
                needs.
            normaliser: Normalisation used when transform is None. Built from
                the saved statistics if both are omitted.
            load_skeleton: Whether to read the skeleton maps. Defaults to
                config.SKELETON_LOSS_CONFIG["load_skeleton"]. When off, the
                skeleton tensor is returned as zeros, so the shape of a batch
                never depends on the configuration.

        Raises:
            ValueError: If the split name is unknown or a modality holds a
                different set of files than the labels.
            FileNotFoundError: If a required folder is missing.
        """
        if split_name not in SPLIT_NAMES:
            raise ValueError(f"Unknown split '{split_name}'. Available: {list(SPLIT_NAMES)}")

        self.split_name = split_name
        self.transform = transform
        self.load_skeleton = (
            config.SKELETON_LOSS_CONFIG["load_skeleton"]
            if load_skeleton is None
            else load_skeleton
        )

        split_dirs = config.PATHS["augmented"][split_name]
        self.modality_dirs: Dict[str, Path] = {
            modality: split_dirs[modality] for modality in enabled_modality_names()
        }
        self.label_dir: Path = split_dirs["labels_multi"]
        self.skeleton_dir: Path = split_dirs["skeleton"]

        required: Dict[str, Path] = dict(self.modality_dirs)
        required["labels_multi"] = self.label_dir
        if self.load_skeleton:
            required["skeleton"] = self.skeleton_dir

        missing_dirs = [f"{name} ({path})" for name, path in required.items() if not path.exists()]
        if missing_dirs:
            raise FileNotFoundError(
                f"Missing folder(s) for split '{split_name}': {missing_dirs}. "
                "Run notebook 01 first."
            )

        self.stems: List[str] = sorted(path.stem for path in self.label_dir.glob("*.png"))

        if not self.stems:
            raise ValueError(f"No label map found in {self.label_dir}")

        # Checked once at construction rather than per sample: a mismatch means
        # images and labels are no longer paired, which would train the model
        # on the wrong targets without raising anything.
        for name, directory in required.items():
            found = sorted(path.stem for path in directory.glob("*.png"))
            if found != self.stems:
                missing = sorted(set(self.stems) - set(found))
                unexpected = sorted(set(found) - set(self.stems))
                raise ValueError(
                    f"'{name}' does not hold the same files as the labels of split "
                    f"'{split_name}': {len(found)} file(s) against {len(self.stems)}. "
                    f"Missing {missing[:3]}, unexpected {unexpected[:3]}. "
                    "Rebuild with build_augmented_dataset(overwrite=True)."
                )

        # Only needed when no transform carries its own normaliser.
        self._normaliser = normaliser
        if self.transform is None and self._normaliser is None:
            self._normaliser = build_normaliser()

    def __len__(self) -> int:
        """Return the number of samples in the split."""
        return len(self.stems)

    def _load_image(self, stem: str) -> np.ndarray:
        """Read and stack every enabled modality of one sample.

        Args:
            stem: File stem of the sample.

        Returns:
            Array of shape (H, W, config.INPUT_CHANNELS).
        """
        planes: List[np.ndarray] = []

        for modality, directory in self.modality_dirs.items():
            image = read_modality_image(directory / f"{stem}.png", modality)
            planes.append(image[:, :, np.newaxis] if image.ndim == 2 else image)

        return np.concatenate(planes, axis=-1)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return one sample.

        Args:
            index: Position of the sample in the split.

        Returns:
            Tuple of the image tensor (C, H, W) as float32, the label tensor
            (H, W) as int64 and the skeleton tensor (H, W) as int64.
        """
        stem = self.stems[index]

        image = self._load_image(stem)
        label = read_modality_image(self.label_dir / f"{stem}.png", "labels_multi")

        if self.load_skeleton:
            skeleton = read_modality_image(self.skeleton_dir / f"{stem}.png", "skeleton")
        else:
            skeleton = np.zeros_like(label)

        if self.transform is not None:
            image_tensor, (label_tensor, skeleton_tensor) = self.transform(
                image=image, masks=[label, skeleton]
            )
        else:
            # No pipeline: normalise and convert, leaving the geometry alone.
            normaliser = self._normaliser
            if normaliser is None:
                raise RuntimeError("No transform and no normaliser available")
            image_tensor, (label_tensor, skeleton_tensor) = to_tensors(
                normaliser(image), [label, skeleton]
            )

        return image_tensor, label_tensor, skeleton_tensor


# =====================================================
# 4 - DATALOADERS
# =====================================================


def create_dataloaders(
    transforms: Optional[Dict[str, MultiModalTransform]] = None,
    batch_size: int = config.BATCH_SIZE,
    num_workers: Optional[int] = None,
    load_skeleton: Optional[bool] = None,
    verbose: bool = True,
) -> Dict[str, DataLoader]:
    """Build the three dataloaders, seeded so a run can be reproduced.

    Only the training loader shuffles: validation and test must present their
    samples in the same order every epoch, or per-batch numbers stop being
    comparable across runs.

    The training loader also drops a trailing batch of a single sample. Batch
    normalisation cannot compute statistics from one value per channel, and
    PSPNet's 1x1 pyramid level reduces the feature map to exactly that, so a
    lone final sample would stop training mid-epoch. Validation and test are
    unaffected: in evaluation mode batch normalisation uses its running
    statistics, so every sample is kept.

    Args:
        transforms: Mapping with the "train" and "val_test" pipelines, as
            returned by build_transforms(). None leaves the samples unresized,
            for a local check without albumentations.
        batch_size: Samples per batch. Constrained by GPU memory, and it
            affects results — runs meant to be compared must share it.
        num_workers: Loader worker processes. Defaults to config.NUM_WORKERS,
            capped at the number of available CPUs.
        load_skeleton: Whether the skeleton maps are read.
        verbose: If True, print the size of each loader.

    Returns:
        Mapping from split name to its DataLoader.
    """
    if num_workers is None:
        num_workers = min(config.NUM_WORKERS, os.cpu_count() or 1)

    generator = torch.Generator()
    generator.manual_seed(config.RANDOM_SEED)

    dataloaders: Dict[str, DataLoader] = {}

    for split_name in SPLIT_NAMES:
        is_training = split_name == "train"
        transform = None
        if transforms is not None:
            transform = transforms["train"] if is_training else transforms["val_test"]

        dataset = VineSegmentationDataset(
            split_name=split_name,
            transform=transform,
            load_skeleton=load_skeleton,
        )

        # Only a trailing batch of exactly one is a problem, so the samples are
        # dropped only when there would be one.
        drop_last = is_training and len(dataset) % batch_size == 1

        dataloaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_training,
            drop_last=drop_last,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            worker_init_fn=config.seed_worker,
            generator=generator,
            pin_memory=torch.cuda.is_available(),
        )

    if verbose:
        _print_dataloader_summary(dataloaders, batch_size, num_workers)

    return dataloaders


def _print_dataloader_summary(
    dataloaders: Dict[str, DataLoader],
    batch_size: int,
    num_workers: int,
) -> None:
    """Print how many samples and batches each loader holds.

    Args:
        dataloaders: Mapping returned by create_dataloaders().
        batch_size: Samples per batch.
        num_workers: Loader worker processes.
    """
    separator = "=" * 70
    channel_names = config.build_channel_names(enabled_modality_names())

    print(separator)
    print("DATALOADERS")
    print(separator)

    for split_name, loader in dataloaders.items():
        sample_count = len(loader.dataset)  # type: ignore[arg-type]
        print(f"  {split_name:<6} {sample_count:>5} samples, {len(loader):>4} batches")

    print()
    print(f"Batch size:     {batch_size}")
    print(f"Workers:        {num_workers}")
    print(f"Input channels: {config.INPUT_CHANNELS} {channel_names}")
    print(f"Device:         {config.DEVICE}")
    print(separator)
