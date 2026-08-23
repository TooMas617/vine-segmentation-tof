"""Loss functions for multiclass segmentation of thin structures.

    * DiceLoss           -> overlap, insensitive to how large a class is
    * CombinedLoss       -> cross-entropy plus Dice, the default
    * FocalLoss          -> cross-entropy focused on the hard pixels
    * SkeletonRecallLoss -> penalises centrelines the prediction misses

Cross-entropy scores every pixel independently, so a class covering 0.3 % of
the image barely registers; Dice scores the overlap of a class as a whole, so a
small class weighs as much as a large one. Combining them keeps the rare
classes visible to the optimiser while the per-pixel term drives the bulk of
the learning.

Class weights are measured once during EDA and reloaded through
config.load_class_weights(), so the loss never measures the data it is applied
to.

Typical usage from notebook 03:

    from losses import create_loss_function, create_skeleton_loss

    criterion = create_loss_function(class_weights=config.load_class_weights())
    skeleton_criterion = create_skeleton_loss()
"""

from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

import config

# Guards every division by an accumulated count, which is zero whenever a class
# is absent from the batch.
DEFAULT_EPSILON: float = 1e-6


# =====================================================
# 1 - OVERLAP LOSSES
# =====================================================


class DiceLoss(nn.Module):
    """Soft Dice loss, computed per class over the whole batch.

    Overlap is accumulated across the batch rather than averaged per image, so
    an image holding fifty pixels of a class does not weigh as much as one
    holding five thousand. For a class as rare as Node this is the difference
    between a usable signal and noise.

    A class absent from both prediction and target scores a perfect 1 and so
    contributes nothing, which is intended: a model cannot be asked to segment
    what is not there. Predicting it where it is absent still costs, since the
    denominator grows while the intersection stays empty.
    """

    def __init__(
        self,
        num_classes: Optional[int] = None,
        class_weights: Optional[torch.Tensor] = None,
        epsilon: float = DEFAULT_EPSILON,
    ) -> None:
        """Prepare the loss.

        Args:
            num_classes: Number of segmentation classes.
            class_weights: Weight of each class, shape (num_classes,), or None
                for an unweighted mean.
            epsilon: Constant guarding the division.
        """
        if num_classes is None:
            num_classes = config.NUM_CLASSES

        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        # A buffer follows the module to the GPU with .to(device), instead of
        # being copied across on every forward pass.
        self.register_buffer(
            "class_weights",
            None if class_weights is None else class_weights.detach().clone(),
        )

    def forward(self, pred_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the loss.

        Args:
            pred_logits: Raw model output, shape (B, C, H, W).
            targets: Ground truth class indices, shape (B, H, W).

        Returns:
            Scalar loss.
        """
        pred_probs = F.softmax(pred_logits, dim=1)
        targets_one_hot = F.one_hot(targets, self.num_classes).permute(0, 3, 1, 2).float()

        # Summed over batch, height and width: one Dice score per class.
        dims = (0, 2, 3)
        intersection = (pred_probs * targets_one_hot).sum(dims)
        cardinality = (pred_probs + targets_one_hot).sum(dims)

        per_class_loss = 1.0 - (2.0 * intersection + self.epsilon) / (cardinality + self.epsilon)

        if self.class_weights is None:
            return per_class_loss.mean()

        weights = self.class_weights.to(per_class_loss.device)

        return (per_class_loss * weights).sum() / weights.sum()


class CombinedLoss(nn.Module):
    """Cross-entropy and Dice, mixed in fixed proportion.

    The two shares are required to sum to one, so the scale of the total stays
    comparable when the mix changes and two runs differing only in the
    proportion remain readable side by side.

    Note:
        This differs from the reference implementations of the connectivity
        losses, which sum Dice and cross-entropy with coefficient 1 each and so
        work on a scale roughly twice this one. A weight for the skeleton term
        taken from that literature therefore has to be halved here — see
        config.SKELETON_LOSS_CONFIG.
    """

    def __init__(
        self,
        ce_weight: float,
        dice_weight: float,
        num_classes: Optional[int] = None,
        class_weights: Optional[torch.Tensor] = None,
        use_weighted_dice: bool = True,
    ) -> None:
        """Prepare the loss.

        Args:
            ce_weight: Share of the cross-entropy term.
            dice_weight: Share of the Dice term.
            num_classes: Number of segmentation classes.
            class_weights: Weight of each class, or None.
            use_weighted_dice: Whether the Dice term is weighted too. Leaving
                it off keeps Dice as an unweighted counterpart to a weighted
                cross-entropy.

        Raises:
            ValueError: If the two shares do not sum to one.
        """
        if num_classes is None:
            num_classes = config.NUM_CLASSES

        super().__init__()

        if not 0.99 <= ce_weight + dice_weight <= 1.01:
            raise ValueError(
                f"ce_weight and dice_weight should sum to 1.0, got "
                f"{ce_weight} + {dice_weight} = {ce_weight + dice_weight:.4f}"
            )

        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.register_buffer(
            "class_weights",
            None if class_weights is None else class_weights.detach().clone(),
        )
        self.dice_loss = DiceLoss(
            num_classes=num_classes,
            class_weights=class_weights if use_weighted_dice else None,
        )

    def forward(self, pred_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the loss.

        Args:
            pred_logits: Raw model output, shape (B, C, H, W).
            targets: Ground truth class indices, shape (B, H, W).

        Returns:
            Scalar loss.
        """
        weights = (
            None if self.class_weights is None else self.class_weights.to(pred_logits.device)
        )
        cross_entropy = F.cross_entropy(pred_logits, targets, weight=weights)

        return self.ce_weight * cross_entropy + self.dice_weight * self.dice_loss(
            pred_logits, targets
        )


class FocalLoss(nn.Module):
    """Cross-entropy reweighted towards the pixels the model gets wrong.

    Each pixel is scaled by (1 - p)^gamma, where p is the probability given to
    its true class, so a pixel already predicted confidently contributes almost
    nothing and the gradient concentrates on the boundaries of thin structures.
    gamma = 0 recovers plain cross-entropy.
    """

    def __init__(self, gamma: float, class_weights: Optional[torch.Tensor] = None) -> None:
        """Prepare the loss.

        Args:
            gamma: Focusing exponent. Higher values concentrate harder.
            class_weights: Weight of each class, or None.

        Raises:
            ValueError: If gamma is negative.
        """
        super().__init__()

        if gamma < 0:
            raise ValueError(f"gamma must be non-negative, got {gamma}")

        self.gamma = gamma
        self.register_buffer(
            "class_weights",
            None if class_weights is None else class_weights.detach().clone(),
        )

    def forward(self, pred_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the loss.

        Args:
            pred_logits: Raw model output, shape (B, C, H, W).
            targets: Ground truth class indices, shape (B, H, W).

        Returns:
            Scalar loss.
        """
        weights = (
            None if self.class_weights is None else self.class_weights.to(pred_logits.device)
        )

        # Per pixel, so the focusing factor applies before the reduction.
        cross_entropy = F.cross_entropy(pred_logits, targets, weight=weights, reduction="none")

        # The probability of the true class, read straight from the softmax
        # rather than recovered as exp(-CE): with class weights the loss is
        # scaled by the weight, so exp(-CE) would not be a probability and
        # gamma would act differently from one class to the next.
        true_class_probability = F.softmax(pred_logits, dim=1).gather(
            1, targets.unsqueeze(1)
        ).squeeze(1)

        return (((1.0 - true_class_probability) ** self.gamma) * cross_entropy).mean()


# =====================================================
# 2 - SKELETON RECALL LOSS
# =====================================================


class SkeletonRecallLoss(nn.Module):
    """Reward the prediction for covering the centreline of each class.

    Overlap losses treat every pixel of a structure alike, so a prediction that
    breaks a cane in two loses little: the missing pixels are few. Connectivity
    is what makes the segmentation usable, so scoring recall on the precomputed
    centreline puts weight back on the pixels that hold the structure together.

    Per class, over the tubed skeleton of the ground truth:

        recall = sum(p_class * skeleton_class) / sum(skeleton_class)
        loss   = 1 - mean(recall over the classes present)

    where p_class is the *probability* the model assigns to the class.

    Note:
        The probability is what makes this trainable. Taking the predicted
        label with argmax detaches the term from the computational graph
        entirely — the tensor comes back with no grad_fn — so the loss would
        appear in the reported total, change the number printed, and leave
        every weight untouched.

        The reference formulation states the loss as -mean(recall). This uses
        1 - mean(recall), which differs by an additive constant and therefore
        has exactly the same gradient, while keeping the logged value in [0, 1]
        alongside the other losses.

    Note:
        This term rewards recall and never penalises a false positive: the loss
        is zero wherever the skeleton is, so no gradient flows outside it. On a
        model that already over-segments, expect precision to fall while the
        overlap scores hold. Report precision per class alongside IoU.
    """

    def __init__(
        self,
        skeleton_classes: Optional[Sequence[int]] = None,
        epsilon: float = DEFAULT_EPSILON,
    ) -> None:
        """Prepare the loss.

        Args:
            skeleton_classes: Class indices whose centreline is scored.
                Defaults to config.SKELETON_LOSS_CONFIG.
            epsilon: Constant guarding the division.

        Raises:
            ValueError: If no skeleton class is given.
        """
        super().__init__()

        if skeleton_classes is None:
            skeleton_classes = config.SKELETON_LOSS_CONFIG["skeleton_classes"]

        if not skeleton_classes:
            raise ValueError("At least one skeleton class is required")

        self.skeleton_classes = list(skeleton_classes)
        self.epsilon = epsilon

    def forward(self, pred_logits: torch.Tensor, skeleton_gt: torch.Tensor) -> torch.Tensor:
        """Compute the loss.

        Args:
            pred_logits: Raw model output, shape (B, C, H, W).
            skeleton_gt: Precomputed tubed skeleton, shape (B, H, W), holding
                class indices with 0 for background.

        Returns:
            Scalar loss. Zero when the batch holds no skeleton pixel at all,
            which leaves the total and its gradient untouched.
        """
        pred_probs = F.softmax(pred_logits, dim=1)
        skeleton_gt = skeleton_gt.to(pred_logits.device)

        class_recalls: List[torch.Tensor] = []

        for class_index in self.skeleton_classes:
            skeleton_mask = (skeleton_gt == class_index).float()
            skeleton_pixels = skeleton_mask.sum()

            # A class absent from the batch has no centreline to recover, so it
            # is left out of the mean rather than counted as a total miss. The
            # reference formulation divides by the fixed number of classes,
            # which is undefined in that case.
            if skeleton_pixels == 0:
                continue

            recall = (pred_probs[:, class_index] * skeleton_mask).sum() / (
                skeleton_pixels + self.epsilon
            )
            class_recalls.append(recall)

        if not class_recalls:
            return torch.zeros((), device=pred_logits.device, dtype=pred_logits.dtype)

        return 1.0 - torch.stack(class_recalls).mean()


# =====================================================
# 3 - LOSS FACTORY
# =====================================================


def create_loss_function(
    loss_config: Optional[Dict[str, Any]] = None,
    class_weights: Optional[torch.Tensor] = None,
) -> nn.Module:
    """Build the main loss described by the configuration.

    Args:
        loss_config: Loss settings. Defaults to config.LOSS_CONFIG.
        class_weights: Weight of each class, as returned by
            config.load_class_weights(), or None for an unweighted loss.

    Returns:
        The loss module, ready to be moved to the device.

    Raises:
        ValueError: If the loss type is unknown.
    """
    if loss_config is None:
        loss_config = config.LOSS_CONFIG

    loss_type = loss_config["type"]

    if loss_type == "combined":
        return CombinedLoss(
            ce_weight=loss_config["ce_weight"],
            dice_weight=loss_config["dice_weight"],
            class_weights=class_weights,
            use_weighted_dice=loss_config["use_weighted_dice"],
        )

    if loss_type == "ce":
        return nn.CrossEntropyLoss(weight=class_weights)

    if loss_type == "dice":
        return DiceLoss(class_weights=class_weights)

    if loss_type == "focal":
        return FocalLoss(gamma=loss_config["focal_gamma"], class_weights=class_weights)

    raise ValueError(
        f"Unknown loss type '{loss_type}'. Available: 'combined', 'ce', 'dice', 'focal'"
    )


def create_skeleton_loss(
    skeleton_config: Optional[Dict[str, Any]] = None,
) -> Optional[SkeletonRecallLoss]:
    """Build the skeleton recall loss, if the configuration enables it.

    Args:
        skeleton_config: Skeleton settings. Defaults to
            config.SKELETON_LOSS_CONFIG.

    Returns:
        The loss module, or None when the term is disabled. Running once with
        it and once without is the ablation that settles whether it helps on
        this dataset.
    """
    if skeleton_config is None:
        skeleton_config = config.SKELETON_LOSS_CONFIG

    if not skeleton_config["enabled"]:
        return None

    return SkeletonRecallLoss(skeleton_classes=skeleton_config["skeleton_classes"])
