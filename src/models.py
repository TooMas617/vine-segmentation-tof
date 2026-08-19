"""Segmentation architectures, all trained from scratch.

    * CustomUNet    -> encoder-decoder with skip connections
    * SegNet        -> encoder-decoder with pooling indices, no skips
    * CustomPSPNet  -> dilated encoder plus pyramid pooling

No pre-trained backbone is used, so `segmentation_models_pytorch` is not a
dependency: an ImageNet encoder expects three RGB channels while the input here
is five heterogeneous ones — ToF intensity, colourised depth and confidence —
and the transfer would only ever apply to intensity. The modules below are
plain PyTorch and can therefore be checked locally before any GPU time is
spent.

The three architectures answer the same question differently, which is what
makes the comparison worth running:

    U-Net   keeps full resolution through skip connections, so boundaries stay
            sharp. The usual choice for thin structures.
    SegNet  carries only the pooling indices rather than whole feature maps,
            which costs far less memory at some loss of detail. Established in
            this domain: it is the architecture used for apple-tree branch and
            trellis-wire segmentation and in viticulture.
    PSPNet  aggregates context at several scales, which helps when a class is
            defined by its role in the scene rather than by local appearance —
            a cordon differs from a cane mostly by position and continuity.

Typical usage from notebook 03:

    from models import create_model, test_model

    model = create_model().to(config.DEVICE)
    test_model(model)
"""

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import config

# Architectures this module can build. UNetPlusPlus appears in
# config.ARCHITECTURE but is deliberately not implemented yet.
AVAILABLE_ARCHITECTURES: Tuple[str, ...] = ("UNet", "SegNet", "PSPNet")

# Bin sizes of the pyramid pooling module, from the PSPNet paper.
DEFAULT_POOL_SIZES: Tuple[int, ...] = (1, 2, 3, 6)


# =====================================================
# 1 - SHARED BLOCKS
# =====================================================


def conv_block(in_channels: int, out_channels: int, dilation: int = 1) -> nn.Sequential:
    """Build the convolution unit every architecture here is made of.

    Convolution, batch normalisation, activation. The bias is dropped because
    batch normalisation immediately re-centres the output, which makes it a
    parameter with no effect.

    Args:
        in_channels: Channels entering the block.
        out_channels: Channels leaving it.
        dilation: Spacing between kernel taps. Above 1 it widens the receptive
            field without reducing resolution, which is how PSPNet keeps
            spatial detail deep in the encoder.

    Returns:
        The block as a sequential module.
    """
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class DoubleConv(nn.Module):
    """Two convolution blocks in sequence, the unit of U-Net and SegNet."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Build the pair.

        Args:
            in_channels: Channels entering the block.
            out_channels: Channels leaving it.
        """
        super().__init__()
        self.block = nn.Sequential(
            conv_block(in_channels, out_channels),
            conv_block(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the two convolutions.

        Args:
            x: Input of shape (B, in_channels, H, W).

        Returns:
            Output of shape (B, out_channels, H, W).
        """
        return self.block(x)


# =====================================================
# 2 - U-NET
# =====================================================


class Down(nn.Module):
    """Halve the resolution, then double the channels."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Build the step.

        Args:
            in_channels: Channels entering the step.
            out_channels: Channels leaving it.
        """
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Downsample and convolve.

        Args:
            x: Input of shape (B, in_channels, H, W).

        Returns:
            Output of shape (B, out_channels, H/2, W/2).
        """
        return self.block(x)


class Up(nn.Module):
    """Double the resolution and merge the matching encoder features.

    The skip connection is what makes U-Net suited to thin structures: the
    decoder receives the full-resolution features the encoder saw before
    pooling, so a cane a few pixels wide is not reconstructed from a map where
    it has been reduced to a single pixel.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Build the step.

        Args:
            in_channels: Channels entering from the layer below.
            out_channels: Channels leaving the step.
        """
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        # After concatenation the width is in_channels // 2 from the upsampled
        # path plus in_channels // 2 from the skip.
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Upsample, align with the skip connection and convolve.

        Args:
            x: Features from the layer below, shape (B, in_channels, H, W).
            skip: Encoder features to merge, at twice the resolution.

        Returns:
            Output of shape (B, out_channels, 2H, 2W).
        """
        x = self.up(x)

        # An odd input size makes the upsampled map fall one pixel short of the
        # skip; padding here keeps the concatenation valid for any input.
        height_difference = skip.size()[2] - x.size()[2]
        width_difference = skip.size()[3] - x.size()[3]
        x = F.pad(
            x,
            [
                width_difference // 2,
                width_difference - width_difference // 2,
                height_difference // 2,
                height_difference - height_difference // 2,
            ],
        )

        return self.conv(torch.cat([skip, x], dim=1))


class CustomUNet(nn.Module):
    """U-Net with four downsampling steps and symmetric skip connections.

    Resolution follows H, H/2, H/4, H/8, H/16 and back, with the channel count
    doubling at each step from base_channels.
    """

    def __init__(
        self,
        in_channels: int = config.INPUT_CHANNELS,
        num_classes: int = config.NUM_CLASSES,
        base_channels: int = 64,
    ) -> None:
        """Assemble the network.

        Args:
            in_channels: Channels of the input tensor.
            num_classes: Segmentation classes to predict.
            base_channels: Channels after the first block. Every later stage
                doubles it, so this sets the size of the whole network and is
                the knob to turn when the model does not fit in GPU memory.
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        widths = [base_channels * (2**step) for step in range(5)]

        self.input_block = DoubleConv(in_channels, widths[0])
        self.down1 = Down(widths[0], widths[1])
        self.down2 = Down(widths[1], widths[2])
        self.down3 = Down(widths[2], widths[3])
        self.down4 = Down(widths[3], widths[4])

        self.up1 = Up(widths[4], widths[3])
        self.up2 = Up(widths[3], widths[2])
        self.up3 = Up(widths[2], widths[1])
        self.up4 = Up(widths[1], widths[0])

        self.output_block = nn.Conv2d(widths[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the network.

        Args:
            x: Input of shape (B, in_channels, H, W).

        Returns:
            Logits of shape (B, num_classes, H, W).
        """
        skip1 = self.input_block(x)
        skip2 = self.down1(skip1)
        skip3 = self.down2(skip2)
        skip4 = self.down3(skip3)
        bottleneck = self.down4(skip4)

        x = self.up1(bottleneck, skip4)
        x = self.up2(x, skip3)
        x = self.up3(x, skip2)
        x = self.up4(x, skip1)

        return self.output_block(x)


# =====================================================
# 3 - SEGNET
# =====================================================


class SegNet(nn.Module):
    """Encoder-decoder that carries pooling indices instead of feature maps.

    Where U-Net concatenates whole encoder feature maps into the decoder,
    SegNet stores only *where* each max-pooling picked its value and replays
    those positions when upsampling. The memory saved is substantial; what is
    lost is the encoder's feature content, which the decoder has to rebuild
    from the sparse unpooled map.

    That trade-off is the reason to compare it here: on structures a few pixels
    wide, whether the indices alone carry enough information is an empirical
    question rather than a settled one.
    """

    def __init__(
        self,
        in_channels: int = config.INPUT_CHANNELS,
        num_classes: int = config.NUM_CLASSES,
        base_channels: int = 64,
    ) -> None:
        """Assemble the network.

        Args:
            in_channels: Channels of the input tensor.
            num_classes: Segmentation classes to predict.
            base_channels: Channels after the first encoder block.
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        widths = [base_channels * (2**step) for step in range(4)]

        self.encoder1 = DoubleConv(in_channels, widths[0])
        self.encoder2 = DoubleConv(widths[0], widths[1])
        self.encoder3 = DoubleConv(widths[1], widths[2])
        self.encoder4 = DoubleConv(widths[2], widths[3])

        self.decoder4 = DoubleConv(widths[3], widths[2])
        self.decoder3 = DoubleConv(widths[2], widths[1])
        self.decoder2 = DoubleConv(widths[1], widths[0])
        self.decoder1 = DoubleConv(widths[0], widths[0])

        self.output_block = nn.Conv2d(widths[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the network.

        Args:
            x: Input of shape (B, in_channels, H, W).

        Returns:
            Logits of shape (B, num_classes, H, W).
        """
        features1 = self.encoder1(x)
        pooled1, indices1 = F.max_pool2d(features1, 2, stride=2, return_indices=True)

        features2 = self.encoder2(pooled1)
        pooled2, indices2 = F.max_pool2d(features2, 2, stride=2, return_indices=True)

        features3 = self.encoder3(pooled2)
        pooled3, indices3 = F.max_pool2d(features3, 2, stride=2, return_indices=True)

        features4 = self.encoder4(pooled3)
        pooled4, indices4 = F.max_pool2d(features4, 2, stride=2, return_indices=True)

        # output_size is passed explicitly: unpooling alone cannot know whether
        # an odd dimension was rounded down on the way in.
        x = F.max_unpool2d(pooled4, indices4, 2, stride=2, output_size=features4.size())
        x = self.decoder4(x)

        x = F.max_unpool2d(x, indices3, 2, stride=2, output_size=features3.size())
        x = self.decoder3(x)

        x = F.max_unpool2d(x, indices2, 2, stride=2, output_size=features2.size())
        x = self.decoder2(x)

        x = F.max_unpool2d(x, indices1, 2, stride=2, output_size=features1.size())
        x = self.decoder1(x)

        return self.output_block(x)


# =====================================================
# 4 - PSPNET
# =====================================================


class PyramidPoolingModule(nn.Module):
    """Summarise the feature map at several scales and append the summaries.

    Each level average-pools the map down to a fixed grid — 1x1 is the whole
    image, 6x6 is a coarse partition — reduces its channels, then stretches it
    back to full size. Concatenating all of them gives every output pixel
    access to context far beyond its receptive field.
    """

    def __init__(
        self,
        in_channels: int,
        pool_sizes: Sequence[int] = DEFAULT_POOL_SIZES,
    ) -> None:
        """Build the pyramid.

        Args:
            in_channels: Channels of the incoming feature map.
            pool_sizes: Grid size of each level.
        """
        super().__init__()
        # Each level contributes an equal share, so the concatenation ends up
        # with exactly twice the incoming channels.
        self.level_channels = in_channels // len(pool_sizes)
        self.out_channels = in_channels + self.level_channels * len(pool_sizes)

        self.levels = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_size),
                    nn.Conv2d(in_channels, self.level_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(self.level_channels),
                    nn.ReLU(inplace=True),
                )
                for pool_size in pool_sizes
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool, resize and concatenate.

        Args:
            x: Feature map of shape (B, in_channels, H, W).

        Returns:
            Feature map of shape (B, out_channels, H, W).
        """
        height, width = x.shape[2], x.shape[3]
        outputs = [x]

        for level in self.levels:
            pooled = level(x)
            outputs.append(
                F.interpolate(pooled, size=(height, width), mode="bilinear", align_corners=False)
            )

        return torch.cat(outputs, dim=1)


class CustomPSPNet(nn.Module):
    """Dilated encoder followed by pyramid pooling.

    The encoder stops downsampling at H/8 and widens its receptive field with
    dilated convolutions instead, so context grows without resolution being
    given away.

    Note:
        The classifier still runs at H/8 and its output is stretched back to
        full size in one bilinear step. At 512x512 that means the finest
        feature map is 64x64: a cane eight pixels wide occupies a single pixel
        there, and its boundary is reconstructed by interpolation rather than
        predicted. Expect this architecture to place structures correctly and
        to be the least precise of the three on their edges — which is exactly
        the distinction clDice and IoU are reported to separate.

    Note:
        The 1x1 pyramid level reduces the feature map to a single pixel, which
        leaves batch normalisation with one value per channel. In training mode
        that raises an error unless the batch holds at least two samples, which
        is why the training dataloader drops a trailing batch of one.
    """

    def __init__(
        self,
        in_channels: int = config.INPUT_CHANNELS,
        num_classes: int = config.NUM_CLASSES,
        base_channels: int = 64,
        pool_sizes: Sequence[int] = DEFAULT_POOL_SIZES,
        dropout: float = 0.1,
    ) -> None:
        """Assemble the network.

        Args:
            in_channels: Channels of the input tensor.
            num_classes: Segmentation classes to predict.
            base_channels: Channels after the first convolution. Every later
                stage is a multiple of it, so this scales the whole network and
                is what makes a capacity-matched comparison possible.
            pool_sizes: Grid size of each pyramid level.
            dropout: Dropout applied before the classifier.
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        stem = base_channels
        shallow = base_channels * 2
        middle = base_channels * 4
        deep = base_channels * 8

        # H/2, through a strided convolution rather than pooling.
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, stem, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem),
            nn.ReLU(inplace=True),
            conv_block(stem, stem),
            conv_block(stem, shallow),
        )

        # H/4
        self.layer2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            conv_block(shallow, middle),
            conv_block(middle, middle),
        )

        # H/8, from here on the resolution is held and dilation does the work.
        self.layer3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            conv_block(middle, deep, dilation=2),
            conv_block(deep, deep, dilation=2),
        )

        # Still H/8, wider dilation again.
        self.layer4 = nn.Sequential(
            conv_block(deep, deep, dilation=4),
            conv_block(deep, deep, dilation=4),
        )

        self.pyramid = PyramidPoolingModule(in_channels=deep, pool_sizes=pool_sizes)

        self.bottleneck = nn.Sequential(
            conv_block(self.pyramid.out_channels, deep),
            nn.Dropout2d(p=dropout),
        )
        self.classifier = nn.Conv2d(deep, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the network.

        Args:
            x: Input of shape (B, in_channels, H, W).

        Returns:
            Logits of shape (B, num_classes, H, W).
        """
        input_size = (x.shape[2], x.shape[3])

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.pyramid(x)
        x = self.bottleneck(x)
        x = self.classifier(x)

        return F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)


# =====================================================
# 5 - FACTORY AND INSPECTION
# =====================================================


def create_model(
    architecture: Optional[str] = None,
    in_channels: int = config.INPUT_CHANNELS,
    num_classes: int = config.NUM_CLASSES,
    base_channels: Optional[int] = None,
) -> nn.Module:
    """Build the architecture named by the configuration.

    The width comes from config.MODEL_BASE_CHANNELS, which is set so the three
    architectures end up with comparable parameter counts. Without that the
    comparison would confound architecture with capacity.

    Args:
        architecture: One of AVAILABLE_ARCHITECTURES. Defaults to
            config.ARCHITECTURE.
        in_channels: Channels of the input tensor.
        num_classes: Segmentation classes to predict.
        base_channels: Width override. Defaults to the matched value for the
            architecture.

    Returns:
        The model, on the CPU. Move it to the device explicitly.

    Raises:
        ValueError: If the architecture is unknown, or if a pre-trained encoder
            has been requested.
    """
    if architecture is None:
        architecture = config.ARCHITECTURE

    if config.ENCODER_WEIGHTS is not None:
        raise ValueError(
            f"config.ENCODER_WEIGHTS is '{config.ENCODER_WEIGHTS}', but every "
            "architecture here is trained from scratch. A pre-trained encoder "
            f"expects 3 RGB channels and the input has {in_channels} "
            "heterogeneous ones — see section 8 of docs/pipeline_notes.md."
        )

    builders = {
        "UNet": CustomUNet,
        "SegNet": SegNet,
        "PSPNet": CustomPSPNet,
    }

    if architecture not in builders:
        detail = ""
        if architecture == "UNetPlusPlus":
            detail = " UNetPlusPlus is declared in config but not implemented yet."
        raise ValueError(
            f"Unknown architecture '{architecture}'. "
            f"Available: {list(AVAILABLE_ARCHITECTURES)}.{detail}"
        )

    if base_channels is None:
        base_channels = config.MODEL_BASE_CHANNELS[architecture]

    return builders[architecture](
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=base_channels,
    )


def count_parameters(module: nn.Module) -> int:
    """Count the trainable parameters of a module.

    Args:
        module: Any module, including a submodule.

    Returns:
        Number of parameters that require a gradient.
    """
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def test_model(
    model: nn.Module,
    input_shape: Optional[Tuple[int, int, int, int]] = None,
    num_classes: int = config.NUM_CLASSES,
    check_backward: bool = True,
    verbose: bool = True,
) -> bool:
    """Run one batch through the model and check what comes out.

    Worth doing before any GPU time is spent: a shape mismatch found here costs
    seconds, the same mismatch found on Colab costs a session. The backward
    pass is included because that is where memory actually peaks — a model that
    forwards fine can still exhaust the GPU when the gradients are allocated.

    Args:
        model: Model to exercise.
        input_shape: Batch shape to feed it. Defaults to the configured batch
            size, input channels and image size.
        num_classes: Segmentation classes the output must hold.
        check_backward: Whether to run a backward pass as well.
        verbose: If True, print the report.

    Returns:
        True if the output has the expected shape and the backward pass, when
        requested, completed.
    """
    if input_shape is None:
        input_shape = (
            config.BATCH_SIZE,
            config.INPUT_CHANNELS,
            config.IMG_SIZE[0],
            config.IMG_SIZE[1],
        )

    batch_size, in_channels, height, width = input_shape
    device = next(model.parameters()).device
    expected_shape = (batch_size, num_classes, height, width)
    separator = "=" * 70

    if verbose:
        print(separator)
        print(f"MODEL CHECK — {type(model).__name__}")
        print(separator)
        print(f"  Input channels:   {in_channels}")
        print(f"  Output classes:   {num_classes}")
        print(f"  Parameters:       {count_parameters(model):,}")
        print(f"  Weights size:     ~{count_parameters(model) * 4 / 1024**2:.1f} MB (float32)")
        print(f"  Device:           {device}")
        print(f"  Input shape:      {tuple(input_shape)}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    try:
        sample = torch.randn(input_shape, device=device)
        output = model(sample)

        if verbose:
            print(f"  Output shape:     {tuple(output.shape)}")
            print(f"  Expected:         {expected_shape}")

        if tuple(output.shape) != expected_shape:
            if verbose:
                print(f"\n[FAIL] Output shape mismatch.")
                print(separator)
            return False

        if check_backward:
            output.sum().backward()
            model.zero_grad(set_to_none=True)
            if verbose:
                print("  Backward pass:    OK")

        if device.type == "cuda" and verbose:
            peak = torch.cuda.max_memory_allocated(device) / 1024**3
            print(f"  Peak GPU memory:  {peak:.2f} GB")

    except (RuntimeError, ValueError) as error:
        if verbose:
            print(f"\n[FAIL] {type(error).__name__}: {error}")
            print(separator)
        return False

    if verbose:
        print("\n[OK] Architecture check passed.")
        print(separator)

    return True


def compare_architectures(
    input_shape: Optional[Tuple[int, int, int, int]] = None,
    verbose: bool = True,
) -> Dict[str, Dict[str, object]]:
    """Build every available architecture and report size and validity.

    Args:
        input_shape: Batch shape used for the forward check.
        verbose: If True, print the comparison table.

    Returns:
        Mapping from architecture name to its parameter count and whether the
        shape check passed.
    """
    results: Dict[str, Dict[str, object]] = {}

    for architecture in AVAILABLE_ARCHITECTURES:
        model = create_model(architecture)
        results[architecture] = {
            "parameters": count_parameters(model),
            "passed": test_model(model, input_shape=input_shape, verbose=False),
        }

    if verbose:
        separator = "=" * 70
        print(separator)
        print("ARCHITECTURE COMPARISON")
        print(separator)
        print(f"  {'architecture':<16}{'parameters':>14}{'size (MB)':>12}{'check':>9}")

        for architecture, entry in results.items():
            parameters = int(entry["parameters"])
            print(
                f"  {architecture:<16}{parameters:>14,}"
                f"{parameters * 4 / 1024**2:>12.1f}"
                f"{'OK' if entry['passed'] else 'FAIL':>9}"
            )

        print(separator)

    return results
