# Sample data

Five raw frames from the dataset, one per acquisition session, all taken from the **test split**. They show what the model receives as input; the annotations are not included.

Frames were captured with a Basler blaze-101 Time-of-Flight camera mounted on an over-the-row gantry, looking down at dormant vines, without an artificial backdrop.

## Contents

| Folder | Modality | Format |
|---|---|---|
| `intensity/` | Reflected infrared intensity | 640×480, 8-bit grayscale |
| `depth/` | Distance from the camera, colorized over 300–2200 mm | 640×480, 8-bit RGB |
| `confidence/` | Per-pixel reliability of the depth measurement | 640×480, 8-bit grayscale |

Stacked together, the three modalities form the 5-channel input of the model (1 + 3 + 1).

Frames from the March sessions (`blaze1_*`) store intensity and confidence as three identical RGB planes. `src/data_preparation.read_modality_image()` reduces them to a single channel.

## Frames

| Frame | Session |
|---|---|
| `blaze-101-25367037-5013` | 18 February 2026 |
| `blaze-101-25367037-20010` | 25 February 2026 |
| `blaze-101-25367037-20084` | 11 March 2026 |
| `blaze1_20260321_164929` | 21 March 2026 |
| `blaze1_20260324_171004` | 24 March 2026 |
