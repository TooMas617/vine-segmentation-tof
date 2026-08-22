# Vine Segmentation (ToF) — project context

Semantic segmentation of vine components from Time-of-Flight camera data.
Master's thesis project in Data Science, University of Padova, carried out
during an internship at M31.

## Working agreement

- **Explain, don't just deliver.** The author is learning to write production code. Explain what a change does and why, in enough depth to be understood and defended in a thesis discussion.
- **Targeted edits over rewrites.** Prefer small, reviewable diffs. Do not
  regenerate a whole file when a few lines change.
- **Flag behaviour changes.** If a change alters what the code does — not just
  how it reads — say so explicitly instead of applying it silently.
- **Push back when warranted.** If a request would introduce a methodological
  problem, say so rather than implementing it.
- **Language.** Code, comments, docstrings and commit messages in English.
  Files under `docs/` in Italian: they are working notes for the thesis.
- **Commit messages.** Subject line only by default, imperative, under ~50
  characters. Add a body only when the diff does not explain itself — for
  example an encoding change, or a fix whose reason is not visible in the code.
  Do not add a `Co-Authored-By` trailer.

## Code conventions

- Type hints on every function signature.
- Google-style docstrings: one-line summary, then `Args:`, `Returns:`,
  `Raises:` where applicable.
- `UPPER_SNAKE_CASE` for constants, `snake_case` for functions and variables,
  leading underscore for module-private helpers.
- Descriptive names: `class_index` not `ci`, `label_multi` not `lm`.
- No hardcoded values in logic. Anything tunable lives in `config.py`.
- Comments explain *why*, not *what*. The code already says what it does.
- Section banners in modules and notebooks:

  ```python
  # =====================================================
  # 3 - SECTION NAME
  # =====================================================
  ```
  
- Validate inputs before touching the filesystem, so a failure never leaves
  output folders half written.
- Pipeline functions are idempotent: they skip work when the output already
  exists, and take an `overwrite: bool = False` argument to force a rebuild.

## Where things go

`config.py` holds anything that may change between experiments or is needed by
more than one module. A module keeps what is true by construction — for
example, the LabelMe shape types the code knows how to rasterise.

`config.py` must have **no side effects at import time**. Seeding, folder
creation and similar actions are explicit calls made by the notebook.

## Structure

```
VineProject_ToF/
├── src/
│   ├── config.py            paths, hyperparameters, class definitions, run ids
│   ├── data_preparation.py  label maps, skeletons, split, offline augmentation
│   ├── exploration.py       inventory, class distribution, dataset statistics
│   ├── dataset.py           normalisation, online augmentation, dataloaders
│   ├── models.py            U-Net, SegNet, PSPNet, all from scratch
│   ├── losses.py            Dice, combined, focal, skeleton recall
│   ├── metrics.py           confusion matrix, clDice, history tracker
│   ├── training.py          epoch loop, checkpoints, experiments.csv
│   └── evaluation.py        predictions, error maps, cross-run comparison
├── notebooks/
│   ├── 01_Data_Preparation.ipynb
│   ├── 02_EDA.ipynb
│   ├── 03_Training.ipynb
│   └── 04_Evaluation.ipynb
├── docs/
│   └── pipeline_notes.md    design decisions and rationale (Italian)
├── data/                    NOT under version control
│   ├── raw/                 intensity, depth, confidence, annotations
│   ├── processed/           label maps, skeletons, colored masks,
│   │                        class_weights.json, channel_stats.json
│   ├── splits/              splits.json, shared by every notebook
│   └── augmented/           train / val / test, materialised
├── experiments/             one folder per run — NOT under version control
├── results/                 experiments.csv — NOT under version control
└── requirements.txt
```

Notebooks stay thin: they import from `src/` and call functions. Logic lives in
the modules, so the other notebooks can reuse it.

## Environment

```bash
conda create -n vine python=3.11 -y
conda activate vine
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

`albumentations` is **not installed locally**: on Windows its `stringzilla`
dependency needs a C++ compiler. It is only used for online augmentation during
training, which runs on Colab. `config.print_environment_info()` reports
missing optional libraries instead of raising.

Data preparation runs locally (~20 s for the whole dataset). Training will run
on Colab for GPU memory. All runs feeding the final comparison table must use
the same hardware: batch size is constrained by GPU memory and affects results.

## Dataset

100 images, 5 classes: `Background`, `Cane`, `Cordon`, `Node`, `Other`.
Split 70 / 15 / 15 (seed 42, hash `62de0ec4`). This is the **final** dataset:
it was brought to its definitive form before any experiment, because adding or
removing a single image redraws the whole partition and makes earlier runs
incomparable.

Modalities: intensity (1 channel, `uint8`, 480×640), depth (3 channels,
colorised), confidence (1 channel). Total input: 5 channels. Part of the raw
files store the single-channel modalities as three identical planes;
`data_preparation.read_modality_image()` reconciles them, so every consumer
sees the channel count `config.MODALITY_CHANNELS` declares.

Class distribution is heavily imbalanced: Background 84.1 %, Cane 11.1 %,
Cordon 3.6 %, Other 1.0 %, Node 0.27 %. `Node` is also absent from a quarter of
the images.

Annotations are LabelMe JSON. Two conventions coexist: 39 files use Italian
labels and a finer taxonomy, 61 use English labels with better boundary
accuracy. Both map onto the canonical classes through `config.LABEL_ALIASES`.

## Decisions already made — do not silently reverse

Full rationale in `docs/pipeline_notes.md`.

- **Drawing priority differs between masks.** Multiclass: `Cane` wins over
  `Cordon`, because the cane is what is actually visible. Binary: `Cordon`
  wins, because that mask represents the cordon as a continuous structure.
- **The cordon skeleton is extracted from the unified binary mask**, then
  occluded pixels are removed. Skeletonising only visible pixels would produce
  a broken centreline.
- **`dilation_kernel_size` stays at 2**, despite an even kernel being
  asymmetric. The tube it produces covers 34.8 % of the cane mask; at 3 that
  becomes 53.6 % and the skeleton recall loss stops rewarding the centreline
  and turns into plain recall on the class. A kernel of 1 is not an option
  either: `cv2.dilate` with a 1×1 kernel is the identity, so there would be no
  tube at all — the very thing the reference method adds for gradient
  stability. The asymmetry is cosmetic by comparison: connectivity comes from
  the thinning, not from the dilation.
- **The split is computed on original stems, before augmentation**, so flipped
  copies of one image cannot land in different partitions.
- **Offline augmentation covers all modalities**, regardless of which ones the
  current model consumes. The augmented dataset describes the data, not the
  experiment.
- **Raw modalities are read with `IMREAD_UNCHANGED`**, preserving bit depth and
  channel count. Plain `imread()` silently converts to 8-bit BGR.
- **Only flips are used offline.** They preserve labels exactly: no
  interpolation, no invented border pixels.
- **Channel counts are reconciled at read time**, in
  `read_modality_image()`, not by rewriting `data/raw` and not in the Dataset.
  One rule, one place, three consumers. Every file in `data/augmented` is
  decoded and re-encoded for this reason — `shutil.copy2` would carry the
  source layout straight through.
- **Descriptive statistics cover every split; training parameters come from
  the training split alone.** Class weights and channel statistics feed the
  model, so measuring them on the whole dataset would leak validation and test
  information into training.
- **Statistics are measured on the original images, not the augmented ones.**
  A flip permutes pixels without changing their values, so the flipped copies
  carry identical counts at four times the reading cost.
- **Saved statistics record the `splits_hash` they were measured on**, so
  values left over from a different partition are reported instead of silently
  applied. The caller passes the current hash in: `config.py` cannot read the
  split itself without a circular import.
- **A sample is taken out of the dataset by moving its files out of
  `data/raw/`, never by filtering it in code.** The pipeline starts from the
  JSON files in `data/raw/annotations`, so a moved sample disappears from
  labels, split and augmented set on its own — and comes back the same way.

## Status and next steps

Notebooks 01 and 02 are complete and have run end to end on the full dataset.
Notebook 02 writes `data/processed/class_weights.json`,
`data/processed/channel_stats.json` and the figures under
`results/figures/eda/`. Both JSON files are stamped with the `splits_hash` and
reload through `config.load_class_weights` / `config.load_channel_stats`.

`dataset.py`, `losses.py` and `metrics.py` are written and verified locally.
The reference papers for clDice and the Skeleton Recall Loss live in
`reference/` (not versioned); section 7 of the notes records their formulas,
every deviation from them and the reasoning, and should be read before
touching either module.

`models.py` implements U-Net (31 M parameters), SegNet (7 M) and PSPNet (14 M),
all in plain PyTorch and trained from scratch, so
`segmentation_models_pytorch` is not a dependency and the models can be checked
locally. `UNetPlusPlus` is named in `config.ARCHITECTURE` but is deliberately
not implemented. The whole chain — dataloader, model, loss, metrics — has been
run end to end on real samples.

`training.py` and notebook 03 are written and exercised end to end on the CPU:
optimiser and scheduler factories, checkpointing, the epoch loop, early
stopping and the results row. Each run writes `experiments/<run_id>/` and one
of 72 columns into `results/experiments.csv`, stamped with the git commit and
the `splits_hash`.

`evaluation.py` and notebook 04 rebuild a finished run from its own
`config.json`, predict the test split, and produce the error maps and the
cross-run comparison. All four notebooks and all eight modules are written and
exercised locally.

Next: **run the experiments on Colab.** Three architectures at matched
capacity, plus the ablation with and without the skeleton recall loss. Nothing
further needs writing to start.

The experiments must include an **ablation with and without the skeleton
recall loss**. Theory predicts it trades precision for recall, and the model
already over-segments canes, so report precision and recall per class: mean
IoU alone would hide the effect, and clDice is blind to radius inflation by
design. Section 7.6 of the notes states the prediction to test.

Open points are tracked at the end of `docs/pipeline_notes.md`. The most
consequential is whether to drop the `Node` class: measurements are in section 5
of the notes, and the mechanism would be `config.LABELS_TO_IGNORE`, not a merge
into `Cordon` — only 55 % of node pixels lie on the cordon.

## Deadline

Thesis due 17th September 2026. Prefer solutions that are correct and
defensible over solutions that are elegant but time-consuming.
