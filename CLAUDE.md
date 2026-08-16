# Vine Segmentation (ToF) — project context

Semantic segmentation of vine components from Time-of-Flight camera data.
Master's thesis project in Data Science, University of Padova, carried out
during an internship at M31.

## Working agreement

- **Explain, don't just deliver.** The author is learning to write production
  code. Explain what a change does and why, in enough depth to be understood
  and defended in a thesis discussion.
- **Targeted edits over rewrites.** Prefer small, reviewable diffs. Do not
  regenerate a whole file when a few lines change.
- **Flag behaviour changes.** If a change alters what the code does — not just
  how it reads — say so explicitly instead of applying it silently.
- **Push back when warranted.** If a request would introduce a methodological
  problem, say so rather than implementing it.
- **Language.** Code, comments, docstrings and commit messages in English.
  Files under `docs/` in Italian: they are working notes for the thesis.

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
│   └── data_preparation.py  label maps, skeletons, split, offline augmentation
├── notebooks/
│   └── 01_Data_Preparation.ipynb
├── docs/
│   └── pipeline_notes.md    design decisions and rationale (Italian)
├── data/                    NOT under version control
│   ├── raw/                 intensity, depth, confidence, annotations
│   ├── processed/           label maps, skeletons, colored masks
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

Data preparation runs locally (~20 s for 93 images). Training will run on Colab
for GPU memory. All runs feeding the final comparison table must use the same
hardware: batch size is constrained by GPU memory and affects results.

## Dataset

93 images, 5 classes: `Background`, `Cane`, `Cordon`, `Node`, `Other`.
Split 65 / 14 / 14 (seed 42, hash `d57f79eb`).

Modalities: intensity (1 channel, `uint8`, 480×640), depth (3 channels,
colorised), confidence (1 channel). Total input: 5 channels.

Annotations are LabelMe JSON with Italian labels, mapped to canonical classes
through `config.LABEL_ALIASES`.

## Decisions already made — do not silently reverse

Full rationale in `docs/pipeline_notes.md`.

- **Drawing priority differs between masks.** Multiclass: `Cane` wins over
  `Cordon`, because the cane is what is actually visible. Binary: `Cordon`
  wins, because that mask represents the cordon as a continuous structure.
- **The cordon skeleton is extracted from the unified binary mask**, then
  occluded pixels are removed. Skeletonising only visible pixels would produce
  a broken centreline.
- **`dilation_kernel_size` stays at 2**, despite being asymmetric. At 3, three
  images reach over 80 % skeleton coverage of the cane mask, which degenerates
  clDice into an area-overlap measure. The affected images are young vineyard,
  where canes are thinner.
- **The split is computed on original stems, before augmentation**, so flipped
  copies of one image cannot land in different partitions.
- **Offline augmentation covers all modalities**, regardless of which ones the
  current model consumes. The augmented dataset describes the data, not the
  experiment.
- **Raw modalities are read with `IMREAD_UNCHANGED`**, preserving bit depth and
  channel count. Plain `imread()` silently converts to 8-bit BGR.
- **Only flips are used offline.** They preserve labels exactly: no
  interpolation, no invented border pixels.

## Status and next steps

Notebook 01 is complete and has run end to end on the full dataset.

Next:
1. **Notebook 02 (EDA)** — dataset statistics, class distribution, class weight
   computation (saved to disk via `config.save_class_weights`, reloaded during
   training), augmentation and skeleton visualisation.
2. **Training modules** — `dataset.py`, `models.py`, `losses.py`, `metrics.py`,
   extracted from the original monolithic notebook.
3. **Notebook 03 (Training)** — training loop, per-run experiment folder,
   automatic logging to `results/experiments.csv`.
4. **Notebook 04 (Evaluation)** — test set metrics, prediction visualisation,
   comparison table sorted by mean IoU.

Open points are tracked at the end of `docs/pipeline_notes.md`. The most
immediate: the `Dataset` class must handle single-channel intensity, since the
original code may have assumed three channels.

## Deadline

Thesis due late September 2026. Prefer solutions that are correct and
defensible over solutions that are elegant but time-consuming.
