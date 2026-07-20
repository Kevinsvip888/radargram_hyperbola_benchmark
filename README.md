# Radargram Hyperbola Segmentation Benchmark

A clean PyTorch project for benchmarking semantic and instance segmentation models on GPR radargram images containing hyperbola-shaped rebar responses.

The project supports:

- **U-Net** for semantic segmentation.
- **SegFormer** for semantic segmentation.
- **Mask R-CNN** for instance segmentation.
- **Mask2Former** for instance segmentation.
- Automatic bounding-box generation from object masks.
- Semantic mask generation from object masks when needed.
- Prediction export as masks, overlays, JSON, and per-instance pixel-coordinate CSV files.
- Optional safe radargram data augmentation during training.

## Expected raw dataset structure

```text
DATASET_ROOT/
  sim_0001/
    radargram_raw_Ez.png
    MASK/
      radargram_raw_Ez_mask_semantic.png
      objects/
        radargram_raw_Ez_object_001_mask.png
        radargram_raw_Ez_object_002_mask.png
        radargram_raw_Ez_object_003_mask.png

  sim_0002/
    radargram_raw_Ez.png
    MASK/
      radargram_raw_Ez_mask_semantic.png
      objects/
        radargram_raw_Ez_object_001_mask.png
```

Only object masks are essential. The semantic mask can be generated automatically as the union of all object masks.

## Installation

```bash
cd radargram_hyperbola_benchmark
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Install PyTorch according to your CUDA version from the official PyTorch installation page if the generic `requirements.txt` command does not match your system.

## Step 1: Prepare the dataset

For one raw dataset root:

```bash
python scripts/prepare_dataset.py \
  --raw-root /path/to/DATASET_ROOT \
  --processed-root dataset/processed \
  --create-semantic-if-missing \
  --repair-semantic
```

For several raw dataset roots, you can either run the command several times:

```bash
python scripts/prepare_dataset.py \
  --raw-root /path/to/DATASET_ROOT_A \
  --processed-root dataset/processed \
  --create-semantic-if-missing \
  --repair-semantic

python scripts/prepare_dataset.py \
  --raw-root /path/to/DATASET_ROOT_B \
  --processed-root dataset/processed \
  --create-semantic-if-missing \
  --repair-semantic
```

Or process several roots in one command:

```bash
python scripts/prepare_dataset.py \
  --raw-root /path/to/DATASET_ROOT_A /path/to/DATASET_ROOT_B \
  --processed-root dataset/processed \
  --create-semantic-if-missing \
  --repair-semantic
```

The default write mode is append-safe. Existing `annotations/manifest.csv` rows are kept, new rows are merged in, and duplicate `image_id` rows are replaced by the newest version. To avoid collisions when different raw roots both contain names such as `sim_0001`, the processed `image_id` automatically includes a source prefix, for example:

```text
DATASET_ROOT_A_1a2b3c4d__sim_0001
DATASET_ROOT_B_9e8f7a6b__sim_0001
```

If you want a readable custom prefix, use it with one raw root at a time:

```bash
python scripts/prepare_dataset.py \
  --raw-root /path/to/DATASET_ROOT_A \
  --processed-root dataset/processed \
  --source-prefix synthetic_train_A \
  --create-semantic-if-missing \
  --repair-semantic
```

Use `--write-mode replace` to write a manifest containing only the current input roots. Use `--reset` only when you intentionally want to delete the old processed dataset first.

This creates:

```text
dataset/processed/
  images/
  semantic_masks/
  instance_masks/
  annotations/manifest.csv
  annotations/instances_all.json
```

## Step 2: Check the dataset

```bash
python scripts/check_dataset.py \
  --processed-root dataset/processed \
  --save-overlays outputs/dataset_checks
```

## Step 3: Create train/val/test splits

Recreate the split files after every append or replacement of the processed manifest, otherwise training may use an older split list that does not include the newly prepared images.

```bash
python scripts/create_splits.py \
  --manifest dataset/processed/annotations/manifest.csv \
  --output-dir dataset/splits \
  --train 0.70 \
  --val 0.15 \
  --test 0.15 \
  --seed 42
```

## Step 4: Train models

```bash
python train.py --config configs/unet.yaml
python train.py --config configs/segformer.yaml
python train.py --config configs/mask_rcnn.yaml
python train.py --config configs/mask2former.yaml
```

## Data augmentation

Data augmentation is controlled from each YAML config under the `augmentation:` section. It is applied **only to the training split**. Validation, testing, and prediction remain deterministic.

The implemented augmentations are intentionally conservative for radargrams:

```yaml
augmentation:
  enabled: true

  # Applied to image + semantic/object masks using identical parameters.
  horizontal_flip_p: 0.5
  shift_scale_p: 0.4
  shift_limit: 0.03
  scale_limit: 0.05

  # Applied only to the radargram image, not to masks.
  brightness_contrast_p: 0.5
  brightness_limit: 0.12
  contrast_limit: 0.12
  gamma_p: 0.3
  gamma_limit: [0.90, 1.10]
  gaussian_noise_p: 0.3
  noise_std_limit: [0.005, 0.030]
  gaussian_blur_p: 0.15
  blur_kernel_size: 3
```

The project deliberately avoids vertical flips, arbitrary rotations, strong elastic deformation, and perspective transforms because they can make the GPR geometry physically unrealistic. Bounding boxes for Mask R-CNN are generated after augmentation, so boxes always match the transformed object masks.

You can quickly inspect augmentation quality with:

```bash
python scripts/preview_augmentations.py \
  --config configs/mask_rcnn.yaml \
  --output-dir outputs/augmentation_preview \
  --num-samples 8
```


## Step 5: Evaluate

```bash
python evaluate.py --config configs/unet.yaml --checkpoint outputs/unet/best.pt --split test
python evaluate.py --config configs/mask_rcnn.yaml --checkpoint outputs/mask_rcnn/best.pt --split test
```

## Step 6: Predict and export masks/pixel coordinates

```bash
python predict.py \
  --config configs/mask_rcnn.yaml \
  --checkpoint outputs/mask_rcnn/best.pt \
  --input-root dataset/processed/images \
  --output-root outputs/predictions/mask_rcnn
```

For each image, prediction saves:

```text
outputs/predictions/MODEL_NAME/sim_0001/
  overlay.png
  mask_all_pred.png
  objects/
    object_001_pred.png
    object_002_pred.png
  coordinates/
    object_001_pixels.csv
    object_002_pixels.csv
  prediction.json
```

Each pixel-coordinate CSV contains:

```text
x,y
120,45
121,45
122,45
```

## Notes on model outputs

- **U-Net and SegFormer** predict one merged hyperbola mask. Individual hyperbola instances are extracted by connected components.
- **Mask R-CNN and Mask2Former** predict one mask per hyperbola directly.
- Mask R-CNN training requires boxes, but boxes are generated automatically from object masks.

## Recommended first benchmark

Start with:

```bash
python train.py --config configs/unet.yaml
python train.py --config configs/mask_rcnn.yaml
```

Then add SegFormer and Mask2Former after the dataset and baseline training are verified.
