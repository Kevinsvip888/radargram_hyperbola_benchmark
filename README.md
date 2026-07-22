# Radargram Hyperbola Segmentation Benchmark

A clean Python project for benchmarking semantic and instance segmentation models on GPR radargram images containing hyperbola-shaped rebar responses.

The project supports:

- **U-Net** for semantic segmentation.
- **UNet++** for semantic segmentation.
- **SegFormer** for semantic segmentation.
- **DINOv3 + lightweight decoder** for semantic segmentation.
- **Mask R-CNN** for instance segmentation.
- **Mask2Former** for instance segmentation.
- **YOLO11-seg** for instance segmentation through the Ultralytics API.
- **SAM 2 / SAM 2.1** prompt-based evaluation and fine-tuning data export.
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

Optional model dependencies:

```bash
# YOLO11-seg
pip install ultralytics

# SegFormer, Mask2Former, and DINOv3 wrappers
pip install transformers accelerate

# SAM 2 / SAM 2.1: install from the official repository
# https://github.com/facebookresearch/sam2
```

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


## Input resizing, letterboxing, and coordinates

All PyTorch-based models still receive a fixed tensor size from each config:

```yaml
input:
  image_size: [512, 1024]   # [height, width]
  resize_mode: resize       # resize or letterbox
  pad_value: 0
  allow_upscale: true       # used by letterbox mode
  grayscale: false
```

Use `resize_mode: resize` when you want the old behavior: each image and mask is directly stretched to `image_size`. This preserves compatibility with checkpoints already trained using earlier project versions.

Use `resize_mode: letterbox` when your raw radargrams have different aspect ratios and you want to preserve the original hyperbola geometry. In this mode, the image and all masks are scaled with the same factor and padded to the requested model size. Mask padding is always background; image padding uses `pad_value`.

Set `allow_upscale: false` when you do **not** want smaller images to be enlarged. Then any image that is already smaller than the target size is kept at its original size and centered on the canvas with padding only. Larger images are still downscaled as needed.

Example:

```yaml
input:
  image_size: [512, 1024]
  resize_mode: letterbox
  pad_value: 0
  allow_upscale: false
  grayscale: false
```

During prediction, the project records the spatial preprocessing metadata and maps predicted masks back to the original image size before saving. Therefore:

```text
objects/object_001_pred.png
coordinates/object_001_pixels.csv
mask_all_pred.png
overlay.png
```

are saved in the **original radargram coordinate system**, not the resized or padded model-input coordinate system. The file `prediction.json` includes `original_image_size`, `saved_mask_size`, `coordinate_system`, and the preprocessing metadata used for the inverse mapping.

YOLO11-seg uses Ultralytics' own resizing/letterboxing internally and already returns masks in the original image size. SAM 2 prompted utilities also operate on the original image size.

## Step 4: Train models

Semantic models:

```bash
python train.py --config configs/unet.yaml
python train.py --config configs/unetpp.yaml
python train.py --config configs/segformer.yaml
python train.py --config configs/dinov3.yaml
```

Instance models:

```bash
python train.py --config configs/mask_rcnn.yaml
python train.py --config configs/mask2former.yaml
python train.py --config configs/yolo11_seg.yaml
```

SAM 2 is different from the other models: it is a promptable foundation model with its own official training/fine-tuning code. This project therefore supports SAM 2 in two practical ways:

```bash
# Export a COCO-style bundle that can be adapted for the official SAM 2 training code.
python scripts/export_sam2_finetune_data.py \
  --processed-root dataset/processed \
  --splits-dir dataset/splits \
  --output-root dataset/sam2_finetune \
  --copy-images

# Evaluate SAM 2 as a box-prompted mask-refinement baseline using GT boxes.
python scripts/evaluate_sam2_prompted.py \
  --config configs/sam2.yaml \
  --split test
```

Ground-truth prompted SAM 2 is not an automatic detection benchmark. It measures how well SAM 2 refines masks when it is given correct object boxes. For automatic prediction, use boxes from a trained proposal model such as YOLO11-seg or Mask R-CNN, then feed those boxes to SAM 2.

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

The project deliberately avoids vertical flips, arbitrary rotations, strong elastic deformation, and perspective transforms because they can make the GPR geometry physically unrealistic. Bounding boxes for instance models are generated after augmentation, so boxes always match the transformed object masks.

You can quickly inspect augmentation quality with:

```bash
python scripts/preview_augmentations.py \
  --config configs/mask_rcnn.yaml \
  --output-dir outputs/augmentation_preview \
  --num-samples 8
```

## YOLO11-seg dataset export

`train.py --config configs/yolo11_seg.yaml` automatically exports the processed dataset to Ultralytics YOLO segmentation format before training. You can also run the export manually:

```bash
python scripts/export_yolo11_seg.py \
  --processed-root dataset/processed \
  --splits-dir dataset/splits \
  --output-root dataset/yolo11_seg
```

The export converts each object mask into normalized polygon labels and creates:

```text
dataset/yolo11_seg/
  data.yaml
  images/train, images/val, images/test
  labels/train, labels/val, labels/test
```

## Step 5: Evaluate

```bash
python evaluate.py --config configs/unet.yaml --checkpoint outputs/unet/best.pt --split test
python evaluate.py --config configs/unetpp.yaml --checkpoint outputs/unetpp/best.pt --split test
python evaluate.py --config configs/dinov3.yaml --checkpoint outputs/dinov3/best.pt --split test
python evaluate.py --config configs/mask_rcnn.yaml --checkpoint outputs/mask_rcnn/best.pt --split test
python evaluate.py --config configs/yolo11_seg.yaml --checkpoint outputs/yolo11_seg/weights/best.pt --split test
python scripts/evaluate_sam2_prompted.py --config configs/sam2.yaml --split test
```

## Step 6: Predict and export masks/pixel coordinates

```bash
python predict.py \
  --config configs/mask_rcnn.yaml \
  --checkpoint outputs/mask_rcnn/best.pt \
  --input-root dataset/processed/images \
  --output-root outputs/predictions/mask_rcnn \
  --split test \
  --splits-dir dataset/splits
```

predict validation images only:
```bash
python predict.py \
  --config configs/mask_rcnn.yaml \
  --checkpoint outputs/mask_rcnn/best.pt \
  --input-root dataset/processed/images \
  --output-root outputs/predictions/mask_rcnn_val \
  --split val
  
`--input-root dataset/processed/images` contains all processed images. Add `--split train`, `--split val`, or `--split test` to predict only the image IDs listed in the corresponding split file. Without `--split`, `predict.py` intentionally predicts every image under `--input-root`.

For Mask R-CNN, `evaluate.py` and `predict.py` automatically disable COCO-pretrained weight loading before loading your supplied `--checkpoint`. This prevents unnecessary internet downloads during evaluation and prediction.

YOLO11-seg prediction uses the same command style:

```bash
python predict.py \
  --config configs/yolo11_seg.yaml \
  --checkpoint outputs/yolo11_seg/weights/best.pt \
  --input-root dataset/processed/images \
  --output-root outputs/predictions/yolo11_seg \
  --split test \
  --splits-dir dataset/splits
```

SAM 2 prompted prediction on a processed split:

```bash
python scripts/predict_sam2_prompted.py \
  --config configs/sam2.yaml \
  --split test \
  --output-root outputs/predictions/sam2_prompted
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

For the PyTorch-based models, these saved masks and CSV files are mapped back to the original radargram size using the spatial metadata from preprocessing. This is true for both `resize_mode: resize` and `resize_mode: letterbox`.

## Model notes

- **U-Net, UNet++, SegFormer, and DINOv3** produce one semantic mask for all hyperbolas. Individual hyperbola masks are extracted by connected components.
- **Mask R-CNN, Mask2Former, and YOLO11-seg** directly predict instance masks.
- **SAM 2** is included as a prompt-based mask-refinement/foundation-model baseline and a fine-tuning data export route, not as a normal `train.py` model.
- For final per-hyperbola pixel coordinates, prefer instance models first. Use semantic models as useful baselines.
