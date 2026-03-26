# HW03Part06 - ImageNet CNN Pipeline

This repository contains the solution for **Section 6: Convolutional Neural Network for ImageNet** from HW03. The goal of this part is to build a complete CNN training pipeline for image classification on ImageNet, including data loading, preprocessing, training, validation, plotting, model export, and inference.

## Overview

The implemented pipeline includes:

- a reusable convolution block
- a CNN model for ImageNet classification
- a training and validation pipeline
- plot generation for loss and accuracy
- ONNX model export
- inference on unseen images

The model was trained on the cached ImageNet dataset available on Lovelace.

## Dataset

Dataset path used:

```bash
/data/CPE_487-587/imagenet-1k-arrow
```

Original dataset size:

- Training samples: **1,281,167**
- Validation samples: **50,000**
- Number of classes: **1000**

For the final run, a very small subset of the dataset was used in order to complete the full training pipeline efficiently.

Final run settings:

- Epochs: **500**
- Training ratio: **0.0002**
- Validation ratio: **0.0002**
- Batch size: **32**
- Number of workers: **2**
- Learning rate: **0.01**

Approximate subset size used:

- Training subset: **256**
- Validation subset: **10**

## Final Run Command

```bash
export PYTHONPATH=src
python scripts/imagenet_impl.py \
  --data_dir /data/CPE_487-587/imagenet-1k-arrow \
  --output_dir results/imagenet_final \
  --epochs 500 \
  --train_ratio 0.0002 \
  --val_ratio 0.0002 \
  --batch_size 32 \
  --num_workers 2 \
  --lr 0.01
```

## How to Run

Run training directly with:

```bash
export PYTHONPATH=src
python scripts/imagenet_impl.py \
  --data_dir /data/CPE_487-587/imagenet-1k-arrow \
  --output_dir results/imagenet_final \
  --epochs 500 \
  --train_ratio 0.0002 \
  --val_ratio 0.0002 \
  --batch_size 32 \
  --num_workers 2 \
  --lr 0.01
```

Or run the background script:

```bash
bash scripts/imagenet_impl.sh
```

## Inference

After training, run inference on a new image with:

```bash
export PYTHONPATH=src
python scripts/imagenet_inference.py \
  --onnx_model results/imagenet_final/imagenet_cnn.onnx \
  --image_path path/to/your/image.jpg \
  --data_dir /data/CPE_487-587/imagenet-1k-arrow
```

## Outputs

The pipeline produces:

- example training and validation images
- loss and accuracy plots
- ONNX model
- summary file with dataset and model statistics

## Notes

Because only a very small subset of ImageNet was used, the model does not achieve strong validation accuracy. This is expected. The purpose of this part was to complete a working end-to-end CNN training pipeline.

## Section 6.3 Answers

1. Original number of training samples: **1,281,167**
2. Original number of validation samples: **50,000**
3. Number of classes: **1000**
4. Total trainable parameters: **5,464,040**