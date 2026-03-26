#!/bin/bash

mkdir -p results/imagenet_run

nohup python scripts/imagenet_impl.py \
  --data_dir /data/CPE_487-587/imagenet-1k-arrow \
  --output_dir results/imagenet_run \
  --epochs 1000 \
  --train_ratio 0.01 \
  --val_ratio 0.01 \
  --batch_size 128 \
  --num_workers 4 \
  --lr 0.01 \
  > results/imagenet_run/train.log 2>&1 &