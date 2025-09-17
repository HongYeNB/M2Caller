#!/usr/bin/env bash
set -e
python -m m2caller.train \
  --train_records data/train_records \  --val_records data/val_records \  --epochs 10 --batch_size 16 \  --d_model 512 --layers 2 --heads 4 --ssm_dim 256 \  --segment_len 4096 \  --save_dir checkpoints
