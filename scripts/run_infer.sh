#!/usr/bin/env bash
set -e
python -m m2caller.infer \  --records_dir data/test_records \  --ckpt checkpoints/best.pt \  --out_fasta outputs/out.fasta \  --half
