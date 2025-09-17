# M2Caller (Open-Source Reference Implementation)

A minimal, self-contained PyTorch implementation of the **M2Caller** idea: a convolutional signal sampler + a light **state-space** encoder (Mamba2-inspired) + **CTC** decoding for Oxford Nanopore basecalling.

> This repository is engineered to be easy to run end-to-end on `.npy` signal chunks, matching the paper's segmentation workflow. It avoids heavy external deps (beam decoders, custom CUDA ops) while keeping the core modeling choices clear and hackable.

## Highlights

- **Signal Sampler:** two 1D conv layers reduce length, expand channels.
- **Mamba-like Encoder:** linear-time *selective state* update block (A·h + B·x) with gated residual and depthwise conv mixing, stacked `L` times.
- **CTC Training:** 5-class logits (blank + A/C/G/T), standard `nn.CTCLoss`.
- **Greedy Decoding:** fast, dependency-free decoding for inference.
- **Numpy Records I/O:** reads `records_dir/*.npy` where each file stores a variable-length array of shape `(num_windows, window_len, 1)` or `(num_windows, window_len)`.

If you already exported records from your preprocessing (or your previous pipeline), just point `--records_dir` to it and run.

---

## Quickstart

### 1) Install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# or with CUDA build of torch if you have GPUs
pip install -r requirements.txt
```

### 2) Data Layout

We assume a directory of numpy records:
```
records_dir/
  read1.npy
  read2.npy
  ...
```
Each `.npy` contains either a `(N, W, 1)` or `(N, W)` float array (zero-mean, unit-variance), where **W=4096** by default.

You can create these via `python -m m2caller.data.preprocess_fast5 --fast5_root <dir> --out_dir records_dir` (requires `ont-fast5-api`) or by converting your own raw signals to `.npy` with `preprocess_numpy.py` (no fast5 dependency).

### 3) Train

```bash
python -m m2caller.train   --train_records data/train_records   --val_records data/val_records   --epochs 10 --batch_size 16   --d_model 512 --layers 2 --heads 4 --ssm_dim 256   --segment_len 4096   --save_dir checkpoints
```

### 4) Inference

```bash
python -m m2caller.infer   --records_dir data/test_records   --ckpt checkpoints/best.pt   --out_fasta outputs/out.fasta   --half
```

### 5) Evaluate (optional)

```bash
python -m m2caller.decoding.eval_identity   --fasta outputs/out.fasta   --ref reference.fasta
```

> Note: Identity/cer evaluation requires an external aligner; we provide a tiny wrapper that shells out to `edlib` if available else performs a naive DP edit distance (slow).

---

## Repo Structure

```
m2caller/
  data/
    preprocess_fast5.py      # FAST5 -> numpy records (optional; needs ont-fast5-api)
    preprocess_numpy.py      # raw .npy -> chunked records
  decoding/
    greedy.py                # Greedy CTC decoder
    eval_identity.py         # Read identity / CER estimate
  models/
    m2caller.py              # SignalSampler + Mamba2Like stacks + CTC head
  utils/
    dataset.py               # NumpyRecordsDataset + collate
    constants.py             # tokens, blank index, padding
    common.py                # seed, device helpers, logging
  train.py                   # Training entry
  infer.py                   # Inference entry
requirements.txt
LICENSE
README.md
```

---

## Notes

- This code mirrors key design choices reported in the paper (segment length 4096, conv front-end, CTC) and maintains linear-time encoder blocks to balance accuracy/throughput.
- Beam search can be added later (ctcdecode / fast-ctc-decode), but greedy is usually fine for first-pass prototyping.

Enjoy hacking!
