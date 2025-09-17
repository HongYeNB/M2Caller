import argparse, os, numpy as np, h5py, glob

def extract_signal(f5_path):
    with h5py.File(f5_path, 'r') as f:
        # Simple extractor for Raw signal (may differ across chemistries)
        raw = f['Raw/Reads']
        first = list(raw.keys())[0]
        signal = raw[first]['Signal'][()]
        return signal.astype(np.float32)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--fast5_root', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--segment_len', type=int, default=4096)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    paths = glob.glob(os.path.join(args.fast5_root, '**', '*.fast5'), recursive=True)
    for pth in paths:
        try:
            sig = extract_signal(pth)
            sig = (sig - sig.mean()) / (sig.std() + 1e-6)
            L = sig.shape[0]
            N = (L + args.segment_len - 1) // args.segment_len
            chunks = []
            for i in range(N):
                s = i*args.segment_len
                e = min(L, s+args.segment_len)
                x = np.zeros((args.segment_len,), dtype=np.float32)
                x[:e-s] = sig[s:e]
                chunks.append(x[:,None])  # (W,1)
            out = np.stack(chunks, axis=0)  # (N,W,1)
            rid = os.path.splitext(os.path.basename(pth))[0]
            np.save(os.path.join(args.out_dir, f"{rid}.npy"), out)
        except Exception as e:
            print('skip', pth, e)

if __name__ == '__main__':
    main()
