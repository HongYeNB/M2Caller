import argparse, os, numpy as np

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--signals', required=True, help='Path to a big numpy .npy of shape [L] or [N,W] or a folder of .npy arrays')
    p.add_argument('--out_dir', required=True)
    p.add_argument('--segment_len', type=int, default=4096)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    def process_array(arr, rid='read'):
        if arr.ndim == 1:
            L = arr.shape[0]
            N = (L + args.segment_len - 1) // args.segment_len
            chunks = []
            for i in range(N):
                s = i*args.segment_len
                e = min(L, s+args.segment_len)
                x = np.zeros((args.segment_len,), dtype=np.float32)
                x[:e-s] = arr[s:e]
                chunks.append(x[:,None])  # (W,1)
            out = np.stack(chunks, axis=0)  # (N,W,1)
        elif arr.ndim == 2:
            out = arr[..., None] if arr.shape[-1] != 1 else arr
        else:
            out = arr
        np.save(os.path.join(args.out_dir, f"{rid}.npy"), out)

    if os.path.isdir(args.signals):
        for fname in os.listdir(args.signals):
            if fname.endswith('.npy'):
                arr = np.load(os.path.join(args.signals, fname)).astype(np.float32)
                # mean-variance norm
                arr = (arr - arr.mean()) / (arr.std() + 1e-6)
                rid = os.path.splitext(fname)[0]
                process_array(arr, rid)
    else:
        arr = np.load(args.signals).astype(np.float32)
        arr = (arr - arr.mean()) / (arr.std() + 1e-6)
        process_array(arr, 'read')

if __name__ == '__main__':
    main()
