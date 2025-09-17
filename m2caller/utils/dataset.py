import os, numpy as np, torch
from torch.utils.data import Dataset

class NumpyRecordsDataset(Dataset):
    def __init__(self, records_dir, half=False):
        self.records_dir = records_dir
        self.files = [f for f in os.listdir(records_dir) if f.endswith('.npy')]
        self.half = half

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        arr = np.load(os.path.join(self.records_dir, fname))
        # support (N, W) or (N, W, 1)
        if arr.ndim == 2:
            arr = arr[..., None]
        # returns a list of windows (variable number per read)
        # to keep it simple, we return the full array; collate will batch by slicing into chunks of 32 windows
        return os.path.splitext(fname)[0], arr.astype(np.float16 if self.half else np.float32)

def collate_variable_windows(batch, pack=32):
    # batch is list of tuples: (read_id, (N, W, 1))
    items = []
    read_ids = []
    for rid, arr in batch:
        N = arr.shape[0]
        for i in range(0, N, pack):
            chunk = arr[i:i+pack]
            if chunk.size == 0:
                continue
            items.append(torch.from_numpy(chunk))
            read_ids.append(rid)
    # pad by 0 if last chunk smaller? model handles variable batch size; keep as is
    # shape: [B, n_seg, W, 1] -> flatten to [B, W, 1] by stacking along batch
    out = torch.cat(items, dim=0)  # [M, W, 1]
    lengths = torch.tensor([out.shape[1]] * out.shape[0], dtype=torch.int32)
    return read_ids, out, lengths
