import argparse, os, torch
from torch.utils.data import DataLoader
from m2caller.utils.dataset import NumpyRecordsDataset, collate_variable_windows
from m2caller.utils.common import get_device
from m2caller.models.m2caller import M2CallerModel
from m2caller.models.m2caller_mamba import M2CallerMambaModel
from m2caller.decoding.greedy import greedy_decode
from m2caller.utils.constants import TOKENS, BLANK
from tqdm import tqdm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--records_dir', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out_fasta', required=True)
    ap.add_argument('--no_cuda', action='store_true')
    ap.add_argument('--half', action='store_true')
    ap.add_argument('--arch', type=str, default='mamba2', choices=['mamba2','ssm-lite'])
    args = ap.parse_args()

    device = get_device(args.no_cuda)
    ds = NumpyRecordsDataset(args.records_dir, half=args.half)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_variable_windows)

    ckpt = torch.load(args.ckpt, map_location=device)
    margs = ckpt.get('args', {})
    if args.arch == 'mamba2':
        model = M2CallerMambaModel(d_model=margs.get('d_model',512),
                                   layers=margs.get('layers',4),
                                   ssm_d_state=margs.get('ssm_d_state',16),
                                   ssm_d_conv=margs.get('ssm_d_conv',4),
                                   vocab_size=len(TOKENS))
    else:
        model = M2CallerModel(**{k:v for k,v in margs.items()
                                 if k in ['d_model','layers','heads','ssm_dim']},
                              vocab_size=len(TOKENS))
    model.load_state_dict(ckpt['model'])
    model.to(device).eval()
    if args.half:
        model.half()

    with torch.no_grad(), open(args.out_fasta, 'w') as fw:
        for read_ids, batch, lengths in tqdm(loader, desc='Infer'):
            batch = batch.to(device)
            log_probs, out_lengths = model(batch, lengths.to(device))
            seqs_idx = greedy_decode(log_probs, out_lengths, blank=BLANK)
            concat = {}
            for rid, seq in zip(read_ids, seqs_idx):
                s = ''.join(TOKENS[i] for i in seq if i != BLANK)
                s = s.replace('-', '')
                concat.setdefault(rid, []).append(s)
            for rid, parts in concat.items():
                fw.write(f">{rid}\n")
                fw.write(''.join(parts) + "\n")

if __name__ == '__main__':
    main()
