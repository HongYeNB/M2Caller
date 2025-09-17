import argparse, os, torch
from torch.utils.data import DataLoader
from m2caller.utils.dataset import NumpyRecordsDataset, collate_variable_windows
from m2caller.utils.common import seed_everything, get_device
from m2caller.models.m2caller import M2CallerModel
from m2caller.models.m2caller_mamba import M2CallerMambaModel
from m2caller.utils.constants import VOCAB_SIZE, BLANK
from tqdm import tqdm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_records', required=True)
    ap.add_argument('--val_records', required=True)
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--d_model', type=int, default=512)
    ap.add_argument('--arch', type=str, default='mamba2', choices=['mamba2','ssm-lite'])
    ap.add_argument('--ssm_d_state', type=int, default=16)
    ap.add_argument('--ssm_d_conv', type=int, default=4)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--heads', type=int, default=4)
    ap.add_argument('--ssm_dim', type=int, default=256)
    ap.add_argument('--segment_len', type=int, default=4096)
    ap.add_argument('--save_dir', default='checkpoints')
    ap.add_argument('--no_cuda', action='store_true')
    ap.add_argument('--seed', type=int, default=1337)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = get_device(args.no_cuda)
    os.makedirs(args.save_dir, exist_ok=True)

    train_ds = NumpyRecordsDataset(args.train_records)
    val_ds = NumpyRecordsDataset(args.val_records)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_variable_windows)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_variable_windows)

    if args.arch == 'mamba2':
        model = M2CallerMambaModel(d_model=args.d_model, layers=args.layers,
                                   ssm_d_state=args.ssm_d_state, ssm_d_conv=args.ssm_d_conv,
                                   vocab_size=5).to(device)
    else:
        model = M2CallerModel(d_model=args.d_model, layers=args.layers,
                              heads=args.heads, ssm_dim=args.ssm_dim, vocab_size=5).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ctc = torch.nn.CTCLoss(blank=BLANK, reduction='mean', zero_infinity=True)

    best = 0.0
    for epoch in range(1, args.epochs+1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Train {epoch}")
        for read_ids, batch, lengths in pbar:
            batch = batch.to(device)  # [M, W, 1]
            # --- Placeholder targets: replace with real labeled transcripts in practice ---
            B = batch.shape[0]
            T = 200
            targets = torch.randint(1,5,(B*T//10,), dtype=torch.long, device=device)
            target_lengths = torch.full((B,), len(targets)//B, dtype=torch.int32, device=device)
            # ---------------------------------------------------------------------------

            log_probs, out_lengths = model(batch, lengths.to(device))
            loss = ctc(log_probs, targets, out_lengths, target_lengths)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            pbar.set_postfix(loss=float(loss))

        model.eval()
        total_len = 0; n = 0
        with torch.no_grad():
            for read_ids, batch, lengths in val_loader:
                batch = batch.to(device)
                log_probs, out_lengths = model(batch, lengths.to(device))
                total_len += out_lengths.float().mean().item(); n += 1
        score = total_len/max(1,n)
        if score > best:
            best = score
            ckpt = {'model': model.state_dict(), 'args': vars(args)}
            torch.save(ckpt, os.path.join(args.save_dir, 'best.pt'))
        print(f"[Val] avg output length {score:.2f} (best {best:.2f})")

if __name__ == '__main__':
    main()
