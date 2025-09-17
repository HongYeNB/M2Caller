import torch, torch.nn as nn
from m2caller.utils.constants import VOCAB_SIZE

_MAMBA2_IMPORTED = False
_MAMBA_ERR = None
try:
    from mamba_ssm.modules.mamba2 import Mamba2  # common path (v2)
    _MAMBA2_IMPORTED = True
except Exception as e1:
    try:
        from mamba_ssm import Mamba2  # fallback older path
        _MAMBA2_IMPORTED = True
    except Exception as e2:
        _MAMBA_ERR = (e1, e2)

class SignalSampler(nn.Module):
    def __init__(self, in_ch=1, d_model=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, d_model//2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(d_model//2),
            nn.ReLU(inplace=True),
            nn.Conv1d(d_model//2, d_model, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):  # x: [B, 1, L]
        return self.net(x)  # [B, C, L']

class M2CallerMambaModel(nn.Module):
    """
    CNN SignalSampler -> stacked official Mamba2 blocks -> CTC head.

    Requires `mamba-ssm` package:
        pip install mamba-ssm           # or: mamba-ssm-cu121 / cu124
    Repo: https://github.com/state-spaces/mamba
    """
    def __init__(self, d_model=512, layers=4, ssm_d_state=16, ssm_d_conv=4, vocab_size=5):
        super().__init__()
        if not _MAMBA2_IMPORTED:
            hints = "\n".join([f"- {type(e).__name__}: {e}" for e in (_MAMBA_ERR or []) if e])
            raise ImportError(
                "Failed to import Mamba2 from `mamba_ssm`.\n"
                "Install, e.g.:\n  pip install mamba-ssm\n"
                "or CUDA wheels: mamba-ssm-cu121 / mamba-ssm-cu124\n"
                f"Import errors:\n{hints}"
            )
        self.sampler = SignalSampler(1, d_model)
        self.pos = nn.Parameter(torch.randn(1, 2048, d_model) * 0.01)
        blocks = []
        for _ in range(layers):
            blocks.append(Mamba2(d_model=d_model, d_state=ssm_d_state, d_conv=ssm_d_conv))
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, signal, signal_lengths):
        if signal.dim() == 3 and signal.shape[1] != 1:
            signal = signal.transpose(1,2)  # [B, 1, W]
        feat = self.sampler(signal)          # [B, C, L']
        x = feat.transpose(1,2)              # [B, L', C]
        L = x.shape[1]
        if self.pos.shape[1] >= L:
            x = x + self.pos[:, :L, :]
        for blk in self.blocks:
            x = blk(x)                       # Mamba2 expects [B, L, C]
        x = self.norm(x)
        logits = self.out(x)                 # [B, L, V]
        log_probs = logits.log_softmax(-1).transpose(0,1).contiguous()  # [T, B, V]
        B = logits.shape[0]
        lengths = torch.full((B,), log_probs.shape[0], dtype=torch.int32, device=log_probs.device)
        return log_probs, lengths
