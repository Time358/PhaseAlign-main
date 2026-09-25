import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import PatchEmbedding, TemporalEmbedding, TimeFeatureEmbedding
from layers.StandardNorm import Normalize


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        return self.dropout(self.linear(self.flatten(x)))


class MultiScaleConvBranch(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.dw3 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.dw5 = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2, groups=d_model)
        self.pw = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        y = x.transpose(1, 2)
        y = self.pw(self.act(self.dw3(y) + self.dw5(y)))
        y = self.drop(y.transpose(1, 2))
        return self.norm(x + y)


class FrequencyCrossAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.q_patch = nn.Linear(d_model, d_model)
        self.k_freq = nn.Linear(d_model, d_model)
        self.v_freq = nn.Linear(d_model, d_model)
        self.q_freq = nn.Linear(d_model, d_model)
        self.k_patch = nn.Linear(d_model, d_model)
        self.v_patch = nn.Linear(d_model, d_model)
        self.o_patch = nn.Linear(d_model, d_model)
        self.o_freq = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.norm_patch = nn.LayerNorm(d_model)
        self.norm_freq = nn.LayerNorm(d_model)

    def _split_heads(self, x):
        b, l, _ = x.shape
        return x.view(b, l, self.n_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x):
        b, _, l, _ = x.shape
        return x.transpose(1, 2).contiguous().view(b, l, self.n_heads * self.head_dim)

    def _cross_attend(self, q, k, v):
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale, dim=-1)
        attn = self.drop(attn)
        return torch.matmul(attn, v)

    def forward(self, patches, freq_tokens):
        patch_q = self._split_heads(self.q_patch(patches))
        freq_k = self._split_heads(self.k_freq(freq_tokens))
        freq_v = self._split_heads(self.v_freq(freq_tokens))
        patch_out = self._merge_heads(self._cross_attend(patch_q, freq_k, freq_v))
        patches = self.norm_patch(patches + self.drop(self.o_patch(patch_out)))

        freq_q = self._split_heads(self.q_freq(freq_tokens))
        patch_k = self._split_heads(self.k_patch(patches))
        patch_v = self._split_heads(self.v_patch(patches))
        freq_out = self._merge_heads(self._cross_attend(freq_q, patch_k, patch_v))
        freq_tokens = self.norm_freq(freq_tokens + self.drop(self.o_freq(freq_out)))
        return patches, freq_tokens


class PhaseAdapterBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, use_rel_phase=True):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d = d_model // n_heads
        # Ablation gate: False disables relative-phase RoPE + phase bias,
        # parameters are kept so all variants share identical parameter counts.
        self.use_rel_phase = bool(use_rel_phase)
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.gate = nn.Parameter(torch.zeros(n_heads))
        self.rope_gate = nn.Parameter(torch.zeros(n_heads))
        self.rope_freq = nn.Parameter(torch.linspace(1.0, 0.25, steps=n_heads))
        self.drop = nn.Dropout(dropout)
        self.conv_branch = MultiScaleConvBranch(d_model, dropout=dropout)
        self.mix_gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Linear(d_model, d_model))

    def _phase_rope(self, q, k):
        if self.d < 2:
            return q, k

        rotary_dim = (self.d // 2) * 2
        pos = torch.arange(q.size(-2), device=q.device, dtype=q.dtype)
        freq = self.rope_freq.to(dtype=q.dtype).view(1, self.n_heads, 1, 1).abs().clamp_min(1e-4)
        gate = torch.tanh(self.rope_gate).to(dtype=q.dtype).view(1, self.n_heads, 1, 1)
        angle = 2 * torch.pi * pos.view(1, 1, -1, 1) * freq / max(q.size(-2), 1)
        angle = angle * gate
        cos = torch.cos(angle)
        sin = torch.sin(angle)

        def rotate(x):
            x_rot = x[..., :rotary_dim]
            x_pass = x[..., rotary_dim:]
            x_pair = x_rot.view(*x_rot.shape[:-1], rotary_dim // 2, 2)
            x_even = x_pair[..., 0]
            x_odd = x_pair[..., 1]
            y = torch.stack([x_even * cos - x_odd * sin, x_even * sin + x_odd * cos], dim=-1)
            y = y.flatten(-2)
            if x_pass.size(-1) > 0:
                y = torch.cat([y, x_pass], dim=-1)
            return y

        return rotate(q), rotate(k)

    def forward(self, x, phase_bias=None, causal_mask=None, rel_on=True):
        b, l, d = x.shape
        q = self.q(x).view(b, l, self.n_heads, self.d).transpose(1, 2)
        k = self.k(x).view(b, l, self.n_heads, self.d).transpose(1, 2)
        v = self.v(x).view(b, l, self.n_heads, self.d).transpose(1, 2)
        do_rel = bool(self.use_rel_phase and rel_on)
        if do_rel:
            q, k = self._phase_rope(q, k)
        s = (q @ k.transpose(-2, -1)) / math.sqrt(self.d)
        if do_rel and phase_bias is not None:
            s = s + phase_bias * torch.tanh(self.gate).view(1, -1, 1, 1)
        if causal_mask is not None:
            s = s + causal_mask
        a = self.drop(torch.softmax(s, dim=-1))
        attn_out = self.o((a @ v).transpose(1, 2).contiguous().view(b, l, d))
        x = self.norm1(x + attn_out)
        conv_out = self.conv_branch(x)
        mix = torch.sigmoid(self.mix_gate(torch.cat([x, conv_out], dim=-1)))
        x = x + mix * (conv_out - x)
        return self.norm2(x + self.ff(x))


class FuturePhaseDecoder(nn.Module):
    def __init__(self, d_model, pred_len, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.pred_len = pred_len
        self.n_heads = n_heads
        self.d = d_model // n_heads
        self.query = nn.Parameter(torch.randn(pred_len, d_model) * 0.02)
        self.horizon_emb = nn.Embedding(pred_len, d_model)
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, d_model))
        self.drop = nn.Dropout(dropout)
        self.short_gate = nn.Parameter(torch.tensor(0.33))
        self.mid_gate = nn.Parameter(torch.tensor(0.33))
        self.long_gate = nn.Parameter(torch.tensor(0.33))

    def forward(self, memory, phase_feats=None):
        b, l, d = memory.shape
        hid = torch.arange(self.pred_len, device=memory.device)
        base_q = self.query.unsqueeze(0).expand(b, -1, -1) + self.horizon_emb(hid).unsqueeze(0)
        if phase_feats is not None:
            base_q = base_q + phase_feats

        q = self.q(base_q).view(b, self.pred_len, self.n_heads, self.d).transpose(1, 2)
        k = self.k(memory).view(b, l, self.n_heads, self.d).transpose(1, 2)
        v = self.v(memory).view(b, l, self.n_heads, self.d).transpose(1, 2)

        attn = torch.softmax((q @ k.transpose(-2, -1)) / math.sqrt(self.d), dim=-1)

        short_len = max(1, l // 4)
        mid_len = max(1, l // 2)
        short_ctx = attn[..., -short_len:] @ v[..., -short_len:, :]
        mid_ctx = attn[..., -mid_len:] @ v[..., -mid_len:, :]
        long_ctx = attn @ v

        g = torch.softmax(torch.stack([self.short_gate, self.mid_gate, self.long_gate]), dim=0)
        out = g[0] * short_ctx + g[1] * mid_ctx + g[2] * long_ctx
        out = out.transpose(1, 2).contiguous().view(b, self.pred_len, d)
        return self.o(out)


class CycleEvolution(nn.Module):
    """Phase-specific cycle-evolution residual branch.

    Reshapes the history into (n_cycles, period). For each phase r in [0, period):
        recent_r = last cycle value at phase r
        mean_r   = mean over available cycles at phase r
        slope_r  = least-squares slope across cycle index at phase r
        z_r      = MLP([recent_r, mean_r, slope_r])      # static per-phase correction
    Then for future horizon h (0-indexed):
        phase  = h % period
        day_off = h // period                             # 0-indexed future day
        residual[h] = z[phase] + tanh(drift_scale) * slope[phase] * day_off
    The drift term extrapolates the estimated phase-specific trend across future
    days, which is exactly what a naive static seasonal copy misses and what made
    large-alpha MSE degrade in the naive-residual probe.

    x: [N, L] scalar series per sample (N = B or B*C). Returns [N, pred_len].
    """
    def __init__(self, period, pred_len, d_hidden=16):
        super().__init__()
        self.period = max(1, int(period))
        self.pred_len = pred_len
        self.mlp = nn.Sequential(
            nn.Linear(3, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, 1),
        )
        self.drift_scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        device, dtype = x.device, x.dtype
        n, L = x.shape
        n_cyc = L // self.period
        if n_cyc < 1:
            return torch.zeros(n, self.pred_len, device=device, dtype=dtype)

        tail = x[:, -n_cyc * self.period:]                     # [N, n_cyc*period]
        vals = tail.view(n, n_cyc, self.period)                # [N, n_cyc, period]

        recent = vals[:, -1, :]                                # [N, period]
        mean_ = vals.mean(dim=1)                               # [N, period]

        cyc_idx = torch.arange(n_cyc, device=device, dtype=dtype)
        centered = cyc_idx - cyc_idx.mean()
        denom = (centered ** 2).sum() + 1e-6
        slope = (vals * centered.view(1, -1, 1)).sum(dim=1) / denom   # [N, period]

        feats = torch.stack([recent, mean_, slope], dim=-1)    # [N, period, 3]
        z = self.mlp(feats).squeeze(-1)                        # [N, period]

        phase_idx = torch.arange(self.pred_len, device=device) % self.period
        day_off = torch.arange(self.pred_len, device=device, dtype=dtype)
        base = z[:, phase_idx]                                 # [N, pred_len]
        drift = slope[:, phase_idx]                            # [N, pred_len]
        out = base + self.drift_scale * drift * day_off.unsqueeze(0)
        return out


class Model(nn.Module):
    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.n_heads = configs.n_heads
        self.top_k = int(getattr(configs, 'top_k', 9))
        self.fourier_fmax = float(getattr(configs, 'fourier_fmax', 100.0))
        self.use_fixed_basis = bool(int(getattr(configs, 'use_fixed_basis', 1)))
        self.patch_stride = configs.stride
        self.patch_len = configs.patch_len

        self.norm = Normalize(configs.enc_in, affine=False)
        self.patch = PatchEmbedding(configs.d_model, configs.patch_len, configs.stride, configs.dropout)
        self.freq_stat_dim = self.top_k * 4 + 3
        self.freq_proj = nn.Sequential(
            nn.Linear(self.freq_stat_dim, configs.d_model),
            nn.GELU(),
            nn.Linear(configs.d_model, configs.d_model),
        )
        self.freq_tokens = nn.Parameter(torch.randn(self.top_k, configs.d_model) * 0.02)
        self.freq_peak_proj = nn.Sequential(
            nn.Linear(4, configs.d_model),
            nn.GELU(),
            nn.Linear(configs.d_model, configs.d_model),
        )
        self.freq_peak_gate = nn.Parameter(torch.tensor(0.1))
        self.freq_token_proj = nn.Linear(configs.d_model, configs.d_model)
        self.freq_cross_attn = FrequencyCrossAttention(configs.d_model, configs.n_heads, dropout=configs.dropout)
        # ---- Ablation switches (gating only, params always kept) ----
        # use_rel_phase: encoder relative phase (RoPE + phase bias)
        # use_fut_phase: decoder future absolute-phase extrapolation (adaptive part only;
        #                generic fixed sinusoidal position is always kept)
        # use_tf_interaction: time-frequency cross-attention fusion
        # shared_phase: stage-specificity control.
        #   0 = stage-specific (default Full: Rel in encoder + window-referenced
        #       trajectory in decoder);
        #   1 = Shared-Rel: both stages use the horizon-relative harmonic operator
        #       (decoder drops phase0 anchor and hist_len offset, i.e. phase = 2π f h);
        #   2 = Shared-Abs: both stages use the window-referenced absolute operator
        #       (encoder drops relative bias/RoPE and instead adds absolute
        #       trajectory feats built with the SAME future_phase_proj weights).
        #   All params are always instantiated; modes only gate data paths, so every
        #   variant reports identical parameter counts.
        self.use_rel_phase = bool(getattr(configs, 'use_rel_phase', 1))
        self.use_fut_phase = bool(getattr(configs, 'use_fut_phase', 1))
        self.use_tf_interaction = bool(getattr(configs, 'use_tf_interaction', 1))
        self.shared_phase = int(getattr(configs, 'shared_phase', 0))
        self.phase_proj = nn.Linear(configs.d_model, configs.d_model)
        self.diff_patch = PatchEmbedding(configs.d_model, configs.patch_len, configs.stride, configs.dropout)
        self.diff_proj = nn.Linear(configs.d_model, configs.d_model)
        self.diff_phase_proj = nn.Linear(configs.d_model, configs.d_model)
        self.diff_fuse = nn.Sequential(
            nn.Linear(configs.d_model * 2, configs.d_model),
            nn.GELU(),
            nn.Linear(configs.d_model, configs.d_model),
        )
        self.phase_blocks = nn.ModuleList([
            PhaseAdapterBlock(configs.d_model, configs.n_heads, configs.d_ff, configs.dropout,
                              use_rel_phase=self.use_rel_phase)
            for _ in range(configs.e_layers)
        ])
        self.future_phase_proj = nn.Sequential(
            nn.Linear(self.top_k * 2, configs.d_model),
            nn.GELU(),
            nn.Linear(configs.d_model, configs.d_model),
        )
        self.future_phase_gate = nn.Parameter(torch.tensor(0.1))
        self.future_encoder = nn.Sequential(
            nn.Linear(1, configs.d_model),
            nn.GELU(),
            nn.Linear(configs.d_model, configs.d_model),
        )
        self.future_proj = nn.Sequential(
            nn.Linear(configs.d_model * 2, configs.d_model),
            nn.GELU(),
            nn.Linear(configs.d_model, configs.d_model),
        )
        self.future_align_proj = nn.Linear(configs.d_model, configs.d_model)
        self.future_align_gate = nn.Parameter(torch.tensor(0.1))

        self.use_time_mark = bool(getattr(configs, 'use_time_mark', True))
        if self.use_time_mark:
            embed_type = getattr(configs, 'embed', 'timeF')
            if embed_type == 'timeF':
                self.time_embedding = TimeFeatureEmbedding(configs.d_model, embed_type='timeF', freq=configs.freq)
                self.future_time_embedding = TimeFeatureEmbedding(configs.d_model, embed_type='timeF', freq=configs.freq)
            else:
                self.time_embedding = TemporalEmbedding(configs.d_model, embed_type=embed_type, freq=configs.freq)
                self.future_time_embedding = TemporalEmbedding(configs.d_model, embed_type=embed_type, freq=configs.freq)
            self.time_enc_gate = nn.Parameter(torch.tensor(0.0))
            self.future_time_gate = nn.Parameter(torch.tensor(0.0))

        self.decoder = FuturePhaseDecoder(configs.d_model, configs.pred_len, configs.n_heads, configs.dropout)
        self.head = FlattenHead(configs.enc_in, configs.d_model, configs.pred_len, head_dropout=configs.dropout)
        self.res_head = FlattenHead(configs.enc_in, configs.d_model, configs.pred_len, head_dropout=configs.dropout)
        self.res_out_scale = nn.Parameter(torch.full((configs.pred_len,), 0.1))

        self.use_phase_residual = bool(getattr(configs, 'use_phase_residual', False))
        if self.use_phase_residual:
            self.seq_len = getattr(configs, 'seq_len', None)
            self.cycle_evo = CycleEvolution(
                period=getattr(configs, 'phase_residual_period', 24),
                pred_len=configs.pred_len,
            )
            self.cycle_gate = nn.Parameter(torch.tensor(0.05))
        else:
            self.cycle_evo = None

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            return self.forecast(x_enc, x_mark_enc, x_mark_dec)[:, -self.pred_len:, :]
        return None

    def forecast(self, x_enc, x_mark_enc=None, x_mark_dec=None):
        x_raw = x_enc                                           # pre-norm, output space
        x_enc = self.norm(x_enc, 'norm')
        b, l, c = x_enc.shape
        x_flat = x_enc.permute(0, 2, 1).contiguous().reshape(b * c, l, 1)
        freq_stats, freq_tokens, freq_attn, freq_token_seq = self._freq_stats(x_flat)
        freq_tok = self.freq_proj(freq_stats).unsqueeze(1)
        patches, _ = self.patch(x_flat.permute(0, 2, 1).contiguous().float())
        freq_token_bias = self.freq_token_proj(freq_tokens).unsqueeze(1)
        x = self.phase_proj(patches) + freq_tok + freq_token_bias
        freq_context = torch.sum(freq_attn.unsqueeze(-1) * freq_token_seq, dim=1, keepdim=True)
        shared = int(getattr(self, 'shared_phase', 0))
        if shared == 1:
            eff_rel, eff_fut = True, True
        elif shared == 2:
            eff_rel, eff_fut = False, True
        else:
            eff_rel, eff_fut = self.use_rel_phase, self.use_fut_phase
        if self.use_tf_interaction:
            x, freq_tokens = self.freq_cross_attn(x + freq_context, freq_token_seq)
            x = x + self.freq_token_proj(freq_tokens.mean(dim=1, keepdim=True))
        # else: skip TF cross-attention fusion entirely (params retained for
        # identical parameter counts across ablation variants)

        if self.use_time_mark and x_mark_enc is not None:
            time_enc = self.time_embedding(x_mark_enc.float())
            time_enc = self._time_patch_pool(time_enc)
            n_patch = time_enc.size(1)
            time_enc = time_enc.unsqueeze(1).expand(-1, c, -1, -1).reshape(b * c, n_patch, self.d_model)
            x = x + torch.tanh(self.time_enc_gate) * time_enc

        diff_x = x_enc.diff(dim=1, prepend=x_enc[:, :1, :])
        diff_flat = diff_x.permute(0, 2, 1).contiguous().reshape(b * c, l, 1)
        diff_patches, _ = self.diff_patch(diff_flat.permute(0, 2, 1).contiguous().float())
        diff_x = self.diff_proj(diff_patches)
        diff_x = self.diff_phase_proj(diff_x)
        x = self.diff_fuse(torch.cat([x, diff_x], dim=-1))
        freq_values = freq_stats[:, 1 + self.top_k : 1 + 2 * self.top_k]
        # Stage-specificity control:
        #   Shared-Abs (shared==2): encoder skips the relative-phase bias/RoPE and
        #   instead receives window-referenced absolute trajectory features built
        #   with the SAME future_phase_proj weights as the decoder (translation
        #   invariance deliberately broken in history).
        if shared == 2:
            abs_hist_feats = self._absolute_hist_feats(freq_stats, hist_len=l)
            # abs_hist_feats is [B*C, L, d]; pool time axis L -> n_patch so it
            # can be added to patch tokens (differentiable, no new params).
            if abs_hist_feats.size(1) != x.size(1):
                abs_hist_feats = F.interpolate(
                    abs_hist_feats.transpose(1, 2), size=x.size(1),
                    mode='linear', align_corners=False).transpose(1, 2)
            x = x + abs_hist_feats
            phase_bias = None
        else:
            phase_bias = self._phase_bias(x, freq_values=freq_values) if eff_rel else None
        for blk in self.phase_blocks:
            x = blk(x, phase_bias=phase_bias, causal_mask=None, rel_on=eff_rel)

        memory = x
        fixed_phase_feats = self._fixed_future_feats(memory.size(0), memory.device)
        if eff_fut:
            if shared == 1:
                # Shared-Rel: decoder extrapolates WITHOUT the phase0 anchor and
                # WITHOUT the hist_len offset (phase = 2π f h), i.e. the same
                # horizon-relative harmonic operator used in the encoder.
                adaptive_phase_feats = self._adaptive_future_feats(
                    freq_stats, memory.device, hist_len=0, drop_phase0=True)
            else:
                adaptive_phase_feats = self._adaptive_future_feats(freq_stats, memory.device, hist_len=l)
        else:
            adaptive_phase_feats = torch.zeros_like(fixed_phase_feats)
        phase_feats = fixed_phase_feats + adaptive_phase_feats
        if self.use_time_mark and x_mark_dec is not None:
            future_mark = x_mark_dec[:, -self.pred_len:, :].float()
            future_time = self.future_time_embedding(future_mark)
            future_time = future_time.unsqueeze(1).expand(-1, c, -1, -1).reshape(b * c, self.pred_len, self.d_model)
            phase_feats = phase_feats + torch.tanh(self.future_time_gate) * future_time
        dec = self.decoder(memory, phase_feats)
        dec = dec.reshape(b, c, self.pred_len, self.d_model).permute(0, 2, 1, 3).contiguous()
        base_out = self.head(dec)
        res_out = self.res_head(dec)
        out = self.norm(base_out + self.res_out_scale.view(1, -1, 1) * res_out, 'denorm')
        if self.use_phase_residual and self.cycle_evo is not None and c == 1:
            flat = x_raw.squeeze(-1)                                   # [B, L]
            resid = self.cycle_evo(flat)                               # [B, pred_len]
            out = out + self.cycle_gate * resid.unsqueeze(-1)
        return out

    def _time_patch_pool(self, time_enc):
        b, l, d = time_enc.shape
        x = time_enc.transpose(1, 2)
        x = F.pad(x, (0, self.patch_stride), mode='replicate')
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_stride)
        x = x.mean(dim=-1)
        return x.transpose(1, 2)

    def future_alignment_repr(self, history_x, future_x, future_mask=None):
        history_x = self.norm(history_x, 'norm')
        future_x = self.norm(future_x, 'norm')

        hist_repr = self.future_align_proj(self.future_encoder(history_x.mean(dim=-1, keepdim=True)))
        hist_repr = hist_repr.mean(dim=1)

        future_repr = self.future_align_proj(self.future_encoder(future_x.mean(dim=-1, keepdim=True)))
        if future_mask is not None:
            mask = future_mask.to(dtype=future_repr.dtype, device=future_repr.device)
            if mask.dim() == 3:
                mask = mask.mean(dim=-1)
            if mask.size(1) != future_repr.size(1):
                if mask.size(1) > future_repr.size(1):
                    mask = mask[:, : future_repr.size(1)]
                else:
                    pad_len = future_repr.size(1) - mask.size(1)
                    pad = mask[:, -1:].expand(mask.size(0), pad_len)
                    mask = torch.cat([mask, pad], dim=1)
            mask = mask.unsqueeze(-1)
            denom = mask.sum(dim=1).clamp_min(1.0)
            future_repr = (future_repr * mask).sum(dim=1) / denom
        else:
            future_repr = future_repr.mean(dim=1)

        return hist_repr, future_repr

    def _freq_stats(self, x):
        x = x.squeeze(-1) - x.squeeze(-1).mean(dim=1, keepdim=True)
        spec = torch.fft.rfft(x, dim=1)
        mag = torch.abs(spec)
        phase = torch.angle(spec)
        mag[:, 0] = 0
        k = min(self.top_k, mag.size(-1))
        top = torch.topk(mag, k=k, dim=-1)
        amp = top.values
        frq = top.indices.float() / max(x.size(1), 1)
        top_phase = torch.gather(phase, dim=-1, index=top.indices)
        phase_sin = torch.sin(top_phase)
        phase_cos = torch.cos(top_phase)
        amp_sum = amp.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        amp_norm = amp / amp_sum
        frq_mean = frq.mean(dim=-1, keepdim=True)
        frq_std = frq.std(dim=-1, keepdim=True, unbiased=False)
        stats = torch.cat([amp_sum, amp_norm, frq, phase_sin, phase_cos, frq_mean, frq_std], dim=-1)
        if stats.size(-1) < self.freq_stat_dim:
            pad = torch.zeros(stats.size(0), self.freq_stat_dim - stats.size(-1), device=stats.device, dtype=stats.dtype)
            stats = torch.cat([stats, pad], dim=-1)
        stats = stats[:, : self.freq_stat_dim]
        token_scores = amp_norm

        freq_basis = self.freq_tokens.unsqueeze(0).expand(x.size(0), -1, -1)
        freq_meta = torch.stack([
            amp_norm,
            frq,
            phase_sin,
            phase_cos,
        ], dim=-1)
        freq_token_seq = freq_basis + torch.tanh(self.freq_peak_gate) * self.freq_peak_proj(freq_meta)
        freq_tokens = torch.sum(token_scores.unsqueeze(-1) * freq_token_seq, dim=1)
        return stats, freq_tokens, token_scores, freq_token_seq

    def _phase_bias(self, x, freq_values=None):
        bsz, n_patch = x.size(0), x.size(1)
        device = x.device
        dtype = x.dtype

        pos = torch.arange(n_patch, device=device, dtype=dtype)
        rel = pos[:, None] - pos[None, :]
        rel_time = rel * float(self.patch_stride)

        fixed_freq = torch.linspace(
            1.0 / max(n_patch * self.patch_stride, 1),
            0.5,
            steps=self.n_heads,
            device=device,
            dtype=dtype,
        )

        if freq_values is None:
            head_freq = fixed_freq.unsqueeze(0).expand(bsz, -1)
        else:
            sample_freq = freq_values.to(device=device, dtype=dtype)
            if sample_freq.dim() == 1:
                sample_freq = sample_freq.unsqueeze(0)
            if sample_freq.size(0) != bsz:
                sample_freq = sample_freq[:1].expand(bsz, -1)

            if sample_freq.size(1) >= self.n_heads:
                head_freq = sample_freq[:, : self.n_heads]
            else:
                pad = fixed_freq[: self.n_heads - sample_freq.size(1)].unsqueeze(0).expand(bsz, -1)
                head_freq = torch.cat([sample_freq, pad], dim=1)

            head_freq = torch.where(head_freq > 0, head_freq, fixed_freq.unsqueeze(0))

        phase = 2 * torch.pi * head_freq[:, :, None, None] * rel_time[None, None, :, :]
        return torch.cos(phase)

    def _fixed_future_feats(self, bsz, device):
        if not self.use_fixed_basis:
            return torch.zeros(bsz, self.pred_len, self.d_model,
                               device=device, dtype=torch.float32)
        half_dim = max(1, self.d_model // 2)
        t = torch.arange(self.pred_len, device=device).float()
        t = (t / max(self.pred_len - 1, 1)).unsqueeze(1)

        freqs = torch.exp(torch.linspace(
            math.log(1.0),
            math.log(max(float(self.fourier_fmax), 1.0)),
            half_dim,
            device=device
        )).unsqueeze(0)

        phase = 2 * torch.pi * t * freqs
        feats = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)

        if feats.size(-1) < self.d_model:
            pad = torch.zeros(self.pred_len, self.d_model - feats.size(-1), device=device, dtype=feats.dtype)
            feats = torch.cat([feats, pad], dim=-1)
        elif feats.size(-1) > self.d_model:
            feats = feats[:, :self.d_model]

        return feats.unsqueeze(0).expand(bsz, -1, -1)

    def _adaptive_future_feats(self, freq_stats, device, hist_len, drop_phase0=False):
        bsz = freq_stats.size(0)
        dtype = freq_stats.dtype
        k = self.top_k

        amp_norm = freq_stats[:, 1 : 1 + k]
        frq = freq_stats[:, 1 + k : 1 + 2 * k]
        phase_sin = freq_stats[:, 1 + 2 * k : 1 + 3 * k]
        phase_cos = freq_stats[:, 1 + 3 * k : 1 + 4 * k]

        if drop_phase0:
            # Shared-Rel control: remove the window-referenced phase0 anchor so
            # the decoder uses the same horizon-relative harmonic operator as the
            # encoder (phase = 2π f h, translation-invariant by construction).
            phase0 = torch.zeros_like(phase_sin)
        else:
            phase0 = torch.atan2(phase_sin, phase_cos)
            phase0 = torch.nan_to_num(phase0)

        future_t = hist_len + torch.arange(self.pred_len, device=device, dtype=dtype)
        future_t = future_t.view(1, -1, 1)

        phase_h = phase0.unsqueeze(1) + 2 * torch.pi * frq.unsqueeze(1) * future_t

        feats = torch.cat([
            amp_norm.unsqueeze(1) * torch.sin(phase_h),
            amp_norm.unsqueeze(1) * torch.cos(phase_h),
        ], dim=-1)

        return torch.tanh(self.future_phase_gate) * self.future_phase_proj(feats)

    def _absolute_hist_feats(self, freq_stats, hist_len):
        """Window-referenced absolute trajectory features for history positions.

        Uses the SAME future_phase_proj weights as the decoder so the Shared-Abs
        control compares operators, not parameter budgets: history time steps
        t = 0..hist_len-1 get phase0 + 2π f t (absolute, window-referenced),
        exactly like the decoder's future trajectory, instead of the
        translation-invariant relative bias used by the stage-specific encoder.
        Returns [B*C, hist_len, d_model].
        """
        dtype = freq_stats.dtype
        device = freq_stats.device
        k = self.top_k

        amp_norm = freq_stats[:, 1 : 1 + k]
        frq = freq_stats[:, 1 + k : 1 + 2 * k]
        phase_sin = freq_stats[:, 1 + 2 * k : 1 + 3 * k]
        phase_cos = freq_stats[:, 1 + 3 * k : 1 + 4 * k]

        phase0 = torch.atan2(phase_sin, phase_cos)
        phase0 = torch.nan_to_num(phase0)

        hist_t = torch.arange(max(1, int(hist_len)), device=device, dtype=dtype).view(1, -1, 1)
        phase_h = phase0.unsqueeze(1) + 2 * torch.pi * frq.unsqueeze(1) * hist_t
        feats = torch.cat([
            amp_norm.unsqueeze(1) * torch.sin(phase_h),
            amp_norm.unsqueeze(1) * torch.cos(phase_h),
        ], dim=-1)
        return torch.tanh(self.future_phase_gate) * self.future_phase_proj(feats)

