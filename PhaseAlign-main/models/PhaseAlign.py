import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import PatchEmbedding, TimeFeatureEmbedding
from layers.StandardNorm import Normalize


def num_patches(seq_len, patch_len, stride):
    return 1 + (seq_len + stride - patch_len) // stride


class FlattenHead(nn.Module):
    def __init__(self, d_model, pred_len, head_dropout=0.0):
        super().__init__()
        self.linear = nn.Linear(d_model, pred_len)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        return self.dropout(self.linear(x))


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
        return torch.matmul(self.drop(attn), v)

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
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d = d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model))
        self.gate = nn.Parameter(torch.zeros(n_heads))
        self.rope_gate = nn.Parameter(torch.zeros(n_heads))
        self.rope_freq = nn.Parameter(torch.linspace(1.0, 0.25, steps=n_heads))
        self.drop = nn.Dropout(dropout)
        self.conv_branch = MultiScaleConvBranch(d_model, dropout=dropout)
        self.mix_gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.GELU(),
                                      nn.Linear(d_model, d_model))

    def _phase_rope(self, q, k):
        if self.d < 2:
            return q, k
        rotary_dim = (self.d // 2) * 2
        pos = torch.arange(q.size(-2), device=q.device, dtype=q.dtype)
        freq = self.rope_freq.to(dtype=q.dtype).view(1, self.n_heads, 1, 1).abs().clamp_min(1e-4)
        gate = torch.tanh(self.rope_gate).to(dtype=q.dtype).view(1, self.n_heads, 1, 1)
        angle = 2 * torch.pi * pos.view(1, 1, -1, 1) * freq / max(q.size(-2), 1) * gate
        cos, sin = torch.cos(angle), torch.sin(angle)

        def rotate(x):
            x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
            pair = x_rot.view(*x_rot.shape[:-1], rotary_dim // 2, 2)
            even, odd = pair[..., 0], pair[..., 1]
            out = torch.stack([even * cos - odd * sin, even * sin + odd * cos], dim=-1).flatten(-2)
            return torch.cat([out, x_pass], dim=-1) if x_pass.size(-1) > 0 else out

        return rotate(q), rotate(k)

    def forward(self, x, phase_bias):
        b, l, d = x.shape
        q = self.q(x).view(b, l, self.n_heads, self.d).transpose(1, 2)
        k = self.k(x).view(b, l, self.n_heads, self.d).transpose(1, 2)
        v = self.v(x).view(b, l, self.n_heads, self.d).transpose(1, 2)
        q, k = self._phase_rope(q, k)
        s = (q @ k.transpose(-2, -1)) / math.sqrt(self.d)
        s = s + phase_bias * torch.tanh(self.gate).view(1, -1, 1, 1)
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
        self.o = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
                               nn.Linear(d_model, d_model))
        self.drop = nn.Dropout(dropout)
        self.short_gate = nn.Parameter(torch.tensor(0.33))
        self.mid_gate = nn.Parameter(torch.tensor(0.33))
        self.long_gate = nn.Parameter(torch.tensor(0.33))

    def forward(self, memory, phase_feats):
        b, l, d = memory.shape
        hid = torch.arange(self.pred_len, device=memory.device)
        base_q = self.query.unsqueeze(0).expand(b, -1, -1) + self.horizon_emb(hid).unsqueeze(0)
        base_q = base_q + phase_feats

        q = self.q(base_q).view(b, self.pred_len, self.n_heads, self.d).transpose(1, 2)
        k = self.k(memory).view(b, l, self.n_heads, self.d).transpose(1, 2)
        v = self.v(memory).view(b, l, self.n_heads, self.d).transpose(1, 2)

        attn = torch.softmax((q @ k.transpose(-2, -1)) / math.sqrt(self.d), dim=-1)
        attn = self.drop(attn)

        short_len = max(1, l // 4)
        mid_len = max(1, l // 2)
        short_ctx = attn[..., -short_len:] @ v[..., -short_len:, :]
        mid_ctx = attn[..., -mid_len:] @ v[..., -mid_len:, :]
        long_ctx = attn @ v

        g = torch.softmax(torch.stack([self.short_gate, self.mid_gate, self.long_gate]), dim=0)
        out = g[0] * short_ctx + g[1] * mid_ctx + g[2] * long_ctx
        out = out.transpose(1, 2).contiguous().view(b, self.pred_len, d)
        return self.o(out)


class PhaseAlign(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.n_heads = configs.n_heads
        self.top_k = int(configs.top_k)
        self.fourier_fmax = float(configs.fourier_fmax)
        self.patch_len = configs.patch_len
        self.patch_stride = configs.stride
        self.n_patch = num_patches(configs.seq_len, configs.patch_len, configs.stride)
        assert self.n_patch >= 1, 'seq_len too short for the requested patch length/stride'

        self.norm = Normalize()
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
        self.freq_cross_attn = FrequencyCrossAttention(configs.d_model, configs.n_heads,
                                                     dropout=configs.dropout)

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
            PhaseAdapterBlock(configs.d_model, configs.n_heads, configs.d_ff, configs.dropout)
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
        self.future_align_proj = nn.Linear(configs.d_model, configs.d_model)

        self.time_embedding = TimeFeatureEmbedding(configs.d_model, freq=configs.freq)
        self.future_time_embedding = TimeFeatureEmbedding(configs.d_model, freq=configs.freq)
        self.time_enc_gate = nn.Parameter(torch.tensor(0.0))
        self.future_time_gate = nn.Parameter(torch.tensor(0.0))

        self.decoder = FuturePhaseDecoder(configs.d_model, configs.pred_len, configs.n_heads, configs.dropout)
        self.head = FlattenHead(configs.d_model, configs.pred_len, head_dropout=configs.dropout)
        self.res_head = FlattenHead(configs.d_model, configs.pred_len, head_dropout=configs.dropout)
        self.res_out_scale = nn.Parameter(torch.full((configs.pred_len,), 0.1))

        self._align_cache = None

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        return self.forecast(x_enc, x_mark_enc, x_mark_dec)

    def forecast(self, x_enc, x_mark_enc=None, x_mark_dec=None):
        memory, ctx = self.encode(x_enc, x_mark_enc)
        self._align_cache = (memory, ctx)
        return self._decode(memory, ctx, x_mark_dec)

    def clear_align_cache(self):
        self._align_cache = None

    def encode(self, x_enc, x_mark_enc=None):
        stats = self.norm.norm_stats(x_enc)
        b, l, c = x_enc.shape
        x_norm = self.norm.apply_norm(x_enc, stats)
        x_flat = x_norm.permute(0, 2, 1).contiguous().reshape(b * c, l, 1)

        freq_stats, comp_tokens, spec_token = self._spectral_context(x_flat)
        patches, _ = self.patch(x_flat.permute(0, 2, 1).contiguous().float())
        x = self.phase_proj(patches) + self.freq_proj(freq_stats).unsqueeze(1) \
            + self.freq_token_proj(spec_token).unsqueeze(1)

        x, comp_tokens = self.freq_cross_attn(x, comp_tokens)

        if x_mark_enc is not None:
            time_enc = self._time_patch_pool(self.time_embedding(x_mark_enc.float()))
            n_patch = time_enc.size(1)
            time_enc = time_enc.unsqueeze(1).expand(-1, c, -1, -1).reshape(b * c, n_patch, self.d_model)
            x = x + torch.tanh(self.time_enc_gate) * time_enc

        diff_x = x_norm.diff(dim=1, prepend=x_norm[:, :1, :])
        diff_flat = diff_x.permute(0, 2, 1).contiguous().reshape(b * c, l, 1)
        diff_patches, _ = self.diff_patch(diff_flat.permute(0, 2, 1).contiguous().float())
        diff_x = self.diff_phase_proj(self.diff_proj(diff_patches))
        x = self.diff_fuse(torch.cat([x, diff_x], dim=-1))

        phase_bias = self._harmonic_bias(freq_stats)
        for blk in self.phase_blocks:
            x = blk(x, phase_bias)

        ctx = {'stats': stats, 'b': b, 'c': c, 'l': l, 'freq_stats': freq_stats}
        return x, ctx

    def _spectral_context(self, x):
        x = x.squeeze(-1) - x.squeeze(-1).mean(dim=1, keepdim=True)
        spec = torch.fft.rfft(x, dim=1)
        mag, phase = torch.abs(spec), torch.angle(spec)
        mag = mag.clone()
        mag[:, 0] = 0.0
        k = min(self.top_k, mag.size(-1))
        top = torch.topk(mag, k=k, dim=-1)
        amp = top.values
        frq = top.indices.float() / max(x.size(1), 1)
        top_phase = torch.gather(phase, dim=-1, index=top.indices)
        phase_sin, phase_cos = torch.sin(top_phase), torch.cos(top_phase)
        amp_sum = amp.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        amp_norm = amp / amp_sum
        frq_mean = frq.mean(dim=-1, keepdim=True)
        frq_std = frq.std(dim=-1, keepdim=True, unbiased=False)

        def pad_to_k(block):
            if block.size(-1) >= self.top_k:
                return block
            return torch.cat([block, block.new_zeros(block.size(0),
                                                      self.top_k - block.size(-1))], dim=-1)

        stats = torch.cat([amp_sum, pad_to_k(amp_norm), pad_to_k(frq), pad_to_k(phase_sin),
                           pad_to_k(phase_cos), frq_mean, frq_std], dim=-1)
        assert stats.size(-1) == self.freq_stat_dim

        comp_meta = torch.stack([amp_norm, frq, phase_sin, phase_cos], dim=-1)
        comp_tokens = self.freq_tokens[:k].unsqueeze(0).expand(x.size(0), -1, -1) \
            + torch.tanh(self.freq_peak_gate) * self.freq_peak_proj(comp_meta)
        spec_token = (amp_norm.unsqueeze(-1) * comp_tokens).sum(dim=1)
        return stats, comp_tokens, spec_token

    def _head_freqs(self, freq_stats, bsz, n_patch, device):
        k_obs = min(self.top_k, self.n_heads)
        frq = freq_stats[:, 1 + self.top_k: 1 + self.top_k + k_obs].to(device=device, dtype=torch.float32)
        if k_obs == self.n_heads:
            return frq
        n_fill = self.n_heads - k_obs
        f_lo = 1.0 / max(n_patch * self.patch_stride, 1)
        if self.n_heads == 1:
            fillers = torch.full((1,), f_lo, device=device, dtype=torch.float32)
        else:
            j = torch.arange(1, self.n_heads + 1, device=device, dtype=torch.float32)
            grid = f_lo + (j - 1) / (self.n_heads - 1) * (0.5 - f_lo)
            fillers = grid[:n_fill]
        return torch.cat([frq, fillers.unsqueeze(0).expand(bsz, -1)], dim=-1)

    def _harmonic_bias(self, freq_stats):
        bsz, n_patch = freq_stats.size(0), self.n_patch
        device, dtype = freq_stats.device, freq_stats.dtype
        head_freq = self._head_freqs(freq_stats, bsz, n_patch, device).to(dtype)
        tau = torch.arange(n_patch, device=device, dtype=dtype) * float(self.patch_stride)
        rel_time = tau[:, None] - tau[None, :]
        phase = 2 * torch.pi * head_freq[:, :, None, None] * rel_time[None, None, :, :]
        return torch.cos(phase)

    def _time_patch_pool(self, time_enc):
        x = time_enc.transpose(1, 2)
        x = F.pad(x, (0, self.patch_stride), mode='replicate')
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_stride)
        assert x.size(-2) == self.n_patch, (x.size(-2), self.n_patch)
        return x.mean(dim=-1).transpose(1, 2)

    def _decode(self, memory, ctx, x_mark_dec=None):
        b, c, l = ctx['b'], ctx['c'], ctx['l']
        phase_feats = self._fixed_future_feats(memory.size(0), memory.device)
        phase_feats = phase_feats + self._future_phase_feats(ctx['freq_stats'], hist_len=l)
        if x_mark_dec is not None:
            future_time = self.future_time_embedding(x_mark_dec[:, -self.pred_len:, :].float())
            future_time = future_time.unsqueeze(1).expand(-1, c, -1, -1).reshape(
                b * c, self.pred_len, self.d_model)
            phase_feats = phase_feats + torch.tanh(self.future_time_gate) * future_time

        dec = self.decoder(memory, phase_feats)
        return self._head_and_denorm(dec, ctx)

    def _head_and_denorm(self, dec, ctx):
        b, c, stats = ctx['b'], ctx['c'], ctx['stats']
        out = self.head(dec) + self.res_out_scale.view(1, -1, 1) * self.res_head(dec)
        out = out.mean(dim=-1)
        out = out.reshape(b, c, self.pred_len).permute(0, 2, 1)
        return self.norm.apply_denorm(out, stats)

    def _fixed_future_feats(self, bsz, device):
        half_dim = max(1, self.d_model // 2)
        t = torch.arange(self.pred_len, device=device, dtype=torch.float32)
        t = (t / max(self.pred_len - 1, 1)).unsqueeze(1)
        freqs = torch.exp(torch.linspace(
            math.log(1.0), math.log(max(self.fourier_fmax, 1.0)), half_dim,
            device=device)).unsqueeze(0)
        phase = 2 * torch.pi * t * freqs
        feats = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)
        if feats.size(-1) < self.d_model:
            feats = torch.cat([feats, feats.new_zeros(self.pred_len, self.d_model - feats.size(-1))], dim=-1)
        else:
            feats = feats[:, :self.d_model]
        return feats.unsqueeze(0).expand(bsz, -1, -1)

    def _future_phase_feats(self, freq_stats, hist_len):
        k = self.top_k
        amp_norm = freq_stats[:, 1: 1 + k]
        frq = freq_stats[:, 1 + k: 1 + 2 * k]
        phase_sin = freq_stats[:, 1 + 2 * k: 1 + 3 * k]
        phase_cos = freq_stats[:, 1 + 3 * k: 1 + 4 * k]
        phase0 = torch.nan_to_num(torch.atan2(phase_sin, phase_cos))

        h = torch.arange(self.pred_len, device=frq.device, dtype=frq.dtype).view(1, -1, 1)
        theta = phase0.unsqueeze(1) + 2 * torch.pi * frq.unsqueeze(1) * (hist_len + h)
        u = torch.cat([amp_norm.unsqueeze(1) * torch.sin(theta),
                       amp_norm.unsqueeze(1) * torch.cos(theta)], dim=-1)
        return torch.tanh(self.future_phase_gate) * self.future_phase_proj(u)

    def alignment_repr(self, history_x, future_x, history_mark=None):
        b, _, c = history_x.shape
        d = self.d_model

        if self._align_cache is not None:
            memory, ctx = self._align_cache
        else:
            memory, ctx = self.encode(history_x, history_mark)
        stats = ctx['stats']
        hist_repr = memory.reshape(b, c, memory.size(1), d).mean(dim=2).mean(dim=1)
        hist_repr = self.future_align_proj(hist_repr)

        future_norm = self.norm.apply_norm(future_x, stats)
        h = future_norm.size(1)
        f_flat = future_norm.permute(0, 2, 1).contiguous().reshape(b * c, h, 1)
        future_repr = self.future_encoder(f_flat)
        future_repr = self.future_align_proj(future_repr).reshape(b, c, h, d).mean(dim=1).mean(dim=1)
        return hist_repr, future_repr.detach()
