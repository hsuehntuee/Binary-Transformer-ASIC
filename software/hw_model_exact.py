"""
hw_model_exact.py — Hardware-exact 1-bit Transformer model.

Targets the 9-bit Newton scale generator in rtl/bitlinear/ (an earlier
12-bit prototype is not in this repository). Forward = exact RTL integer output;
backward = differentiable float approximation (STE).

Newton (rtl/bitlinear/Newton_Scale_Generator.v):
  - 9-bit scale_o, 14-bit M_frac, y0² >> 2 to 16 bits
  - constants 0x6000, 0x2D41
  - T_q5 = 8 + (E_eff >> 1) with E_eff = E - 5
  - K = max(0, L - 8)

Dequant (rtl/bitlinear/Dequant_Formatter.v):
  - guard-bit add-before-shift round-half-up
  - sat to [-127, 127] (sat == -128 → -127)

LeakyReLU forward matches Verilog `(x[7]) ? (x + 4) >>> 3 : x`.
ConSmax/P_SHIFT = LUT-exact (consmax_core.v).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── STE helpers ───────────────────────────────────────────────────────────────

def ste_sign(x):
    bin_x = torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))
    return (bin_x - x).detach() + x

def ste_clamp_round(x, min_val, max_val):
    x_clamped = torch.clamp(torch.round(x), min_val, max_val)
    return (x_clamped - x).detach() + x

# ── Newton-Raphson inverse sqrt (matches rtl/bitlinear/Newton_Scale_Generator.v) ──

_Y0_LUT = np.array([254,250,247,243,240,236,233,230,228,225,222,220,217,215,212,210,
                     208,206,204,202,200,198,196,194,193,191,189,188,186,185,183,182],
                    dtype=np.int64)

def _newton_scale_9b(sum_sq_val, beta_i):
    """Scalar 9-bit Newton — matches rtl/bitlinear/Newton_Scale_Generator.v."""
    sq = int(sum_sq_val) & 0x7FFFF
    if sq == 0:
        return 0, 8
    E = sq.bit_length() - 1
    sum_norm = (sq << (18 - E)) & 0x7FFFF
    lut_idx  = (sum_norm >> 13) & 0x1F
    y0       = int(_Y0_LUT[lut_idx])
    M_frac   = (sum_norm >> 4) & 0x3FFF
    y0_sq    = ((y0 * y0) >> 2) & 0xFFFF
    M_full   = (1 << 14) | M_frac
    m_y0_sq  = M_full * y0_sq
    half_m   = (m_y0_sq >> 15) & 0x7FFF
    diff     = (0x6000 - half_m) & 0xFFFF
    y1       = ((y0 * diff) >> 8) & 0xFFFF
    E_eff    = E - 5
    is_odd   = E_eff & 1
    if is_odd:
        inv_M = ((y1 * 0x2D41) >> 14) & 0xFFFF
    else:
        inv_M = y1
    shift_q5  = 8 + (E_eff >> 1)
    scale_raw = (int(beta_i) * inv_M) & 0xFFFFFFFF
    L = scale_raw.bit_length() - 1 if scale_raw > 0 else 0
    K = max(0, L - 8)
    scale_o     = (scale_raw >> K) & 0x1FF
    total_shift = max(0, 18 + shift_q5 - K)
    return scale_o, total_shift


def _hw_bitlinear_batched(x_int8_2d, w_float, beta_float):
    """
    Exact 9-bit Newton BitLinear matching rtl/bitlinear/*.

    x_int8_2d : int8  numpy [N, in_features]
    w_float   : float numpy [out_features, in_features]
    returns   : int8  numpy [N, out_features]
    """
    N, _in_f = x_int8_2d.shape
    out_f    = w_float.shape[0]

    safe  = np.where(x_int8_2d == -128, -127, x_int8_2d.astype(np.int32))
    w_bin = np.where(w_float >= 0, 1, -1).astype(np.int32)
    y_mac = safe.astype(np.int64) @ w_bin.T.astype(np.int64)              # [N, out_f]

    sum_sq_arr = np.clip(np.sum(safe.astype(np.int64) ** 2, axis=1),
                         0, 0x7FFFF)                                       # [N]
    beta_i = max(1, min(65535, int(round(abs(float(beta_float)) * 4096))))

    # Newton scale per token (scalar loop — N is small in practice)
    scale_o_arr = np.empty(N, dtype=np.int64)
    shift_arr   = np.empty(N, dtype=np.int64)
    for i in range(N):
        s, sh = _newton_scale_9b(int(sum_sq_arr[i]), beta_i)
        scale_o_arr[i] = s
        shift_arr[i]   = sh

    # Dequant: guard-bit add-before-shift round-half-up, then saturate
    prod = y_mac * scale_o_arr[:, None] * 2                                # [N, out_f]
    out  = np.empty((N, out_f), dtype=np.int64)
    for i in range(N):
        sh      = int(shift_arr[i])
        shifted = prod[i] >> sh                                            # [out_f]
        rounded = (shifted >> 1) + (shifted & 1)
        sat     = np.clip(rounded, -128, 127)
        out[i]  = np.where(sat == -128, -127, sat)
    return out.astype(np.int8)

# ── ConSmax LUT (matches consmax_core.v exactly) ──────────────────────────────

_LUT_MSB = np.array([0x0100, 0x0130, 0x016A, 0x01AE, 0x0200, 0x0260, 0x02D4, 0x035D,
                      0x0400, 0x04C1, 0x05A8, 0x06BA, 0x0800, 0x0983, 0x0B50, 0x0D74],
                     dtype=np.int64)
_LUT_LSB = np.array([0x0100, 0x0102, 0x0105, 0x0108, 0x010B, 0x010E, 0x0111, 0x0114,
                      0x0117, 0x011A, 0x011D, 0x0120, 0x0123, 0x0126, 0x0129, 0x012D],
                     dtype=np.int64)

def _hw_consmax_batched(s_mac_int64, beta_reg):
    """
    Exact ConSmax matching consmax_core.v.
    s_mac_int64 : int64 numpy, raw Q·K dot product
    beta_reg    : integer ConSmax C constant
    returns     : uint8 numpy, p_ij in [0, 127]
    """
    s = s_mac_int64.astype(np.int64)
    s_quant = np.where(
        s < 0,                           np.int64(0),
        np.where((s >> 11) & 0x1FF != 0, np.int64(127),
                 (s >> 4) & np.int64(0x7F))
    )
    mult = _LUT_MSB[s_quant >> 4] * _LUT_LSB[s_quant & 0xF]
    exp  = (mult >> 8) & 0xFFFFFF
    p    = (exp * int(beta_reg)) >> 8
    return np.clip(p, 0, 127).astype(np.uint8)

# ── Hardware-exact model classes ───────────────────────────────────────────────

class HWExactBitLinear(nn.Module):
    def __init__(self, in_features, out_features, init_beta=8.0, use_hw_forward=True):
        super().__init__()
        self.in_features    = in_features
        self.out_features   = out_features
        self.use_hw_forward = use_hw_forward
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.uniform_(self.weight, -0.1, 0.1)
        self.beta = nn.Parameter(torch.tensor(float(init_beta)))

    def forward(self, x):
        x_int8  = ste_clamp_round(x, -127, 127)
        w_bin   = ste_sign(self.weight)
        y_mac_f = F.linear(x_int8, w_bin, None)

        # Float approx — backward gradient (differentiable w.r.t. all params)
        sum_sq    = torch.sum(x_int8 ** 2, dim=-1, keepdim=True) + 1e-5
        out_float = torch.clamp(
            y_mac_f * torch.abs(self.beta) / torch.sqrt(sum_sq), -127, 127)

        if not self.use_hw_forward:
            return out_float

        # HW-exact forward: float division + round-to-nearest (matches FFN_golden)
        x_np  = x_int8.detach().cpu().numpy().astype(np.int8)
        shape = x_np.shape
        hw_np = _hw_bitlinear_batched(
            x_np.reshape(-1, shape[-1]),
            self.weight.detach().cpu().numpy(),
            self.beta.item())
        hw_out = torch.tensor(
            hw_np.reshape(shape[:-1] + (self.out_features,)),
            dtype=torch.float32, device=x.device)

        # STE: forward = hw_out, backward ≈ out_float
        return (hw_out - out_float).detach() + out_float


class HWExactConSmaxAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.beta_scale = nn.Parameter(torch.tensor(8.0))

    def forward(self, q, k, v):
        q_int8 = ste_clamp_round(q, -127, 127)
        k_int8 = ste_clamp_round(k, -127, 127)
        v_int8 = ste_clamp_round(v, -127, 127)

        s_mac = torch.matmul(q_int8, k_int8.transpose(-2, -1))

        # Float approx for backward. STE around floor so gradient reaches Q/K:
        # forward = floor(s_mac/16) clamped to [0,127]; backward = identity on
        # the clamped linear value. Without this, torch.floor zeroes the
        # gradient and Q/K projections never learn.
        s_raw   = s_mac * 0.0625
        s_f_fwd = torch.clamp(torch.floor(s_raw), 0, 127)
        s_f_bwd = torch.clamp(s_raw,              0, 127)
        s_f     = (s_f_fwd - s_f_bwd).detach() + s_f_bwd
        p_f   = torch.clamp((2.0 ** (s_f * 0.015625)) * torch.abs(self.beta_scale), 0, 127)
        pv_f  = torch.matmul(p_f, v_int8)
        out_f = torch.clamp(pv_f / 8.0, -127, 127)

        # Exact hw output: LUT ConSmax + floor P_SHIFT
        beta_r  = max(1, min(65535, int(round(abs(self.beta_scale.item())))))
        s_np    = s_mac.detach().cpu().numpy().astype(np.int64)
        p_hw_np = _hw_consmax_batched(s_np, beta_r).astype(np.float32)
        p_hw    = torch.tensor(p_hw_np, dtype=torch.float32, device=q.device)
        pv_hw   = torch.matmul(p_hw, v_int8)
        out_hw  = torch.clamp(torch.floor(pv_hw / 8.0), -127, 127)

        # STE: forward = out_hw, backward ≈ out_f
        return (out_hw - out_f).detach() + out_f


def hw_leaky_relu(x):
    """STE-wrapped HW LeakyReLU: forward = (x[7]) ? (x+4)>>3 : x, integer-exact."""
    x_q = ste_clamp_round(x, -127, 127)                 # ensure INT8 range
    x_i = x_q.to(torch.int32)
    pos_mask = (x_i >= 0)
    neg_out  = torch.where(pos_mask, torch.zeros_like(x_i), (x_i + 4) >> 3)
    out_int  = torch.where(pos_mask, x_i, neg_out).to(torch.float32)
    # STE: forward = hw integer; backward gradient via float (out_float ≈ x_q for >=0,
    # x_q/8 for <0 — same shape but differentiable)
    out_float = torch.where(x_q >= 0, x_q, x_q / 8.0)
    return (out_int - out_float).detach() + out_float


class HWExactTransformerBlock(nn.Module):
    def __init__(self, dim, use_hw_bitlinear=True):
        super().__init__()
        self.q_proj        = HWExactBitLinear(dim, dim, init_beta=8.0, use_hw_forward=use_hw_bitlinear)
        self.k_proj        = HWExactBitLinear(dim, dim, init_beta=8.0, use_hw_forward=use_hw_bitlinear)
        self.v_proj        = HWExactBitLinear(dim, dim, init_beta=8.0, use_hw_forward=use_hw_bitlinear)
        self.attn          = HWExactConSmaxAttention(dim)
        self.o_proj        = HWExactBitLinear(dim, dim, init_beta=8.0, use_hw_forward=use_hw_bitlinear)
        self.ffn_1         = HWExactBitLinear(dim, dim, init_beta=8.0, use_hw_forward=use_hw_bitlinear)
        self.ffn_2         = HWExactBitLinear(dim, dim, init_beta=2.0, use_hw_forward=use_hw_bitlinear)

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        attn_out = self.attn(q, k, v)
        o = self.o_proj(attn_out)
        # Residual 1 (computed on the host, see tb/tb_layer.v) saturates
        # to [-127, 127] — both bounds.
        res1     = ste_clamp_round(x + o, -127, 127)
        ffn1_out = hw_leaky_relu(self.ffn_1(res1))
        ffn_out  = self.ffn_2(ffn1_out)
        # Residual 2 (also on the host) saturates to [-127, 127], matching the
        # post-attention residual and BitNet symmetric-INT8 convention. -128
        # never appears on the inter-layer wire.
        return ste_clamp_round(res1 + ffn_out, -127, 127)


class HWExact1BitModel(nn.Module):
    """
    Chip runs only the 4 transformer blocks. input_proj and classifier are
    off-chip and use full-precision nn.Linear for better end-to-end accuracy.

    input_scale: per-dim learnable scale applied to input_proj output before
    adding pos_emb and quantizing to INT8. nn.Linear's default init produces
    outputs in [-1, 1] which rounds to {-1, 0, 1} when quantized — destroying
    most of the image information at the chip's input. input_scale lifts that
    into INT8 range; init=127 so the chip starts seeing full-range inputs.
    """
    def __init__(self, num_layers=4, dim=32, num_classes=10, use_hw_bitlinear=True):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, 16, dim) * 0.2)
        self.input_proj    = nn.Linear(dim, dim, bias=False)            # off-chip, FP32
        self.input_scale   = nn.Parameter(torch.full((dim,), 127.0))    # per-dim learnable
        self.layers        = nn.ModuleList(
            [HWExactTransformerBlock(dim, use_hw_bitlinear=use_hw_bitlinear) for _ in range(num_layers)])
        self.classifier    = nn.Linear(dim, num_classes, bias=False)    # off-chip, FP32

    def forward(self, x):
        # Off-chip tokenizer: float linear (scaled) + scaled pos_emb, then quantize to INT8 for the chip.
        pos_hw = ste_clamp_round(self.pos_embedding * 127.0, -127, 127)
        proj   = self.input_proj(x) * self.input_scale                  # broadcast over (B, T, dim)
        x      = ste_clamp_round(proj + pos_hw, -127, 127)
        # On-chip: 4 HW-exact transformer blocks (9-bit Newton).
        for layer in self.layers:
            x = layer(x)
        # Off-chip classifier: mean pool then float linear → logits.
        return self.classifier(x.mean(dim=1))
