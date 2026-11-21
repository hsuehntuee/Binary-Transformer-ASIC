#!/usr/bin/env python3
"""
Export bit-exact test patterns for the 1-bit Transformer RTL from the trained
checkpoint `scripts/1bit_best.pth`. All files are written under `tb/`.

Per-layer patterns for the chained-layer testbench:
    pattern_L0/  pattern_L1/  pattern_L2/  pattern_L3/

Host-side tokenizer and classifier data (Python only, not read by the RTL):
    pattern_full/
        raw_in_fp32.npy         input image (16 x 32, FP32)
        input_proj_w.npy        input projection weights (FP32)
        input_scale.npy         learned per-dim input scale
        pos_emb_int8.hex        positional embedding, INT8
        x_layer0_in.hex         = pattern_L0/x_in.hex
        mean_fp32.npy           mean-pooled layer-3 output
        classifier_w.npy        classifier weights (FP32)
        logits_gold.npy         classifier output
        predicted.txt           argmax + ground-truth label

Usage (from the repository root or from scripts/):
    python scripts/export_all.py [--sample N | --synthetic]
"""
import argparse
import os
import sys
import numpy as np
import torch

DIM         = 32
SEQ         = 16
P_SHIFT     = 3
NUM_LAYERS  = 4
NUM_CLASSES = 10
# Resolve the script's own directory ONCE at import, before main() chdir's into
# tb/. All input lookups (checkpoint, MNIST npz, dataset) anchor to this so they
# don't depend on cwd or whether __file__ is relative.
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
CKPT_PATH   = os.path.join(SCRIPT_DIR, "1bit_best.pth")


# =============================================================================
# HW-exact arithmetic (bit-accurate with the Verilog RTL)
# =============================================================================
def _newton_scale(sum_sq_val, beta_i):
    sq = int(sum_sq_val) & 0x7FFFF
    if sq == 0: return 0, 8
    E = sq.bit_length() - 1
    sum_norm = (sq << (18 - E)) & 0x7FFFF
    lut_idx  = (sum_norm >> 13) & 0x1F
    Y0_LUT = [254,250,247,243,240,236,233,230,228,225,222,220,217,215,212,210,
              208,206,204,202,200,198,196,194,193,191,189,188,186,185,183,182]
    y0 = Y0_LUT[lut_idx]
    M_frac = (sum_norm >> 4) & 0x3FFF
    y0_sq = ((y0 * y0) >> 2) & 0xFFFF
    M_full = (1 << 14) | M_frac
    m_y0_sq_raw = M_full * y0_sq
    half_m_y0_sq = (m_y0_sq_raw >> 15) & 0x7FFF
    diff = (0x6000 - half_m_y0_sq) & 0xFFFF
    y1 = ((y0 * diff) >> 8) & 0xFFFF
    E_eff = E - 5
    is_odd = E_eff & 1
    if is_odd:
        y1_comp = (y1 * 0x2D41) & 0xFFFFFFFF
        inv_M = (y1_comp >> 14) & 0xFFFF
    else:
        inv_M = y1
    shift_q5 = 8 + (E_eff >> 1)
    scale_raw = (int(beta_i) * inv_M) & 0xFFFFFFFF
    L = scale_raw.bit_length() - 1 if scale_raw > 0 else 0
    K = max(0, L - 8)
    scale_o = (scale_raw >> K) & 0x1FF
    total_shift = max(0, 18 + shift_q5 - K)
    return scale_o, total_shift


def beta_to_q4_12(beta_float):
    return max(1, min(65535, int(round(abs(float(beta_float)) * 4096))))


def hw_bitlinear(x_int8, w_float, beta_float):
    """INT8 input @ 1-bit weight, Newton-iterated INT8 dequant. w shape (out, in)."""
    safe = np.where(x_int8 == -128, -127, x_int8.astype(np.int32))
    w_bin = np.where(w_float >= 0, 1, -1).astype(np.int32)
    y_mac = safe.astype(np.int64) @ w_bin.T.astype(np.int64)
    beta_i = beta_to_q4_12(beta_float)
    out = np.zeros_like(y_mac, dtype=np.int8)
    for t in range(safe.shape[0]):
        sq = min(int(np.sum(safe[t].astype(np.int64) ** 2)), 0x7FFFF)
        scale_o, adj_shift = _newton_scale(sq, beta_i)
        for c in range(y_mac.shape[1]):
            shifted = (int(y_mac[t, c]) * scale_o * 2) >> adj_shift
            out[t, c] = max(-127, min(127, (shifted >> 1) + (shifted & 1)))
    return out


def hw_consmax_scalar(s_mac_int, beta_scale_int):
    s = int(s_mac_int)
    if s & (1 << 20): s_quant = 0
    elif (s >> 11) & 0x1FF: s_quant = 127
    else: s_quant = (s >> 4) & 0x7F
    _LUT_MSB = [0x0100, 0x0130, 0x016A, 0x01AE, 0x0200, 0x0260, 0x02D4, 0x035D,
                0x0400, 0x04C1, 0x05A8, 0x06BA, 0x0800, 0x0983, 0x0B50, 0x0D74]
    _LUT_LSB = [0x0100, 0x0102, 0x0105, 0x0108, 0x010B, 0x010E, 0x0111, 0x0114,
                0x0117, 0x011A, 0x011D, 0x0120, 0x0123, 0x0126, 0x0129, 0x012D]
    mult_lut = _LUT_MSB[s_quant >> 4] * _LUT_LSB[s_quant & 0xF]
    exp = (mult_lut >> 8) & 0xFFFFFF
    return min(127, (exp * int(beta_scale_int)) >> 8)


# =============================================================================
# Hex writers
# =============================================================================
def int8_to_2bit_msb_first(x):
    ux = int(x) & 0xFF
    return [(ux >> (6 - 2 * i)) & 0x3 for i in range(4)]


def export_hex(arr, path):
    with open(path, 'w') as f:
        for v in arr.flatten().astype(np.int8):
            f.write(f"{int(v) & 0xFF:02X}\n")


def export_1b(w_float, path):
    bits = np.where(w_float.flatten() >= 0, 1, 0)
    with open(path, 'w') as f:
        for b in bits:
            f.write(f"{int(b):01X}\n")


def export_16b(val, path):
    with open(path, 'w') as f:
        f.write(f"{int(val) & 0xFFFF:04X}\n")


def ste_clamp_round_int8(x_float):
    return np.clip(np.round(x_float), -127, 127).astype(np.int8)


# =============================================================================
# Layer-level pipeline (one transformer block, identical to hw_model_exact.py)
# =============================================================================
def run_layer(X_in, Wq, Wk, Wv, Wo, Wf1, Wf2,
              beta_q, beta_k, beta_v, beta_o, beta_f1, beta_f2, beta_attn_int,
              out_dir):
    os.makedirs(out_dir, exist_ok=True)

    export_hex(X_in, f"{out_dir}/x_in.hex")
    for w, name in [(Wq, 'q'), (Wk, 'k'), (Wv, 'v'), (Wo, 'o'), (Wf1, 'f1'), (Wf2, 'f2')]:
        export_1b(w.T, f"{out_dir}/{name}_w.hex")
    export_16b(beta_to_q4_12(beta_q),  f"{out_dir}/q_beta_16b.hex")
    export_16b(beta_to_q4_12(beta_k),  f"{out_dir}/k_beta_16b.hex")
    export_16b(beta_to_q4_12(beta_v),  f"{out_dir}/v_beta_16b.hex")
    export_16b(beta_to_q4_12(beta_o),  f"{out_dir}/o_beta_16b.hex")
    export_16b(beta_to_q4_12(beta_f1), f"{out_dir}/f1_beta_16b.hex")
    export_16b(beta_to_q4_12(beta_f2), f"{out_dir}/f2_beta_16b.hex")
    export_16b(beta_attn_int,          f"{out_dir}/attn_beta_scale.hex")

    Q = hw_bitlinear(X_in, Wq, beta_q)
    K = hw_bitlinear(X_in, Wk, beta_k)
    V = hw_bitlinear(X_in, Wv, beta_v)
    export_hex(Q, f"{out_dir}/q_out_8b.hex")
    export_hex(K, f"{out_dir}/k_out_8b.hex")
    export_hex(V, f"{out_dir}/v_out_8b.hex")

    with open(f"{out_dir}/stim_qk.hex", 'w') as f:
        for qi in range(SEQ):
            for ki in range(SEQ):
                for d in range(DIM):
                    q_c = int8_to_2bit_msb_first(Q[qi, d])
                    k_c = int8_to_2bit_msb_first(K[ki, d])
                    for c in range(4):
                        f.write(f"{(q_c[c] << 4) | k_c[c]:02X}\n")

    attn_out_full = np.zeros((SEQ, DIM), dtype=np.int8)
    for qi in range(SEQ):
        acc = np.zeros(DIM, dtype=np.int64)
        for ki in range(SEQ):
            p_ij = hw_consmax_scalar(
                int(np.dot(Q[qi].astype(np.int32), K[ki].astype(np.int32))),
                beta_attn_int)
            for d in range(DIM):
                acc[d] += int(p_ij) * int(V[ki, d])
        for d in range(DIM):
            attn_out_full[qi, d] = max(-127, min(127, int(acc[d]) >> P_SHIFT))

    # Output projection: o = o_proj(attn_out); residual-1 = clip(x + o).
    o_out    = hw_bitlinear(attn_out_full, Wo, beta_o)
    res1_out = np.clip(o_out.astype(np.int32) + X_in.astype(np.int32),
                       -127, 127).astype(np.int8)
    out_ffn1 = hw_bitlinear(res1_out, Wf1, beta_f1)
    relu     = np.where(out_ffn1 < 0,
                        np.floor((out_ffn1.astype(np.int32) + 4) / 8.0),
                        out_ffn1).astype(np.int8)
    out_ffn2 = hw_bitlinear(relu, Wf2, beta_f2)
    layer_out = np.clip(out_ffn2.astype(np.int32) + res1_out.astype(np.int32),
                        -127, 127).astype(np.int8)

    export_hex(attn_out_full, f"{out_dir}/golden_attn.hex")
    export_hex(o_out,         f"{out_dir}/golden_o.hex")
    export_hex(res1_out,      f"{out_dir}/golden_res1.hex")
    export_hex(out_ffn1,      f"{out_dir}/golden_ffn1.hex")
    export_hex(out_ffn2,      f"{out_dir}/golden_ffn2.hex")
    export_hex(layer_out,     f"{out_dir}/layer_gold.hex")
    return layer_out


# =============================================================================
# Checkpoint loader (full model, incl. tokenizer + classifier)
# =============================================================================
def load_checkpoint(path):
    # weights_only=True was added in PyTorch 1.13. Older torch (e.g. the
    # python3.6 / torch <=1.10 env on some lab servers) rejects the kwarg, so
    # try the safe form first and fall back to the legacy unpickler. Our
    # checkpoints are produced locally, so dropping the safety check is OK.
    try:
        sd = torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:
        sd = torch.load(path, map_location='cpu')
    pos_emb = sd['pos_embedding'].squeeze(0).numpy()
    ip_w    = sd['input_proj.weight'].numpy()
    cls_w   = sd['classifier.weight'].numpy()
    # Learnable per-dim input scale (lifts input_proj output into INT8 range
    # before quantization). Older checkpoints without this key default to 1.0
    # - matches the pre-scale behavior so old .pth files still run.
    if 'input_scale' in sd:
        ip_scale = sd['input_scale'].numpy().astype(np.float32)
    else:
        ip_scale = np.ones(ip_w.shape[0], dtype=np.float32)
        print("note: checkpoint has no 'input_scale'; defaulting to 1.0")
    layers = []
    for L in range(NUM_LAYERS):
        layers.append(dict(
            Wq=sd[f'layers.{L}.q_proj.weight'].numpy(),
            Wk=sd[f'layers.{L}.k_proj.weight'].numpy(),
            Wv=sd[f'layers.{L}.v_proj.weight'].numpy(),
            Wo=sd[f'layers.{L}.o_proj.weight'].numpy(),
            Wf1=sd[f'layers.{L}.ffn_1.weight'].numpy(),
            Wf2=sd[f'layers.{L}.ffn_2.weight'].numpy(),
            bq=float(sd[f'layers.{L}.q_proj.beta'].item()),
            bk=float(sd[f'layers.{L}.k_proj.beta'].item()),
            bv=float(sd[f'layers.{L}.v_proj.beta'].item()),
            bo=float(sd[f'layers.{L}.o_proj.beta'].item()),
            bf1=float(sd[f'layers.{L}.ffn_1.beta'].item()),
            bf2=float(sd[f'layers.{L}.ffn_2.beta'].item()),
            b_attn=max(1, min(65535,
                int(round(abs(float(sd[f'layers.{L}.attn.beta_scale'].item())))))),
        ))
    return pos_emb, ip_w, ip_scale, cls_w, layers


# =============================================================================
# Input source: real MNIST sample (dataset.py path) with synthetic fallback
# =============================================================================
def make_real_input(sample_idx=0):
    """Load test-set MNIST sample matching MNIST16x16Dataset preprocessing.

    Prefers a pre-baked bundle next to this script (images (N,16,32) float32,
    labels (N,)): mnist_test.npz (all 10,000, built by build_mnist_npz.py),
    then mnist_test_100.npz (first 100, bundled); else loads via torchvision.
    """
    for name in ('mnist_test.npz', 'mnist_test_100.npz'):
        npz_path = os.path.join(SCRIPT_DIR, name)
        if os.path.exists(npz_path):
            d = np.load(npz_path)
            if sample_idx < len(d['labels']):
                return d['images'][sample_idx].astype(np.float32), int(d['labels'][sample_idx])
    sys.path.insert(0, os.path.join(SCRIPT_DIR, '..', 'software'))
    from dataset import MNIST16x16Dataset
    data_dir = os.path.join(SCRIPT_DIR, '..', 'software', 'data')
    ds = MNIST16x16Dataset(root=data_dir, is_train=False, hidden_dim=DIM)
    x, label = ds[sample_idx]
    return x.numpy().astype(np.float32), int(label)


def make_synthetic_input(seed=42):
    rng = np.random.default_rng(seed)
    img = rng.uniform(-1.0, 1.0, (SEQ, 16)).astype(np.float32)
    if DIM > 16:
        img = np.pad(img, ((0, 0), (0, DIM - 16)))
    return img, -1


# =============================================================================
# Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--synthetic', action='store_true',
                    help='use deterministic random input instead of MNIST')
    ap.add_argument('--sample', type=int, default=0,
                    help='MNIST test sample index (default 0)')
    args = ap.parse_args()

    pos_emb, ip_w, ip_scale, cls_w, layers = load_checkpoint(CKPT_PATH)

    # Always write next to the testbench so RTL sim can find them.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(os.path.join(script_dir, "..", "tb"))

    if args.synthetic:
        raw, label = make_synthetic_input(seed=42)
        print(f"input: synthetic seed=42 (no label)")
    else:
        try:
            raw, label = make_real_input(sample_idx=args.sample)
            print(f"input: MNIST test sample {args.sample}, label={label}")
        except Exception as e:
            print(f"warning: MNIST load failed ({e}); falling back to synthetic")
            raw, label = make_synthetic_input(seed=42)

    # ---- pattern_full/: tokenizer + classifier artefacts (Python-verified) ----
    full_dir = "pattern_full"
    os.makedirs(full_dir, exist_ok=True)

    # Off-chip tokenizer (must mirror HWExact1BitModel.forward exactly):
    #   pos_hw = round( pos_emb * 127 )            INT8
    #   proj   = (raw @ ip_w.T) * ip_scale         FP32, broadcast over dim
    #   x_in   = round( proj + pos_hw )            INT8 (chip input)
    proj_out_fp = (raw @ ip_w.T) * ip_scale                          # (16, 32) float
    pos_hw      = ste_clamp_round_int8(pos_emb * 127.0)
    x_fp        = proj_out_fp + pos_hw.astype(np.float32)
    x           = ste_clamp_round_int8(x_fp)                          # INT8 chip input

    np.save(f"{full_dir}/raw_in_fp32.npy",   raw.astype(np.float32))
    np.save(f"{full_dir}/input_proj_w.npy",  ip_w.astype(np.float32))
    np.save(f"{full_dir}/input_scale.npy",   ip_scale)
    export_hex(pos_hw, f"{full_dir}/pos_emb_int8.hex")
    export_hex(x,      f"{full_dir}/x_layer0_in.hex")

    # ---- 4 transformer layers, per-layer pattern dirs ----
    for L, p in enumerate(layers):
        x = run_layer(
            x,
            p['Wq'], p['Wk'], p['Wv'], p['Wo'], p['Wf1'], p['Wf2'],
            p['bq'], p['bk'], p['bv'], p['bo'], p['bf1'], p['bf2'], p['b_attn'],
            f"pattern_L{L}")
        print(f"  layer {L} -> pattern_L{L}/")

    # ---- Off-chip post-processing: float mean pool + float classifier ----
    # `x` here is layer-3 INT8 output (chip's final output). The model does
    # x.mean(dim=1) - a true float mean - then a float Linear.
    mean_fp = x.astype(np.float32).mean(axis=0)                          # (32,) float
    logits  = mean_fp @ cls_w.T                                          # (NUM_CLASSES,) float

    np.save(f"{full_dir}/mean_fp32.npy",      mean_fp)
    np.save(f"{full_dir}/classifier_w.npy",   cls_w.astype(np.float32))
    np.save(f"{full_dir}/logits_gold.npy",    logits.astype(np.float32))

    pred = int(np.argmax(logits))
    with open(f"{full_dir}/predicted.txt", 'w') as f:
        f.write(f"pred={pred}\n")
        f.write(f"label={label}\n")
        f.write(f"logits={[round(float(v), 4) for v in logits]}\n")
    print(f"  predicted={pred}  label={label}  logits={[round(float(v),3) for v in logits]}")
    print(f"Done. pattern_L0..L3/  pattern_full/")


if __name__ == '__main__':
    main()
