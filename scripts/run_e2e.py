#!/usr/bin/env python3
"""
run_e2e.py - End-to-end inference using the trained checkpoint and RTL sim.

Pipeline (28-pad redesign):
  1. (Python) Load checkpoint, run off-chip tokenizer (input_proj FP32 + pos_emb)
     and write per-layer patterns for the 4 transformer layers, including the
     o_proj goldens (delegated to export_all.py).
  2. (Verilog/VCS) Run tb_4layer (run_4layer_vcs.sh): drives all chip modes
     (PROJ Q/K/V/O -> ATTN -> off-chip res1 -> FFN -> off-chip res2) for the 4
     cascaded layers, checks every stage vs goldens, and dumps the L3 chip
     output to tb/pattern_full/rtl_l3_out.hex.
  3. (Python) Read that hex, run off-chip mean-pool + classifier (FP32) on the
     RTL output, print logits + prediction.

The crucial property: the classifier sees the *RTL-simulated* L3 tensor, not
the Python-emulated one. If the gold mean+classifier (in pattern_full/) and
the RTL-driven one agree, the chain is end-to-end verified.

VCS only. MNIST samples 0-99 come from the bundled mnist_test_100.npz; for
other indices run build_mnist_npz.py once (or let export_all.py load MNIST
through torchvision). With --synthetic the label reads -1 (the
RTL-vs-Python-gold comparison still holds).

Usage:
    cd scripts/
    python3 run_e2e.py                       # sample 0, VCS
    python3 run_e2e.py --sample 42
    python3 run_e2e.py --synthetic
    python3 run_e2e.py --samples 0-9
"""
import argparse
import os
import subprocess
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def read_int8_hex(path, n):
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    out = []
    for h in lines[:n]:
        v = int(h, 16)
        if v > 127:
            v -= 256
        out.append(v)
    return np.array(out, dtype=np.int8)


def parse_samples(s):
    """'0,3,7' -> [0,3,7];  '0-4' -> [0,1,2,3,4];  '5' -> [5]."""
    out = []
    for tok in s.split(','):
        tok = tok.strip()
        if '-' in tok:
            a, b = tok.split('-')
            out.extend(range(int(a), int(b) + 1))
        elif tok:
            out.append(int(tok))
    return out


# 28-pad redesign: the 4-layer E2E TB (tb_4layer.v) drives all chip modes
# (PROJ Q/K/V/O -> ATTN -> off-chip res1 -> FFN -> off-chip res2) for 4 cascaded
# layers and dumps the L3 chip output to pattern_full/rtl_l3_out.hex. VCS only.
SIM_SCRIPTS = {
    'vcs': 'run_4layer_vcs.sh',
}


def run_one(sample, synthetic, verbose, sim='vcs'):
    """Returns dict with sample, rtl_pred, py_pred, label, rtl_ok, label_ok."""
    cmd = [sys.executable, 'export_all.py']
    if synthetic:
        cmd.append('--synthetic')
    else:
        cmd += ['--sample', str(sample)]
    subprocess.check_call(cmd, cwd=HERE,
        stdout=(None if verbose else subprocess.DEVNULL))

    sim_script = SIM_SCRIPTS[sim]
    # capture_output= was added in Python 3.7; spell it out so this runs on
    # the python3.6 lab server too.
    res = subprocess.run(['bash', sim_script],
        cwd=os.path.join(HERE, '..', 'tb'),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        universal_newlines=True)
    if '4-LAYER E2E PASS' not in res.stdout:
        print(f"[ERR sample={sample}, sim={sim}] RTL sim failed")
        print(res.stdout[-2000:])
        if res.stderr:
            print('--- stderr ---')
            print(res.stderr[-2000:])
        sys.exit(1)

    rtl_l3 = read_int8_hex(
        os.path.join(HERE, '..', 'tb', 'pattern_full', 'rtl_l3_out.hex'),
        16 * 32).reshape(16, 32)
    cls_w  = np.load(os.path.join(HERE, '..', 'tb', 'pattern_full', 'classifier_w.npy'))
    logits = rtl_l3.astype(np.float32).mean(axis=0) @ cls_w.T
    rtl_pred = int(np.argmax(logits))

    meta = {}
    with open(os.path.join(HERE, '..', 'tb', 'pattern_full', 'predicted.txt')) as f:
        for ln in f.read().strip().splitlines():
            if '=' in ln:
                k, _, v = ln.partition('=')
                meta[k.strip()] = v.strip()
    py_pred = int(meta.get('pred', -1))
    label   = int(meta.get('label', -1))

    return dict(sample=sample, rtl_pred=rtl_pred, py_pred=py_pred,
                label=label, logits=logits)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--sample',  type=int, default=0,
                    help='single MNIST test-set index (default 0)')
    g.add_argument('--samples', type=str,
                    help='batch: comma list or range, e.g. "0,3,7" or "0-9"')
    ap.add_argument('--synthetic', action='store_true',
                    help='use synthetic random input (no label)')
    ap.add_argument('--sim', choices=sorted(SIM_SCRIPTS), default='vcs',
                    help='RTL simulator backend (default: vcs)')
    args = ap.parse_args()

    samples = parse_samples(args.samples) if args.samples else [args.sample]
    batch   = len(samples) > 1

    results = []
    for i, s in enumerate(samples):
        if batch:
            print(f"[{i+1}/{len(samples)}] sample={s} ...", end=' ', flush=True)
        else:
            print(f"running sample={s}")
        r = run_one(s, args.synthetic, verbose=not batch, sim=args.sim)
        results.append(r)
        ok_rtl = (r['rtl_pred'] == r['py_pred'])
        ok_lab = (r['label'] >= 0 and r['rtl_pred'] == r['label'])
        if batch:
            tag = ('OK' if ok_lab else ('=' if ok_rtl else 'X'))
            print(f"rtl={r['rtl_pred']} py={r['py_pred']} label={r['label']} {tag}")

    # Summary
    if batch:
        n        = len(results)
        rtl_eq   = sum(r['rtl_pred'] == r['py_pred'] for r in results)
        rtl_corr = sum(r['label'] >= 0 and r['rtl_pred'] == r['label'] for r in results)
        py_corr  = sum(r['label'] >= 0 and r['py_pred']  == r['label'] for r in results)
        labelled = sum(r['label'] >= 0 for r in results)
        print(f"\n{'='*55}")
        print(f"  Samples:           {n}")
        print(f"  RTL == Python:     {rtl_eq}/{n}  ({100.0*rtl_eq/n:.1f}%)")
        if labelled:
            print(f"  RTL correct:       {rtl_corr}/{labelled}  ({100.0*rtl_corr/labelled:.1f}%)")
            print(f"  Python correct:    {py_corr}/{labelled}  ({100.0*py_corr/labelled:.1f}%)")
        print('='*55)
        sys.exit(0 if rtl_eq == n else 2)

    # Single-sample detail
    r = results[0]
    gold_logits = np.load(os.path.join(HERE, '..', 'tb', 'pattern_full', 'logits_gold.npy'))
    print(f"\n  RTL  logits   = [{', '.join(f'{v:7.3f}' for v in r['logits'])}]")
    print(f"  Gold logits   = [{', '.join(f'{v:7.3f}' for v in gold_logits)}]")
    print(f"  max |diff|    = {np.max(np.abs(r['logits'] - gold_logits)):.6f}")
    print(f"  RTL    pred   = {r['rtl_pred']}")
    print(f"  Python pred   = {r['py_pred']}")
    print(f"  Dataset label = {r['label']}"
          + ("  (synthetic - no label)" if r['label'] < 0 else ""))
    if r['rtl_pred'] != r['py_pred']:
        print("\n[E2E FAIL] RTL prediction differs from Python gold.")
        sys.exit(2)
    if r['label'] < 0:
        print("\n[E2E PASS] RTL == Python gold (no dataset label).")
    elif r['rtl_pred'] == r['label']:
        print(f"\n[E2E PASS] RTL prediction matches dataset label {r['label']}.")
    else:
        print(f"\n[E2E PARTIAL] RTL == Python gold, but model misclassified "
              f"(pred={r['rtl_pred']}, label={r['label']}).")


if __name__ == '__main__':
    main()