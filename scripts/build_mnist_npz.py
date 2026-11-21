#!/usr/bin/env python3
"""
Build the MNIST test bundles used by export_all.py.

Loads the MNIST test set through software/dataset.py (torchvision downloads it
on first use), applies the same preprocessing as training (resize to 16x16,
normalise to [-1, 1], pad each 16-pixel row to DIM=32), and writes

    scripts/mnist_test.npz       all 10,000 test images  (images (N,16,32) float32, labels (N,))
    scripts/mnist_test_100.npz   the first 100 of them   (bundled in the repository)

Usage:
    python scripts/build_mnist_npz.py                  # downloads MNIST into software/data/
    python scripts/build_mnist_npz.py --data-dir DIR   # reuse an existing torchvision MNIST root
"""
import argparse
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..', 'software'))
from dataset import MNIST16x16Dataset  # noqa: E402

DIM = 32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default=os.path.join(SCRIPT_DIR, '..', 'software', 'data'),
                    help='torchvision MNIST root (default: software/data)')
    args = ap.parse_args()

    ds = MNIST16x16Dataset(root=args.data_dir, is_train=False, hidden_dim=DIM)
    images = np.empty((len(ds), 16, DIM), dtype=np.float32)
    labels = np.empty(len(ds), dtype=np.int64)
    for i in range(len(ds)):
        x, y = ds[i]
        images[i] = x.numpy()
        labels[i] = y

    full = os.path.join(SCRIPT_DIR, 'mnist_test.npz')
    np.savez(full, images=images, labels=labels)
    print(f"wrote {full}  ({len(ds)} images)")
    small = os.path.join(SCRIPT_DIR, 'mnist_test_100.npz')
    np.savez(small, images=images[:100], labels=labels[:100])
    print(f"wrote {small}  (100 images)")


if __name__ == '__main__':
    main()
