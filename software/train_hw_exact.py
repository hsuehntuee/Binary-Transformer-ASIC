"""
train_hw_exact.py — Train with hardware-exact forward pass.

Weights trained here are optimal for what the RTL actually computes (Newton
inverse sqrt, LUT ConSmax, floor P_SHIFT) rather than float approximations.
Backward pass still uses differentiable float approximations
(STE), so standard optimizers work unchanged.

GPU is supported but each forward pass has CPU↔GPU data-transfer overhead
(numpy requires CPU). For this model size, CPU training is typically faster.

Saves weights to (relative to the working directory):
  checkpoints/1bit.pth            final epoch
  checkpoints/1bit_best.pth       best test accuracy so far
  checkpoints/1bit_epochN.pth     checkpoint every CKPT_EVERY epochs
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import MNIST16x16Dataset
from hw_model_exact import HWExact1BitModel

SAVE_DIR = "checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

MODEL_SAVE      = os.path.join(SAVE_DIR, '1bit.pth')
MODEL_SAVE_BEST = os.path.join(SAVE_DIR, '1bit_best.pth')
CKPT_EVERY      = 50   # also save 1bit_epochN.pth every N epochs


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print("Note: hw-exact forward has CPU<->GPU transfer overhead per layer. "
              "Consider running on CPU: CUDA_VISIBLE_DEVICES='' python train_hw_exact.py")

    HIDDEN_DIM = 32
    NUM_LAYERS = 4
    BATCH_SIZE = 256
    EPOCHS     = 1000

    data_dir = './data'
    os.makedirs(data_dir, exist_ok=True)
    try:
        train_ds = MNIST16x16Dataset(root=data_dir, is_train=True,  hidden_dim=HIDDEN_DIM)
        test_ds  = MNIST16x16Dataset(root=data_dir, is_train=False, hidden_dim=HIDDEN_DIM)
    except TypeError:
        train_ds = MNIST16x16Dataset(data_dir, True,  HIDDEN_DIM)
        test_ds  = MNIST16x16Dataset(data_dir, False, HIDDEN_DIM)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # use_hw_bitlinear=True: NR forward (dynamic normalisation, correct β/√Σx²).
    # Newton_Scale_Generator_new fixes is_odd parity and uses LZD — NR ≈ float sqrt,
    # so STE gap is small and training converges normally.
    model = HWExact1BitModel(
        num_layers=NUM_LAYERS, dim=HIDDEN_DIM, num_classes=10,
        use_hw_bitlinear=True).to(device)

    # Separate LRs: scale/bias params need no weight decay
    scale_params, weight_params = [], []
    for name, p in model.named_parameters():
        if 'beta' in name or 'pos_embedding' in name or 'input_scale' in name:
            scale_params.append(p)
        else:
            weight_params.append(p)

    optimizer = torch.optim.AdamW([
        {'params': weight_params, 'weight_decay': 0.01},
        {'params': scale_params,  'weight_decay': 0.0},
    ], lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    print(f"Training HWExact1BitModel on {device} for {EPOCHS} epochs")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Train batches: {len(train_loader)},  Test batches: {len(test_loader)}")

    best_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            output = model(data)
            loss   = criterion(output, target)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Keep binary weights in [-1, 1] so ste_sign stays meaningful
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if 'weight' in name:
                        p.data.clamp_(-1.0, 1.0)

            total_loss += loss.item()
            pred    = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total   += target.size(0)

        train_acc = 100.0 * correct / total
        scheduler.step()

        model.eval()
        test_correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred   = output.argmax(dim=1, keepdim=True)
                test_correct += pred.eq(target.view_as(pred)).sum().item()

        test_acc = 100.0 * test_correct / len(test_loader.dataset)

        tags = []
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), MODEL_SAVE_BEST)
            tags.append(f"** best {best_acc:.2f}% → {MODEL_SAVE_BEST}")

        if (epoch + 1) % CKPT_EVERY == 0:
            ckpt = MODEL_SAVE.replace('.pth', f'_epoch{epoch+1}.pth')
            torch.save(model.state_dict(), ckpt)
            tags.append(f"ckpt → {ckpt}")

        print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
              f"LR: {optimizer.param_groups[0]['lr']:.5f} | "
              f"Loss: {total_loss / len(train_loader):.4f} | "
              f"Train: {train_acc:.2f}% | Test: {test_acc:.2f}%"
              + (f"  [{', '.join(tags)}]" if tags else ""))

    torch.save(model.state_dict(), MODEL_SAVE)
    print(f"\nSaved final → {MODEL_SAVE}")
    print(f"Best test acc: {best_acc:.2f}%  (weights in {MODEL_SAVE_BEST})")


if __name__ == '__main__':
    train()
