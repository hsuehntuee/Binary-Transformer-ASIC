"""
Continue fine-tuning HWExact1BitModel, resuming from previous checkpoints if available.

- If '1bit_finetune_ckpt.pth' exists, resumes training (model, optimizer, scheduler, epoch, best_acc).
- Else, can optionally load a best pre-trained model for fine-tuning.
- Checkpoints are saved every CKPT_EVERY epochs and at finish.

You may adjust FINETUNE_CKPT, PRETRAINED_BEST, and EPOCHS_FT as needed.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import MNIST16x16Dataset
from hw_model_exact import HWExact1BitModel

SAVE_DIR = "checkpoints_ft"
os.makedirs(SAVE_DIR, exist_ok=True)

# File to save and resume fine-tuning progress
FINETUNE_CKPT = os.path.join(SAVE_DIR, '1bit_finetune_ckpt.pth')
# Optionally use the best pre-trained weight for starting
PRETRAINED_BEST = os.path.join(SAVE_DIR, '../checkpoints/1bit_best.pth')
# How many fine-tuning epochs to run (if resuming, continues for this many epochs more)
EPOCHS_FT = 1000

CKPT_EVERY = 50

def train_finetune():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    HIDDEN_DIM = 32
    NUM_LAYERS = 4
    BATCH_SIZE = 256

    data_dir = './data'
    os.makedirs(data_dir, exist_ok=True)
    try:
        train_ds = MNIST16x16Dataset(root=data_dir, is_train=True,  hidden_dim=HIDDEN_DIM)
        test_ds  = MNIST16x16Dataset(root=data_dir, is_train=False, hidden_dim=HIDDEN_DIM)
    except TypeError:
        train_ds = MNIST16x16Dataset(data_dir, True,  HIDDEN_DIM)
        test_ds  = MNIST16x16Dataset(data_dir, False, HIDDEN_DIM)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = HWExact1BitModel(
        num_layers=NUM_LAYERS, dim=HIDDEN_DIM, num_classes=10,
        use_hw_bitlinear=True
    ).to(device)

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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_FT)
    criterion = nn.CrossEntropyLoss()

    # ---- Resume logic ----
    start_epoch = 0
    best_acc = 0.0
    resumed = False

    if os.path.exists(FINETUNE_CKPT):
        print(f"Resuming finetuning from checkpoint: {FINETUNE_CKPT}")
        ckpt = torch.load(FINETUNE_CKPT, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_acc = ckpt.get('best_acc', 0.0)
        resumed = True
    elif os.path.exists(PRETRAINED_BEST):
        print(f"Initializing from best pre-trained weights: {PRETRAINED_BEST}")
        model.load_state_dict(torch.load(PRETRAINED_BEST, map_location=device))
        # optimizer and scheduler fresh; start_epoch=0, best_acc=0
    else:
        print("Training from scratch.")

    print(f"Fine-tuning HWExact1BitModel on {device} for {EPOCHS_FT} additional epochs")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    print(f"Resumed checkpoint: {resumed}  |  Start epoch: {start_epoch}  |  Previous best_acc: {best_acc:.2f}%")

    for epoch in range(start_epoch, start_epoch + EPOCHS_FT):
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

            # Clamp binary weights to [-1, 1]
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
        # Save best model weights and checkpoint every time test_acc improves
        if test_acc > best_acc:
            best_acc = test_acc
            # Save both "model-only" and "full checkpoint" for best
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, '1bit_finetune_best_model.pth'))
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
            }, os.path.join(SAVE_DIR, '1bit_finetune_best.pth'))
            tags.append(f"** best {best_acc:.2f}% → 1bit_finetune_best.pth")
        
        # Regular periodic checkpoint (latest state)
        if (epoch + 1) % CKPT_EVERY == 0 or (epoch == start_epoch + EPOCHS_FT - 1):
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
            }
            torch.save(checkpoint, FINETUNE_CKPT)
            tags.append(f"ckpt → {FINETUNE_CKPT}")

        print(f"Epoch {epoch+1:04d} | "
              f"LR: {optimizer.param_groups[0]['lr']:.5f} | "
              f"Loss: {total_loss / len(train_loader):.4f} | "
              f"Train: {train_acc:.2f}% | Test: {test_acc:.2f}%"
              + (f"  [{', '.join(tags)}]" if tags else ""))

    # Save final model as separate file for inference (weights only)
    FINAL_PTH = os.path.join(SAVE_DIR, '1bit_finetune_final.pth')
    torch.save(model.state_dict(), FINAL_PTH)
    print(f"\nSaved final model weights → {FINAL_PTH}")
    print(f"Best test acc during fine-tuning: {best_acc:.2f}%  (full checkpoint in {FINETUNE_CKPT})")

if __name__ == '__main__':
    train_finetune()
