import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, classification_report,
)
from sklearn.model_selection import StratifiedKFold

from dataset import RugbyClipDataset, CLASS_NAMES
from model import create_model


BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "..", "knock_on_dataset")
SAVE_PATH  = os.path.join(BASE_DIR, "knock_on_classifier.pt")
PLOT_DIR   = os.path.join(BASE_DIR, "..", "training_plots")


def compute_metrics(all_preds, all_labels):
    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    accuracy = correct / len(all_labels) if all_labels else 0

    tp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(all_preds, all_labels) if p == 0 and l == 1)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return accuracy, precision, recall


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for clips, labels in loader:
        clips = clips.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(clips)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * clips.size(0)
        _, preds = outputs.max(1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    epoch_loss = running_loss / len(all_labels)
    accuracy, precision, recall = compute_metrics(all_preds, all_labels)
    return epoch_loss, accuracy, precision, recall


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    for clips, labels in loader:
        clips = clips.to(device)
        labels = labels.to(device)

        outputs = model(clips)
        loss = criterion(outputs, labels)
        probs = torch.softmax(outputs, dim=1)[:, 1]

        running_loss += loss.item() * clips.size(0)
        _, preds = outputs.max(1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())

    epoch_loss = running_loss / len(all_labels)
    accuracy, precision, recall = compute_metrics(all_preds, all_labels)
    return epoch_loss, accuracy, precision, recall, all_preds, all_labels, all_probs


def plot_confusion_matrix(all_preds, all_labels):
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Confusion Matrix (All Folds Aggregated)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()
    print("Saved confusion_matrix.png")


def plot_roc_curve(all_labels, all_probs):
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 7))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--", label="Random (AUC = 0.5)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Knock-On Detection (All Folds)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "roc_curve.png"), dpi=150)
    plt.close()
    print(f"Saved roc_curve.png  (AUC = {roc_auc:.4f})")
    return roc_auc


def _plot_fold_curves(history, fold_idx):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history["epoch"], history["train_loss"], label="Train Loss", marker="o", markersize=2)
    ax1.plot(history["epoch"], history["val_loss"], label="Val Loss", marker="s", markersize=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"Fold {fold_idx} - Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history["epoch"], history["train_acc"], label="Train Acc", marker="o", markersize=2)
    ax2.plot(history["epoch"], history["val_acc"], label="Val Acc", marker="s", markersize=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title(f"Fold {fold_idx} - Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"fold_{fold_idx}_curves.png"), dpi=150)
    plt.close()
    print(f"Saved fold_{fold_idx}_curves.png")


class _SubsetWithAugment(torch.utils.data.Dataset):

    def __init__(self, full_dataset, indices, augment):
        self.full_dataset = full_dataset
        self.indices = indices
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        original = self.full_dataset.augment
        self.full_dataset.augment = self.augment
        item = self.full_dataset[self.indices[idx]]
        self.full_dataset.augment = original
        return item


def main():
    parser = argparse.ArgumentParser(description="Train Knock-On Video Classifier")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--bs", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--folds", type=int, default=5, help="K-Fold splits")
    parser.add_argument("--frames", type=int, default=24, help="Frames per clip")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    full_dataset = RugbyClipDataset(
        root_dir=DATA_DIR,
        frames_per_clip=args.frames,
        frame_size=224,
        augment=False,
    )
    if len(full_dataset) == 0:
        print("ERROR: No video clips found. Check knock_on_dataset/ folder.")
        return

    all_labels = [label for _, label in full_dataset.samples]

    # k-fold
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)

    fold_results = []
    all_fold_preds, all_fold_labels, all_fold_probs = [], [], []

    for fold_idx, (train_indices, val_indices) in enumerate(skf.split(np.zeros(len(all_labels)), all_labels), 1):
        print(f"\n{'#'*70}")
        print(f"  FOLD {fold_idx} / {args.folds}")
        print(f"{'#'*70}")

        train_dataset = _SubsetWithAugment(full_dataset, train_indices.tolist(), augment=True)
        val_dataset = _SubsetWithAugment(full_dataset, val_indices.tolist(), augment=False)

        train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True,
                                  num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False,
                                num_workers=0, pin_memory=True)

        train_ko = sum(1 for i in train_indices if all_labels[i] == 1)
        val_ko = sum(1 for i in val_indices if all_labels[i] == 1)
        print(f"  Train: {len(train_dataset)} clips ({train_ko} knock_on)")
        print(f"  Val:   {len(val_dataset)} clips ({val_ko} knock_on)")

        # fresh model per fold
        model = create_model(num_classes=2, pretrained=True, freeze_early=True)
        model = model.to(device)

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=5e-4,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=5, factor=0.5,
        )

        best_val_acc = 0.0
        best_preds, best_labels, best_probs = [], [], []
        history = {
            "epoch": [], "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [], "val_prec": [], "val_rec": [],
        }

        header = (f"{'Epoch':>5} | {'T.Loss':>7} | {'T.Acc':>6} | "
                  f"{'V.Loss':>7} | {'V.Acc':>6} | {'Prec':>5} | {'Rec':>5} | {'Time':>5}")
        print(f"\n{'-'*70}")
        print(header)
        print(f"{'-'*70}")

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()

            t_loss, t_acc, _, _ = train_one_epoch(model, train_loader, criterion, optimizer, device)
            v_loss, v_acc, v_prec, v_rec, v_preds, v_labels, v_probs = validate(
                model, val_loader, criterion, device
            )

            scheduler.step(v_acc)
            elapsed = time.time() - t0

            history["epoch"].append(epoch)
            history["train_loss"].append(t_loss)
            history["train_acc"].append(t_acc)
            history["val_loss"].append(v_loss)
            history["val_acc"].append(v_acc)
            history["val_prec"].append(v_prec)
            history["val_rec"].append(v_rec)

            print(f"{epoch:>5} | {t_loss:>7.4f} | {t_acc:>5.1%} | "
                  f"{v_loss:>7.4f} | {v_acc:>5.1%} | {v_prec:>5.2f} | {v_rec:>5.2f} | "
                  f"{elapsed:>4.0f}s")

            if v_acc > best_val_acc:
                best_val_acc = v_acc
                best_preds, best_labels, best_probs = v_preds, v_labels, v_probs
                torch.save({
                    "epoch": epoch,
                    "fold": fold_idx,
                    "model_state_dict": model.state_dict(),
                    "val_acc": v_acc,
                    "precision": v_prec,
                    "recall": v_rec,
                    "class_names": CLASS_NAMES,
                    "frames_per_clip": args.frames,
                }, SAVE_PATH)

        # collect fold results
        _, best_prec, best_rec = compute_metrics(best_preds, best_labels)
        fold_results.append({
            "fold": fold_idx,
            "val_acc": best_val_acc,
            "precision": best_prec,
            "recall": best_rec,
        })
        all_fold_preds.extend(best_preds)
        all_fold_labels.extend(best_labels)
        all_fold_probs.extend(best_probs)

        print(f"\n  Fold {fold_idx} best -> acc: {best_val_acc:.1%}, "
              f"prec: {best_prec:.2f}, rec: {best_rec:.2f}")

        os.makedirs(PLOT_DIR, exist_ok=True)
        _plot_fold_curves(history, fold_idx)

    # cv summary
    accs = [r["val_acc"] for r in fold_results]
    precs = [r["precision"] for r in fold_results]
    recs = [r["recall"] for r in fold_results]

    print(f"\n{'='*70}")
    print(f"  {args.folds}-FOLD CROSS-VALIDATION RESULTS")
    print(f"{'='*70}")
    for r in fold_results:
        print(f"  Fold {r['fold']}: Acc={r['val_acc']:.4f}  "
              f"Prec={r['precision']:.4f}  Rec={r['recall']:.4f}")
    print(f"{'-'*70}")
    print(f"  Mean Accuracy:  {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
    print(f"  Mean Precision: {np.mean(precs):.4f} +/- {np.std(precs):.4f}")
    print(f"  Mean Recall:    {np.mean(recs):.4f} +/- {np.std(recs):.4f}")
    print(f"{'='*70}")

    plot_confusion_matrix(all_fold_preds, all_fold_labels)
    roc_auc = plot_roc_curve(all_fold_labels, all_fold_probs)

    print(f"\nClassification Report (all folds aggregated):")
    print(classification_report(all_fold_labels, all_fold_preds, target_names=CLASS_NAMES))

    print(f"  AUC: {roc_auc:.4f}")
    print(f"  Model saved to: {SAVE_PATH}")
    print(f"  Plots saved to: {PLOT_DIR}")
    print(f"\nRun inference with: python inference.py --video path/to/video.mp4")


if __name__ == "__main__":
    main()
