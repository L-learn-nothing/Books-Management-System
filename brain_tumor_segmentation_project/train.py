from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import NPZSegmentationDataset
from losses import DiceBCELoss, dice_score, iou_score
from models.unet import build_model
from utils import ensure_dir, plot_history, save_json, seed_everything


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    for batch in tqdm(loader, leave=False):
        image = batch["image"].to(device)
        mask = batch["mask"].to(device)
        with torch.set_grad_enabled(is_train):
            output = model(image)
            logits = output["logits"]
            loss = criterion(logits, mask)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += float(loss.item())
        total_dice += dice_score(logits.detach(), mask)
        total_iou += iou_score(logits.detach(), mask)
    size = max(1, len(loader))
    return total_loss / size, total_dice / size, total_iou / size


def evaluate(model, loader, device):
    model.eval()
    total_dice = 0.0
    total_iou = 0.0
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            mask = batch["mask"].to(device)
            logits = model(image)["logits"]
            total_dice += dice_score(logits, mask)
            total_iou += iou_score(logits, mask)
    size = max(1, len(loader))
    return {"dice": total_dice / size, "iou": total_iou / size}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a toy/beginner-friendly brain tumor segmentation model.")
    parser.add_argument("--data_dir", default="toy_dataset")
    parser.add_argument("--model", choices=["baseline", "gate"], default="gate")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--output_root", default="outputs")
    args = parser.parse_args()

    seed_everything(args.seed)
    data_dir = Path(args.data_dir)
    output_dir = ensure_dir(Path(args.output_root) / f"{args.model}_{time.strftime('%Y%m%d_%H%M%S')}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = NPZSegmentationDataset(data_dir / "train.json")
    val_dataset = NPZSegmentationDataset(data_dir / "val.json")
    test_dataset = NPZSegmentationDataset(data_dir / "test.json")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_model(args.model).to(device)
    criterion = DiceBCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = {"train_loss": [], "val_loss": [], "val_dice": [], "val_iou": []}
    best_dice = -1.0
    best_path = output_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_dice, train_iou = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_dice, val_iou = run_epoch(model, val_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)
        history["val_iou"].append(val_iou)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_dice={train_dice:.4f} train_iou={train_iou:.4f} | "
            f"val_loss={val_loss:.4f} val_dice={val_dice:.4f} val_iou={val_iou:.4f}"
        )
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({"model_state": model.state_dict(), "model_name": args.model}, best_path)

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = evaluate(model, test_loader, device)

    save_json(history, output_dir / "history.json")
    save_json(test_metrics, output_dir / "test_metrics.json")
    plot_history(history, output_dir / "training_curves.png")
    save_json(vars(args), output_dir / "train_args.json")
    print(f"Best checkpoint: {best_path}")
    print(f"Test metrics: {test_metrics}")


if __name__ == "__main__":
    main()
