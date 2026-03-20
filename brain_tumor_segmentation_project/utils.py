from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def save_json(data: dict, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def plot_history(history: dict[str, list[float]], output_path: str | Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["train_loss"], label="train_loss")
    axes[0].plot(history["val_loss"], label="val_loss")
    axes[0].legend()
    axes[0].set_title("Loss")
    axes[1].plot(history["val_dice"], label="val_dice")
    axes[1].plot(history["val_iou"], label="val_iou")
    axes[1].legend()
    axes[1].set_title("Metrics")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_preview(image: np.ndarray, mask: np.ndarray, pred: np.ndarray, output_path: str | Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image[3], cmap="gray")
    axes[0].set_title("FLAIR")
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("GT")
    axes[2].imshow(pred, cmap="gray")
    axes[2].set_title("Prediction")
    for axis in axes:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
