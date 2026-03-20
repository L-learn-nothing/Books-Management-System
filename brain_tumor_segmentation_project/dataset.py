from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class NPZSegmentationDataset(Dataset):
    def __init__(self, index_file: str | Path) -> None:
        self.index_file = Path(index_file)
        with self.index_file.open("r", encoding="utf-8") as file:
            self.samples: list[dict[str, Any]] = json.load(file)
        self.root = self.index_file.parent

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[idx]
        data = np.load(self.root / sample["path"])
        image = data["image"].astype(np.float32)
        mask = data["mask"].astype(np.float32)
        return {
            "image": torch.from_numpy(image),
            "mask": torch.from_numpy(mask[None, ...]),
            "case_id": sample["case_id"],
            "slice_idx": str(sample["slice_idx"]),
        }
