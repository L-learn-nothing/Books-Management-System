from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from utils import ensure_dir, seed_everything


MODALITY_NAMES = ["t1", "t1ce", "t2", "flair"]


def make_case(case_idx: int, slices_per_case: int, image_size: int) -> list[dict[str, str | int]]:
    center_base = image_size // 2 + np.random.randint(-8, 8, size=2)
    radius_base = np.random.randint(image_size // 10, image_size // 6)
    yy, xx = np.mgrid[:image_size, :image_size]
    records = []
    for slice_idx in range(slices_per_case):
        image = np.zeros((4, image_size, image_size), dtype=np.float32)
        shift = np.random.randint(-4, 5, size=2)
        center = center_base + shift
        radius = radius_base + np.random.randint(-2, 3)
        tumor = ((yy - center[0]) ** 2 + (xx - center[1]) ** 2) <= radius**2
        edema = ((yy - center[0]) ** 2 + (xx - center[1]) ** 2) <= (radius + 6) ** 2
        image[0] = np.random.normal(0.35, 0.08, size=(image_size, image_size))
        image[1] = np.random.normal(0.30, 0.08, size=(image_size, image_size)) + tumor * 0.65
        image[2] = np.random.normal(0.40, 0.09, size=(image_size, image_size)) + edema * 0.35
        image[3] = np.random.normal(0.25, 0.08, size=(image_size, image_size)) + edema * 0.60
        image = np.clip(image, 0.0, 1.0)
        mask = tumor.astype(np.float32)
        records.append({"image": image, "mask": mask, "case_id": f"toy_{case_idx:03d}", "slice_idx": slice_idx})
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic multi-modal MRI toy dataset.")
    parser.add_argument("--output_dir", default="toy_dataset", help="Where to write the dataset.")
    parser.add_argument("--num_cases", type=int, default=24)
    parser.add_argument("--slices_per_case", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    output_dir = ensure_dir(args.output_dir)
    npz_dir = ensure_dir(output_dir / "npz")

    split_names = {"train": [], "val": [], "test": []}
    for case_idx in range(args.num_cases):
        split = "train" if case_idx < int(args.num_cases * 0.7) else "val" if case_idx < int(args.num_cases * 0.85) else "test"
        for sample in make_case(case_idx, args.slices_per_case, args.image_size):
            file_name = f"{sample['case_id']}_slice_{sample['slice_idx']:03d}.npz"
            np.savez_compressed(npz_dir / file_name, image=sample["image"], mask=sample["mask"])
            split_names[split].append({"path": f"npz/{file_name}", "case_id": sample["case_id"], "slice_idx": sample["slice_idx"]})

    for split, records in split_names.items():
        with (output_dir / f"{split}.json").open("w", encoding="utf-8") as file:
            json.dump(records, file, indent=2, ensure_ascii=False)

    summary = {
        "description": "Toy 4-modality MRI dataset for whole tumor segmentation.",
        "modalities": MODALITY_NAMES,
        "label": "WT(binary)",
        "num_cases": args.num_cases,
        "slices_per_case": args.slices_per_case,
        "image_size": args.image_size,
    }
    with (output_dir / "dataset_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    print(f"Toy dataset generated at: {output_dir}")


if __name__ == "__main__":
    main()
