from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from utils import ensure_dir, seed_everything


MODALITIES = ["t1", "t1ce", "t2", "flair"]


def zscore(volume: np.ndarray) -> np.ndarray:
    foreground = volume[volume > 0]
    if foreground.size == 0:
        return volume.astype(np.float32)
    mean = foreground.mean()
    std = foreground.std() + 1e-6
    return ((volume - mean) / std).astype(np.float32)


def center_crop_2d(array: np.ndarray, size: int) -> np.ndarray:
    height, width = array.shape[-2:]
    start_y = max((height - size) // 2, 0)
    start_x = max((width - size) // 2, 0)
    return array[..., start_y:start_y + size, start_x:start_x + size]


def load_case(case_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    case_id = case_dir.name
    volumes = []
    for modality in MODALITIES:
        path = next(case_dir.glob(f"*_{modality}.nii.gz"))
        volumes.append(zscore(nib.load(path).get_fdata()))
    seg_path = next(case_dir.glob("*_seg.nii.gz"))
    seg = nib.load(seg_path).get_fdata()
    wt = (seg > 0).astype(np.float32)
    image = np.stack(volumes, axis=0)
    return image, wt


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert BraTS NIfTI files into 2D npz slices.")
    parser.add_argument("--brats_root", required=True)
    parser.add_argument("--output_dir", default="data_npz")
    parser.add_argument("--max_cases", type=int, default=20)
    parser.add_argument("--image_size", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    root = Path(args.brats_root)
    output_dir = ensure_dir(args.output_dir)
    npz_dir = ensure_dir(output_dir / "npz")
    case_dirs = sorted([p for p in root.iterdir() if p.is_dir()])[: args.max_cases]
    split_records = {"train": [], "val": [], "test": []}

    for case_idx, case_dir in enumerate(case_dirs):
        image, wt = load_case(case_dir)
        split = "train" if case_idx < int(len(case_dirs) * 0.7) else "val" if case_idx < int(len(case_dirs) * 0.85) else "test"
        case_id = case_dir.name
        for slice_idx in range(image.shape[-1]):
            image_slice = center_crop_2d(image[:, :, :, slice_idx], args.image_size)
            mask_slice = center_crop_2d(wt[:, :, slice_idx], args.image_size)
            if mask_slice.sum() == 0:
                continue
            file_name = f"{case_id}_slice_{slice_idx:03d}.npz"
            np.savez_compressed(npz_dir / file_name, image=image_slice.astype(np.float32), mask=mask_slice.astype(np.float32))
            split_records[split].append({"path": f"npz/{file_name}", "case_id": case_id, "slice_idx": slice_idx})

    for split, records in split_records.items():
        with (output_dir / f"{split}.json").open("w", encoding="utf-8") as file:
            json.dump(records, file, indent=2, ensure_ascii=False)

    print(f"Prepared {sum(len(v) for v in split_records.values())} slices into {output_dir}")


if __name__ == "__main__":
    main()
