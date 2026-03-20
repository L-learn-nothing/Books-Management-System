from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from models.unet import build_model
from utils import ensure_dir, save_json, save_preview


MODALITIES = ["t1", "t1ce", "t2", "flair"]


def zscore(volume: np.ndarray) -> np.ndarray:
    foreground = volume[volume > 0]
    if foreground.size == 0:
        return volume.astype(np.float32)
    mean = foreground.mean()
    std = foreground.std() + 1e-6
    return ((volume - mean) / std).astype(np.float32)


def load_case(case_dir: Path) -> np.ndarray:
    volumes = []
    for modality in MODALITIES:
        path = next(case_dir.glob(f"*_{modality}.nii.gz"))
        volumes.append(zscore(nib.load(path).get_fdata()))
    return np.stack(volumes, axis=0).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference on a single BraTS case.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--case_dir", required=True)
    parser.add_argument("--output_dir", default="predictions")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = build_model(checkpoint.get("model_name", "gate")).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    case_dir = Path(args.case_dir)
    case_id = case_dir.name
    image = load_case(case_dir)
    prediction = np.zeros(image.shape[1:], dtype=np.float32)

    with torch.no_grad():
        for slice_idx in range(image.shape[-1]):
            image_slice = torch.from_numpy(image[:, :, :, slice_idx][None, ...]).to(device)
            logits = model(image_slice)["logits"]
            pred = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()[0, 0]
            prediction[:, :, slice_idx] = pred

    output_dir = ensure_dir(args.output_dir)
    affine = np.eye(4)
    nib.save(nib.Nifti1Image(prediction, affine), output_dir / f"{case_id}_pred_wt.nii.gz")

    mid = prediction.shape[-1] // 2
    save_preview(image[:, :, :, mid], np.zeros_like(prediction[:, :, mid]), prediction[:, :, mid], output_dir / f"{case_id}_preview.png")
    save_json({"case_id": case_id, "output": f"{case_id}_pred_wt.nii.gz"}, output_dir / f"{case_id}_result.json")
    print(f"Prediction saved to {output_dir}")


if __name__ == "__main__":
    main()
