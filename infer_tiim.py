#!/usr/bin/env python3
"""Single-image TIIM inference for nuScenes CAM_FRONT or a calibrated camera image."""

from __future__ import annotations

import argparse
import ast
import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

import src.model.network as networks
from src import utils


DEFAULT_PRED_CLASSES_NUSC = [
    "drivable_area",
    "ped_crossing",
    "walkway",
    "carpark_area",
    "bus",
    "bicycle",
    "car",
    "construction_vehicle",
    "motorcycle",
    "trailer",
    "truck",
    "pedestrian",
    "trafficcone",
    "barrier",
]


DEFAULT_CONFIG = {
    "model_name": "PyrOccTranDetr_S_0904_old_rep100x100_out100x100",
    "desired_image_size": [1600, 900],
    "grid_res": 0.5,
    "z_intervals": [1.0, 9.0, 21.0, 39.0, 51.0],
    "pred_classes_nusc": DEFAULT_PRED_CLASSES_NUSC,
    "frontend": "resnet50",
    "focal_length": 1266.417,
    "scales": [8.0, 16.0, 32.0, 64.0],
    "y_crop": [15.0, 15.0, 15.0, 15.0],
    "dla_norm": "GroupNorm",
    "bevt_linear_additions": False,
    "bevt_conv_additions": False,
    "dla_l1_nchannels": 64,
    "n_enc_layers": 2,
    "n_dec_layers": 2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TIIM BEV inference on one nuScenes CAM_FRONT sample or one calibrated image."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pth/.pth.gz.")
    parser.add_argument(
        "--config",
        help="Path to checkpoint config.txt. Defaults to config.txt next to --checkpoint.",
    )
    parser.add_argument("--output-dir", default="outputs/tiim_infer", help="PNG output directory.")
    parser.add_argument("--device", default="cuda", help="Torch device, usually cuda on Jetson Thor.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Mask threshold after sigmoid.")
    parser.add_argument("--fp16", action="store_true", help="Run model/image in FP16.")
    parser.add_argument(
        "--allow-partial-checkpoint",
        action="store_true",
        help="Allow missing/unexpected checkpoint keys. Off by default to avoid invalid predictions.",
    )

    source = parser.add_argument_group("input source")
    source.add_argument("--data-root", default="nuscenes_data", help="nuScenes mini root.")
    source.add_argument("--nusc-version", default="v1.0-mini", help="nuScenes version.")
    source.add_argument("--camera", default="CAM_FRONT", help="nuScenes camera channel.")
    source.add_argument("--sample-index", type=int, default=0, help="Sample index if no token is given.")
    source.add_argument("--sample-token", help="nuScenes sample token.")
    source.add_argument("--sample-data-token", help="nuScenes sample_data token.")
    source.add_argument("--image", help="Direct image path for non-nuScenes camera inference.")
    source.add_argument("--video", help="Direct video path for non-nuScenes camera inference.")
    source.add_argument("--video-frame", type=int, default=30, help="Frame index to read from --video.")
    source.add_argument(
        "--intrinsics",
        help="Direct camera intrinsics as fx,fy,cx,cy or nine comma-separated row-major K values.",
    )
    source.add_argument("--intrinsics-file", help="Direct intrinsics file: .npy, .json, or text.")

    return parser.parse_args()


def load_config(path: str | None) -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    if not path:
        return config

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    text = config_path.read_text().strip()
    loaded: dict[str, Any] | None = None

    for loader in (json.loads, ast.literal_eval):
        try:
            value = loader(text)
        except Exception:
            continue
        if isinstance(value, dict):
            loaded = value
            break

    if loaded is None:
        loaded = {}
        for line in text.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip().strip("'\"")
            value = value.strip()
            if not key:
                continue
            try:
                loaded[key] = ast.literal_eval(value)
            except Exception:
                loaded[key] = value

    aliases = {
        "bevt_linear_additions": "additions_BEVT_linear",
        "bevt_conv_additions": "additions_BEVT_conv",
        "dla_l1_nchannels": "dla_l1_n_channels",
    }
    config.update(loaded)
    for old, new in aliases.items():
        if old in config and new not in config:
            config[new] = config[old]
    return config


def config_path_for_checkpoint(checkpoint: str, explicit_config: str | None) -> str | None:
    if explicit_config:
        return explicit_config
    sibling = Path(checkpoint).resolve().parent / "config.txt"
    return str(sibling) if sibling.exists() else None


def require_model_class(model_name: str) -> type[torch.nn.Module]:
    model_cls = getattr(networks, model_name, None)
    if model_cls is None:
        available = sorted(name for name in networks.__dict__ if name.startswith("Pyr"))
        raise RuntimeError(
            f"Checkpoint model_name '{model_name}' is not present in src/model/network.py. "
            f"Available model classes: {', '.join(available)}"
        )
    return model_cls


def build_model(config: dict[str, Any]) -> torch.nn.Module:
    model_name = str(config["model_name"])
    model_cls = require_model_class(model_name)

    z_range = [float(v) for v in config["z_intervals"]]
    h_cropped = config.get("cropped_height")
    if h_cropped is None:
        h_cropped = utils.calc_cropped_heights(
            float(config.get("focal_length", DEFAULT_CONFIG["focal_length"])),
            np.array(config.get("y_crop", DEFAULT_CONFIG["y_crop"]), dtype=np.float32),
            z_range,
            config.get("scales", DEFAULT_CONFIG["scales"]),
        )

    return model_cls(
        num_classes=len(config["pred_classes_nusc"]),
        frontend=config.get("frontend", DEFAULT_CONFIG["frontend"]),
        grid_res=float(config["grid_res"]),
        pretrained=False,
        img_dims=[int(v) for v in config["desired_image_size"]],
        z_range=z_range,
        h_cropped=[float(v) for v in h_cropped],
        dla_norm=config.get("dla_norm", DEFAULT_CONFIG["dla_norm"]),
        additions_BEVT_linear=bool(
            config.get("additions_BEVT_linear", config.get("bevt_linear_additions", False))
        ),
        additions_BEVT_conv=bool(
            config.get("additions_BEVT_conv", config.get("bevt_conv_additions", False))
        ),
        dla_l1_n_channels=int(
            config.get("dla_l1_n_channels", config.get("dla_l1_nchannels", 64))
        ),
        n_enc_layers=int(config.get("n_enc_layers", 2)),
        n_dec_layers=int(config.get("n_dec_layers", 2)),
    )


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, allow_partial: bool) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    if not isinstance(state, dict):
        raise RuntimeError(f"Checkpoint does not contain a state dict: {checkpoint_path}")

    stripped = OrderedDict()
    for key, value in state.items():
        stripped[key[7:] if key.startswith("module.") else key] = value

    missing, unexpected = model.load_state_dict(stripped, strict=False)
    if missing or unexpected:
        message = []
        if missing:
            message.append(f"missing checkpoint keys: {len(missing)}")
            message.extend(f"  {key}" for key in missing[:20])
        if unexpected:
            message.append(f"unexpected checkpoint keys: {len(unexpected)}")
            message.extend(f"  {key}" for key in unexpected[:20])
        if not allow_partial:
            raise RuntimeError(
                "Checkpoint does not exactly match the selected model. "
                "Use the checkpoint's matching config/model, or pass "
                "--allow-partial-checkpoint only for debugging.\n" + "\n".join(message)
            )
        print("Warning: partial checkpoint load allowed.")
        print("\n".join(message))


def load_intrinsics_from_file(path: str) -> np.ndarray:
    p = Path(path)
    if p.suffix == ".npy":
        value = np.load(p)
    elif p.suffix == ".json":
        data = json.loads(p.read_text())
        value = data.get(
            "camera_matrix",
            data.get("camera_intrinsic", data.get("intrinsics", data.get("K", data))),
        )
    else:
        text = p.read_text()
        value = np.loadtxt(p, delimiter="," if "," in text else None)
    return np.asarray(value, dtype=np.float32).reshape(3, 3)


def parse_intrinsics(value: str | None, path: str | None) -> np.ndarray:
    if path:
        return load_intrinsics_from_file(path)
    if not value:
        raise ValueError("--intrinsics or --intrinsics-file is required with --image")
    vals = [float(v.strip()) for v in value.split(",")]
    if len(vals) == 4:
        fx, fy, cx, cy = vals
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    if len(vals) == 9:
        return np.array(vals, dtype=np.float32).reshape(3, 3)
    raise ValueError("--intrinsics must contain either 4 values (fx,fy,cx,cy) or 9 K values")


def resize_image_and_calib(
    image: Image.Image, calib: np.ndarray, desired_size: list[int]
) -> tuple[Image.Image, np.ndarray]:
    desired_w, desired_h = int(desired_size[0]), int(desired_size[1])
    src_w, src_h = image.size
    scale_w = desired_w / float(src_w)
    scale_h = desired_h / float(src_h)
    resized = image.resize((desired_w, desired_h), Image.BILINEAR)
    scaled_calib = calib.astype(np.float32).copy()
    scaled_calib[0, :] *= scale_w
    scaled_calib[1, :] *= scale_h
    return resized, scaled_calib


def load_nuscenes_sample(args: argparse.Namespace, desired_size: list[int]) -> tuple[Image.Image, np.ndarray, str]:
    from nuscenes.nuscenes import NuScenes

    nusc = NuScenes(version=args.nusc_version, dataroot=args.data_root, verbose=False)
    if args.sample_data_token:
        sample_data = nusc.get("sample_data", args.sample_data_token)
    else:
        if args.sample_token:
            sample = nusc.get("sample", args.sample_token)
            sample_data = nusc.get("sample_data", sample["data"][args.camera])
        else:
            sample_data = None
            for sample_index in range(args.sample_index, len(nusc.sample)):
                sample = nusc.sample[sample_index]
                candidate = nusc.get("sample_data", sample["data"][args.camera])
                if Path(nusc.get_sample_data_path(candidate["token"])).exists():
                    sample_data = candidate
                    if sample_index != args.sample_index:
                        print(
                            f"Sample index {args.sample_index} image is missing; "
                            f"using first local {args.camera} image at index {sample_index}."
                        )
                    break
            if sample_data is None:
                raise FileNotFoundError(
                    f"No local {args.camera} image found in {args.data_root} "
                    f"starting at sample index {args.sample_index}."
                )

    image_path = nusc.get_sample_data_path(sample_data["token"])
    if not Path(image_path).exists():
        raise FileNotFoundError(f"nuScenes image not found: {image_path}")
    sensor = nusc.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
    calib = np.asarray(sensor["camera_intrinsic"], dtype=np.float32)
    image = Image.open(image_path).convert("RGB")
    image, calib = resize_image_and_calib(image, calib, desired_size)
    return image, calib, sample_data["token"]


def load_direct_image(args: argparse.Namespace, desired_size: list[int]) -> tuple[Image.Image, np.ndarray, str]:
    image = Image.open(args.image).convert("RGB")
    calib = parse_intrinsics(args.intrinsics, args.intrinsics_file)
    image, calib = resize_image_and_calib(image, calib, desired_size)
    return image, calib, Path(args.image).stem


def load_direct_video(args: argparse.Namespace, desired_size: list[int]) -> tuple[Image.Image, np.ndarray, str]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for --video. Install python3-opencv.") from exc

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.video_frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {args.video_frame} from {args.video}")

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    calib = parse_intrinsics(args.intrinsics, args.intrinsics_file)
    image, calib = resize_image_and_calib(image, calib, desired_size)
    return image, calib, f"{Path(args.video).stem}_frame{args.video_frame:06d}"


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = np.transpose(array, (2, 0, 1))
    return torch.from_numpy(array).unsqueeze(0)


def save_outputs(
    logits: torch.Tensor,
    class_names: list[str],
    output_dir: str,
    threshold: float,
    source_id: str,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    probs = torch.sigmoid(logits).detach().cpu().float().numpy()
    if probs.ndim != 3:
        raise RuntimeError(f"Expected [C,H,W] output, got {probs.shape}")

    for idx, class_name in enumerate(class_names):
        safe_name = class_name.replace("/", "_")
        prob = np.clip(probs[idx] * 255.0, 0, 255).astype(np.uint8)
        mask = (probs[idx] >= threshold).astype(np.uint8) * 255
        Image.fromarray(prob).save(out / f"{idx:02d}_{safe_name}_prob.png")
        Image.fromarray(mask).save(out / f"{idx:02d}_{safe_name}_mask.png")

    class_idx = np.arange(1, len(class_names) + 1, dtype=np.float32)[:, None, None]
    composite = ((probs >= threshold).astype(np.float32) * class_idx).max(axis=0)
    if composite.max() > 0:
        composite = composite / composite.max()
    Image.fromarray((composite * 255.0).astype(np.uint8)).save(out / "composite_mask.png")

    summary = {
        "source_id": source_id,
        "classes": class_names,
        "threshold": threshold,
        "output_shape": list(probs.shape),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    config = load_config(config_path_for_checkpoint(str(checkpoint), args.config))

    desired_size = [int(v) for v in config["desired_image_size"]]
    class_names = list(config["pred_classes_nusc"])
    z_range = [float(v) for v in config["z_intervals"]]
    grid_size = config.get("grid_size", (z_range[-1] - z_range[0], z_range[-1] - z_range[0]))

    print("Resolved checkpoint config:")
    for key in ("model_name", "desired_image_size", "grid_res", "z_intervals", "pred_classes_nusc"):
        print(f"  {key}: {config[key]}")

    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True

    model = build_model(config)
    load_checkpoint(model, str(checkpoint), args.allow_partial_checkpoint)
    model.to(device).eval()
    if args.fp16:
        model.half()

    if args.image and args.video:
        raise ValueError("Use either --image or --video, not both.")
    if args.image:
        image, calib, source_id = load_direct_image(args, desired_size)
    elif args.video:
        image, calib, source_id = load_direct_video(args, desired_size)
    else:
        image, calib, source_id = load_nuscenes_sample(args, desired_size)

    image_tensor = image_to_tensor(image).to(device)
    if args.fp16:
        image_tensor = image_tensor.half()
    tensor_dtype = torch.float16 if args.fp16 else torch.float32
    calib_tensor = torch.as_tensor(calib, dtype=tensor_dtype, device=device).unsqueeze(0)
    grid = utils.make_grid2d(grid_size, (-grid_size[0] / 2.0, 0.0), float(config["grid_res"]))
    grid = grid.unsqueeze(0).to(device=device, dtype=tensor_dtype)

    with torch.inference_mode():
        pred_ms = model(image_tensor, calib_tensor, grid)

    main_logits = pred_ms[0][0]
    save_outputs(main_logits, class_names, args.output_dir, args.threshold, source_id)
    print(f"Saved {len(class_names)} probability maps and masks to {args.output_dir}")


if __name__ == "__main__":
    main()
