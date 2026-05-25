#!/usr/bin/env python3
"""Live TIIM camera-to-BEV web viewer."""

from __future__ import annotations

import argparse
import json
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import torch

from infer_tiim import (
    build_model,
    config_path_for_checkpoint,
    image_to_tensor,
    load_checkpoint,
    load_config,
    load_intrinsics_from_file,
    resize_image_and_calib,
)
from src import utils


PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TIIM Live BEV</title>
  <style>
    :root {
      color-scheme: dark;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #0f1114;
      color: #eef3f7;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      background: #0f1114;
    }
    header {
      min-height: 56px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 10px 16px;
      border-bottom: 1px solid #2a3036;
      background: #171b20;
    }
    h1 {
      margin: 0;
      font-size: 18px;
      font-weight: 650;
      letter-spacing: 0;
    }
    #status {
      font-size: 13px;
      color: #b6c0ca;
      text-align: right;
    }
    main {
      flex: 1;
      min-height: 0;
      display: grid;
      grid-template-columns: minmax(0, 1.35fr) minmax(0, 1fr);
      gap: 12px;
      padding: 12px;
    }
    section {
      min-width: 0;
      min-height: 0;
      display: flex;
      flex-direction: column;
      border: 1px solid #2a3036;
      background: #080a0c;
    }
    .label {
      height: 34px;
      display: flex;
      align-items: center;
      padding: 0 12px;
      border-bottom: 1px solid #2a3036;
      color: #cbd3db;
      font-size: 13px;
      font-weight: 600;
    }
    .frame {
      flex: 1;
      min-height: 0;
      display: grid;
      place-items: center;
      overflow: hidden;
      background: #030405;
    }
    img {
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: block;
    }
    @media (max-width: 900px) {
      header { align-items: flex-start; flex-direction: column; }
      #status { text-align: left; }
      main { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <h1>TIIM Live BEV</h1>
    <div id="status">Starting</div>
  </header>
  <main>
    <section>
      <div class="label">Front Camera</div>
      <div class="frame"><img src="/camera.mjpg" alt="Front camera stream"></div>
    </section>
    <section>
      <div class="label">BEV Semantic Map</div>
      <div class="frame"><img src="/bev.mjpg" alt="BEV semantic stream"></div>
    </section>
  </main>
  <script>
    async function refreshStatus() {
      try {
        const response = await fetch('/api/status', { cache: 'no-store' });
        const data = await response.json();
        const parts = [
          data.camera_ok ? 'camera on' : 'camera waiting',
          data.model_ok ? 'model on' : 'model loading',
          `camera ${data.camera_frames}`,
          `bev ${data.bev_frames}`,
          `infer ${data.inference_fps.toFixed(2)} fps`,
          `cuda ${data.cuda_allocated_mib.toFixed(0)} MiB`
        ];
        if (data.last_error) parts.push(data.last_error);
        document.getElementById('status').textContent = parts.join(' | ');
      } catch (error) {
        document.getElementById('status').textContent = 'status unavailable';
      }
    }
    setInterval(refreshStatus, 1000);
    refreshStatus();
  </script>
</body>
</html>
"""


PALETTE = np.array(
    [
        [70, 160, 70],
        [245, 215, 75],
        [90, 200, 215],
        [165, 135, 90],
        [215, 90, 70],
        [130, 190, 255],
        [80, 120, 255],
        [190, 110, 210],
        [255, 140, 75],
        [180, 180, 180],
        [120, 150, 180],
        [255, 110, 165],
        [255, 220, 125],
        [210, 70, 110],
    ],
    dtype=np.uint8,
)


class SharedState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.stop = False
        self.camera_ok = False
        self.model_ok = False
        self.camera_frames = 0
        self.bev_frames = 0
        self.last_error = ""
        self.last_frame: np.ndarray | None = None
        self.camera_jpg = placeholder_jpg("Waiting for camera", (1280, 720))
        self.bev_jpg = placeholder_jpg("Loading TIIM model", (720, 720))
        self.inference_fps = 0.0
        self.cuda_allocated_mib = 0.0
        self.cuda_reserved_mib = 0.0


def placeholder_jpg(text: str, size: tuple[int, int]) -> bytes:
    width, height = size
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = (13, 16, 19)
    cv2.putText(
        image,
        text,
        (32, max(48, height // 2)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (224, 230, 236),
        2,
        cv2.LINE_AA,
    )
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise RuntimeError("Could not encode placeholder")
    return encoded.tobytes()


def open_capture(source: str, width: int, height: int, fps: int) -> cv2.VideoCapture:
    if source.isdigit():
        source_obj: str | int = int(source)
        api = cv2.CAP_V4L2
    elif source.startswith("/dev/video"):
        source_obj = source
        api = cv2.CAP_V4L2
    else:
        source_obj = source
        api = cv2.CAP_ANY

    cap = cv2.VideoCapture(source_obj, api)
    if source.startswith("/dev/video") or source.isdigit():
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open camera source {source}")
    return cap


def camera_loop(state: SharedState, source: str, width: int, height: int, fps: int) -> None:
    cap: cv2.VideoCapture | None = None
    throttle_capture = source.startswith("/dev/video") or source.isdigit()
    period = 1.0 / max(fps, 1)
    next_tick = time.monotonic()
    while True:
        with state.lock:
            if state.stop:
                break

        if cap is None:
            try:
                cap = open_capture(source, width, height, fps)
                with state.lock:
                    state.camera_ok = True
                    state.last_error = ""
            except Exception as exc:
                with state.lock:
                    state.camera_ok = False
                    state.last_error = str(exc)
                time.sleep(1.0)
                continue

        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            cap = None
            with state.lock:
                state.camera_ok = False
                state.last_error = "Camera read failed"
            time.sleep(0.25)
            continue

        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if ok:
            with state.condition:
                state.last_frame = frame
                state.camera_jpg = encoded.tobytes()
                state.camera_frames += 1
                state.condition.notify_all()
        if throttle_capture:
            next_tick += period
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
        else:
            next_tick = time.monotonic()

    if cap is not None:
        cap.release()


def prepare_model(args: argparse.Namespace):
    config = load_config(config_path_for_checkpoint(args.checkpoint, args.config))
    model = build_model(config)
    load_checkpoint(model, args.checkpoint, args.allow_partial_checkpoint)

    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True
    model.to(device).eval()
    if args.fp16:
        model.half()

    desired_size = [int(v) for v in config["desired_image_size"]]
    z_range = [float(v) for v in config["z_intervals"]]
    grid_size = config.get("grid_size", (z_range[-1] - z_range[0], z_range[-1] - z_range[0]))
    tensor_dtype = torch.float16 if args.fp16 else torch.float32
    grid = utils.make_grid2d(grid_size, (-grid_size[0] / 2.0, 0.0), float(config["grid_res"]))
    grid = grid.unsqueeze(0).to(device=device, dtype=tensor_dtype)
    calib = load_intrinsics_from_file(args.intrinsics_file)
    class_names = list(config["pred_classes_nusc"])
    return model, config, device, desired_size, grid, calib, class_names, tensor_dtype


def frame_to_inputs(
    frame_bgr: np.ndarray,
    calib: np.ndarray,
    desired_size: list[int],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_like = __import__("PIL.Image").Image.fromarray(rgb)
    resized, scaled_calib = resize_image_and_calib(pil_like, calib, desired_size)
    image_tensor = image_to_tensor(resized).to(device=device, dtype=dtype)
    calib_tensor = torch.as_tensor(scaled_calib, dtype=dtype, device=device).unsqueeze(0)
    return image_tensor, calib_tensor


def logits_to_bev_jpg(logits: torch.Tensor, threshold: float, class_names: list[str]) -> bytes:
    probs = torch.sigmoid(logits).detach().cpu().float().numpy()
    conf = probs.max(axis=0)
    labels = probs.argmax(axis=0)

    color = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
    active = conf >= threshold
    palette = PALETTE[: len(class_names)]
    color[active] = palette[labels[active]]
    confidence = np.clip(conf[..., None] * 0.65 + 0.35, 0.0, 1.0)
    color = (color.astype(np.float32) * confidence).astype(np.uint8)

    scale = 6
    color = cv2.resize(color, (color.shape[1] * scale, color.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
    color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
    cv2.arrowedLine(color, (color.shape[1] // 2, color.shape[0] - 24), (color.shape[1] // 2, 34), (245, 245, 245), 2, cv2.LINE_AA, 0, 0.08)
    cv2.putText(color, "front", (color.shape[1] // 2 + 12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)

    legend_x = 12
    legend_y = 22
    for idx, name in enumerate(class_names[:8]):
        bgr = tuple(int(v) for v in palette[idx][::-1])
        cv2.rectangle(color, (legend_x, legend_y - 10), (legend_x + 10, legend_y), bgr, -1)
        cv2.putText(color, name, (legend_x + 16, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (230, 234, 238), 1, cv2.LINE_AA)
        legend_y += 17

    ok, encoded = cv2.imencode(".jpg", color, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError("Could not encode BEV frame")
    return encoded.tobytes()


def inference_loop(state: SharedState, args: argparse.Namespace) -> None:
    try:
        model, _config, device, desired_size, grid, calib, class_names, dtype = prepare_model(args)
        with state.lock:
            state.model_ok = True
            state.last_error = ""
    except Exception as exc:
        with state.lock:
            state.model_ok = False
            state.last_error = str(exc)
            state.bev_jpg = placeholder_jpg("Model load failed", (720, 720))
        return

    latest_seen = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    while True:
        with state.condition:
            state.condition.wait_for(
                lambda: state.stop or state.camera_frames != latest_seen,
                timeout=1.0,
            )
            if state.stop:
                break
            frame = None if state.last_frame is None else state.last_frame.copy()
            latest_seen = state.camera_frames

        if frame is None:
            continue

        try:
            started = time.perf_counter()
            with torch.inference_mode():
                image_tensor, calib_tensor = frame_to_inputs(frame, calib, desired_size, device, dtype)
                pred_ms = model(image_tensor, calib_tensor, grid)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
            elapsed = max(time.perf_counter() - started, 1e-6)
            bev_jpg = logits_to_bev_jpg(pred_ms[0][0], args.threshold, class_names)
            allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else 0.0
            reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2) if device.type == "cuda" else 0.0
            with state.lock:
                state.bev_jpg = bev_jpg
                state.bev_frames += 1
                state.inference_fps = 1.0 / elapsed
                state.cuda_allocated_mib = allocated
                state.cuda_reserved_mib = reserved
                state.last_error = ""
        except Exception as exc:
            with state.lock:
                state.last_error = str(exc)
            time.sleep(0.1)


class LiveHandler(BaseHTTPRequestHandler):
    state: SharedState

    def log_message(self, format, *args) -> None:
        return

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_bytes(HTTPStatus.OK, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/status":
            self.send_status()
            return
        if parsed.path == "/camera.mjpg":
            self.stream_frames("camera_jpg")
            return
        if parsed.path == "/bev.mjpg":
            self.stream_frames("bev_jpg")
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def send_bytes(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_status(self) -> None:
        with self.state.lock:
            payload = {
                "camera_ok": self.state.camera_ok,
                "model_ok": self.state.model_ok,
                "camera_frames": self.state.camera_frames,
                "bev_frames": self.state.bev_frames,
                "inference_fps": self.state.inference_fps,
                "cuda_allocated_mib": self.state.cuda_allocated_mib,
                "cuda_reserved_mib": self.state.cuda_reserved_mib,
                "last_error": self.state.last_error,
            }
        self.send_bytes(HTTPStatus.OK, json.dumps(payload).encode("utf-8"), "application/json")

    def stream_frames(self, attr: str) -> None:
        boundary = "frame"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={boundary}")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

        last_frame = None
        try:
            while True:
                with self.state.condition:
                    self.state.condition.wait(timeout=0.05)
                    jpg = getattr(self.state, attr)
                if jpg is last_frame:
                    time.sleep(0.02)
                    continue
                last_frame = jpg
                self.wfile.write(
                    f"--{boundary}\r\n"
                    "Content-Type: image/jpeg\r\n"
                    f"Content-Length: {len(jpg)}\r\n\r\n".encode("ascii")
                )
                self.wfile.write(jpg)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live TIIM camera-to-BEV browser viewer")
    parser.add_argument("--checkpoint", default="checkpoints_gdrive_fallback/checkpoint-0020.pth.gz")
    parser.add_argument("--config", default="checkpoints_gdrive_fallback/config.txt")
    parser.add_argument("--intrinsics-file", default="/home/caddy/PRODUCTION/calibration/cameras/REAL_front_intrinsics.json")
    parser.add_argument("--source", default="http://127.0.0.1:8787/camera.mjpg", help="Camera source: /dev/videoN, index, video file, or MJPEG URL.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--allow-partial-checkpoint", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(args.checkpoint)
    if args.config and not Path(args.config).exists():
        raise FileNotFoundError(args.config)
    if not Path(args.intrinsics_file).exists():
        raise FileNotFoundError(args.intrinsics_file)

    state = SharedState()
    LiveHandler.state = state

    camera_thread = threading.Thread(
        target=camera_loop,
        args=(state, args.source, args.width, args.height, args.fps),
        daemon=True,
    )
    model_thread = threading.Thread(target=inference_loop, args=(state, args), daemon=True)
    camera_thread.start()
    model_thread.start()

    server = ThreadingHTTPServer((args.host, args.port), LiveHandler)
    print(f"TIIM live viewer: http://{args.host}:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        with state.condition:
            state.stop = True
            state.condition.notify_all()
        server.server_close()


if __name__ == "__main__":
    main()
