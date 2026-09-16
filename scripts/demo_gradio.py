# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Gradio demo for PointDiT: single image -> 3D point map.

Layout follows the PointDiT project page and the MoGe Gradio space:

- left: input image, examples, settings (sampling steps, downsample, colormap)
- right tabs: 3D view (gr.Model3D), depth map, step comparison (1/2/4 steps),
  measure (click two pixels for depth/distance) and download (.ply/.glb/.png)

Run (weights stay local; the DINOv3 encoder is gated and never redistributed):

    python scripts/demo_gradio.py \
        --checkpoint pretrained/pointditb-512-mixdata-nodinov3-1d42aacf.pth \
        --dinov3-weights-dir pretrained/dinov3 --device cuda --share

The demo fails closed with a clear message when the gated DINOv3 weights or the
model checkpoint are missing.
"""

import argparse
import copy
import os
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from denoiser import Denoiser  # noqa: E402
from main import get_args_parser  # noqa: E402
from util.paths import repo_path  # noqa: E402
from util.viz_depth import viz_depth_tensor  # noqa: E402
from util.viz_pointcloud import save_single_point_cloud  # noqa: E402

MODEL_CHOICES = ("PointDiT-B/16", "PointDiT-L/16", "PointDiT-H/16")
VIT_FOR_MODEL = {
    "PointDiT-B/16": "dinov3_vitb16",
    "PointDiT-L/16": "dinov3_vitl16",
    "PointDiT-H/16": "dinov3_vith16plus",
}
DINOV3_SHA = {
    "vits16": "08c60483",
    "vits16plus": "4057cbaa",
    "vitb16": "73cec8be",
    "vitl16": "8aa4cbdd",
    "vith16plus": "7c1da9a5",
    "vit7b16": "a955f4ea",
}
DEFAULT_CHECKPOINTS = {
    "PointDiT-B/16": "pretrained/pointditb-512-mixdata-nodinov3-1d42aacf.pth",
    "PointDiT-L/16": "pretrained/pointditl-512-mixdata-nodinov3-240c1a4f.pth",
    "PointDiT-H/16": "pretrained/pointdith-512-mixdata-nodinov3-cb01dd3b.pth",
}


def resolve_dinov3_weights(weights_dir: Path, vit_type: str) -> Path:
    path = weights_dir / f"dinov3_{vit_type}_pretrain_lvd1689m-{DINOV3_SHA[vit_type]}.pth"
    if not path.is_file():
        raise FileNotFoundError(
            f"Gated DINOv3 weights not found: {path}\n"
            "Request access at https://github.com/facebookresearch/dinov3 (or convert the "
            "gated Hugging Face safetensors with scripts/convert_dinov3_hf_weights.py) and "
            "place them under pretrained/dinov3/."
        )
    return path


def build_model(model_name: str, img_size: int, steps: int, checkpoint: Path, weights_dir: Path, device: str):
    parser = get_args_parser()
    args = parser.parse_args(
        [
            "--model", model_name,
            "--feature_embedding_type", VIT_FOR_MODEL[model_name],
            "--img_size", str(img_size),
            "--num_sampling_steps", str(steps),
            "--evaluate_gen",
            "--pretrained", str(checkpoint),
        ]
    )
    vit_type = args.feature_embedding_type.split("_")[-1]
    dinov3_path = resolve_dinov3_weights(weights_dir, vit_type)

    model = Denoiser(args)
    encoder_state = torch.load(dinov3_path, map_location="cpu", weights_only=True)
    model.net.y_embedder.load_state_dict(encoder_state, strict=True)
    for parameter in model.net.y_embedder.parameters():
        parameter.requires_grad = False

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    stripped = payload.get("dinov3_stripped")
    if stripped is None:
        model.load_state_dict(payload["model"], strict=True)
    else:
        incompatible = model.load_state_dict(payload["model"], strict=False)
        prefix = stripped.get("prefix", "net.y_embedder.")
        missing = [key for key in incompatible.missing_keys if not key.startswith(prefix)]
        if missing or incompatible.unexpected_keys:
            raise RuntimeError(
                f"checkpoint mismatch for {checkpoint}: missing={missing} "
                f"unexpected={incompatible.unexpected_keys}"
            )
    if "model_ema1" in payload:
        ema_state = payload["model_ema1"]
        model.ema_params1 = [
            ema_state[name].to(device) if name in ema_state else value.detach().clone()
            for name, value in model.named_parameters()
        ]
        model.ema_params2 = None
        # Hard-switch to the EMA weights once (engine does the same before evaluation).
        switched = copy.deepcopy(model.state_dict())
        for index, (name, _value) in enumerate(model.named_parameters()):
            switched[name] = model.ema_params1[index]
        model.load_state_dict(switched)
        model.ema_params1 = None
        model.ema_params2 = None
        print("Using EMA weights from the checkpoint")
    else:
        model.ema_params1 = None
        model.ema_params2 = None

    model.to(device).eval()
    return args, model


@torch.no_grad()
def predict_point_map(model: Denoiser, args, image_rgb: np.ndarray, steps: int, device: str):
    """Run one generation and return the point map at the input resolution."""
    model.steps = int(steps)
    patch_size = int(model.net.patch_size)
    height, width = image_rgb.shape[:2]
    tensor = torch.from_numpy(np.ascontiguousarray(image_rgb)).float().permute(2, 0, 1) / 255.0
    tensor = tensor.unsqueeze(0)

    target_tokens = (args.img_size // patch_size) ** 2
    factor = ((target_tokens * patch_size**2) / (height * width)) ** 0.5
    new_h = max(patch_size, int(round(height * factor / patch_size)) * patch_size)
    new_w = max(patch_size, int(round(width * factor / patch_size)) * patch_size)
    original_size = None
    if (new_h, new_w) != (height, width):
        original_size = (height, width)
        tensor = torch.nn.functional.interpolate(
            tensor, size=(new_h, new_w), mode="bilinear", align_corners=False
        )
    tensor = tensor.to(device)
    autocast = torch.amp.autocast("cuda", dtype=torch.bfloat16) if device.startswith("cuda") else torch.no_grad()
    with autocast:
        generated = model.generate(tensor, return_intermediate_steps=False)
    pointcloud = generated[0] if isinstance(generated, tuple) else generated
    pointcloud = pointcloud.float()
    if original_size is not None:
        pointcloud = torch.nn.functional.interpolate(
            pointcloud, size=original_size, mode="nearest"
        )
    return pointcloud[0].permute(1, 2, 0).cpu().numpy(), (new_h, new_w)


def export_results(image_rgb, points, out_dir: Path, voxel_size: float, colormap: str, remove_edges: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    image_float = image_rgb.astype(np.float32) / 255.0
    mask = None
    if remove_edges:
        from util.viz_pointcloud import depth_flying_points

        mask = ~depth_flying_points(points[..., 2].astype(np.float32), rtol=0.04)
    ply_path = out_dir / "pointcloud.ply"
    save_single_point_cloud(points, image_float, str(ply_path), transform_to_gl=True, mask=mask)

    depth = points[..., 2].astype(np.float32)
    depth_viz = viz_depth_tensor(
        torch.from_numpy(depth), shifted_depth=True, colormap=colormap, return_numpy=True
    )
    depth_path = out_dir / "depth.png"
    from imageio.v2 import imwrite

    imwrite(depth_path, np.asarray(depth_viz, dtype=np.uint8))

    glb_path = out_dir / "pointcloud.glb"
    try:
        import trimesh

        points_gl = points.copy()
        points_gl[..., 1:] *= -1.0  # match the PLY OpenGL-style flip
        cloud = trimesh.PointCloud(vertices=points_gl.reshape(-1, 3), colors=(image_rgb.reshape(-1, 3)))
        cloud.export(glb_path)
    except Exception as error:  # export is best effort; PLY is always written
        print(f"GLB export skipped: {error}")
        glb_path = None

    npy_path = out_dir / "points.npy"
    np.save(npy_path, points)
    if voxel_size and voxel_size > 0:
        from util.rgbd_pointcloud import voxel_downsample

        colors_u8 = image_rgb.reshape(-1, 3)
        pts, cols = voxel_downsample(
            points.reshape(-1, 3).astype(np.float64), colors_u8.astype(np.uint8), float(voxel_size)
        )
        downsampled = out_dir / "pointcloud_downsampled.ply"
        from util.rgbd_pointcloud import save_ply

        save_ply(pts, cols, str(downsampled), transform_to_gl=True)
    else:
        downsampled = None
    files = [str(p) for p in (ply_path, glb_path, depth_path, npy_path, downsampled) if p is not None]
    return files, depth_viz


def build_app(args, model_args, model, device):
    import gradio as gr

    checkpoint_name = Path(args.checkpoint).name
    session_root = repo_path("generation", "demo")

    def run_demo(image, steps, voxel_size, colormap, remove_edges):
        if image is None:
            raise gr.Error("Upload an image (or pick one of the examples) first.")
        started = time.perf_counter()
        points, resolution = predict_point_map(model, model_args, image, int(steps), device)
        out_dir = Path(session_root) / uuid.uuid4().hex[:12]
        files, depth_viz = export_results(
            image, points, out_dir, float(voxel_size or 0.0), colormap, bool(remove_edges)
        )
        info = (
            f"- checkpoint: `{checkpoint_name}`\n"
            f"- model input: {resolution[1]}x{resolution[0]} (token budget {model_args.img_size})\n"
            f"- sampling steps: {int(steps)}\n"
            f"- inference: {time.perf_counter() - started:.2f}s on {device}\n"
            f"- point map: {points.shape[1]}x{points.shape[0]} grid"
        )
        state = {
            "image": image,
            "points": points,
        }
        return (
            str(out_dir / "pointcloud.glb") if (out_dir / "pointcloud.glb").is_file() else str(out_dir / "pointcloud.ply"),
            np.asarray(depth_viz, dtype=np.uint8),
            files,
            info,
            state,
            image.copy(),
        )

    def run_steps_compare(image, colormap):
        if image is None:
            raise gr.Error("Upload an image first.")
        outputs = []
        for step_count in (1, 2, 4):
            points, _ = predict_point_map(model, model_args, image, step_count, device)
            out_dir = Path(session_root) / uuid.uuid4().hex[:12]
            files, depth_viz = export_results(image, points, out_dir, 0.0, colormap, False)
            glb = out_dir / "pointcloud.glb"
            outputs.append(str(glb) if glb.is_file() else str(out_dir / "pointcloud.ply"))
            outputs.append(np.asarray(depth_viz, dtype=np.uint8))
        return outputs

    def measure(state, event: gr.SelectData, measure_points):
        if not state:
            return None, [], ""
        image = state["image"].copy()
        points = state["points"]
        x, y = event.index[0], event.index[1]
        measure_points = list(measure_points or [])
        if len(measure_points) >= 4:
            measure_points = []
        measure_points.extend([int(x), int(y)])
        import cv2

        pairs = [(measure_points[i], measure_points[i + 1]) for i in range(0, len(measure_points), 2)]
        text = ""
        for index, (px, py) in enumerate(pairs):
            depth_value = float(points[py, px, 2])
            text += f"- P{index + 1} ({px},{py}) depth: {depth_value:.3f}\n"
            image = cv2.circle(image, (px, py), 5, (255, 0, 0), 2)
        if len(pairs) == 2:
            first, second = pairs
            image = cv2.line(image, first, second, (255, 0, 0), 2)
            distance = float(
                np.linalg.norm(points[first[1], first[0]] - points[second[1], second[0]])
            )
            text += f"- distance: {distance:.3f}\n"
            measure_points = []
        return image, measure_points, text

    with gr.Blocks(title="PointDiT demo") as demo:
        gr.Markdown(
            "# PointDiT: single image → 3D point map\n"
            "Pixel-space diffusion, one plain ViT. Drag / scroll / right-drag in the 3D view."
        )
        state = gr.State(None)
        measure_points = gr.State([])
        with gr.Row():
            with gr.Column(scale=4):
                input_image = gr.Image(type="numpy", image_mode="RGB", label="Input image")
                examples_dir = repo_path("assets", "demo")
                example_paths = sorted(str(p) for p in Path(examples_dir).glob("*.[jp][pn]*g"))
                if example_paths:
                    gr.Examples(examples=example_paths, inputs=input_image, label="Examples")
                with gr.Accordion("Settings", open=True):
                    steps = gr.Slider(1, 8, value=int(args.num_sampling_steps), step=1, label="Sampling steps")
                    voxel = gr.Number(value=0.005, label="Voxel downsample (m, 0 = off)", precision=3)
                    colormap = gr.Dropdown(["plasma", "viridis", "magma", "turbo"], value="plasma", label="Depth colormap")
                    remove_edges = gr.Checkbox(value=False, label="Remove depth edges (flying points)")
                with gr.Row():
                    submit = gr.Button("Submit", variant="primary")
                    clear = gr.Button("Clear")
                info = gr.Markdown("")
            with gr.Column(scale=6):
                with gr.Tabs():
                    with gr.Tab("3D View"):
                        model_3d = gr.Model3D(display_mode="solid", clear_color=[1.0, 1.0, 1.0, 1.0], label="Point map", height="60vh")
                        gr.Markdown("Drag to orbit · scroll to zoom · right-drag to pan.")
                    with gr.Tab("Depth"):
                        depth_image = gr.Image(type="numpy", label="Colorized depth (shifted)", format="png", interactive=False)
                    with gr.Tab("Steps comparison"):
                        gr.Markdown("Same image, 1 / 2 / 4 sampling steps.")
                        with gr.Row():
                            step_views = [gr.Model3D(display_mode="solid", clear_color=[1.0, 1.0, 1.0, 1.0], label=f"{n} step(s)", height="45vh") for n in (1, 2, 4)]
                        with gr.Row():
                            step_depths = [gr.Image(type="numpy", label=f"{n} step depth", format="png", interactive=False) for n in (1, 2, 4)]
                        compare = gr.Button("Run comparison")
                    with gr.Tab("Measure"):
                        gr.Markdown("Click two pixels: the point map gives each depth and the distance.")
                        measure_image = gr.Image(type="numpy", label="Click on the image", format="webp", interactive=False, sources=[])
                        measure_text = gr.Markdown("")
                    with gr.Tab("Download"):
                        files = gr.File(type="filepath", label="Output files (.ply/.glb/depth.png/points.npy)")

        submit.click(
            fn=lambda: (None, None, [], "", None, None),
            outputs=[model_3d, depth_image, files, info, state, measure_image],
        ).then(
            fn=run_demo,
            inputs=[input_image, steps, voxel, colormap, remove_edges],
            outputs=[model_3d, depth_image, files, info, state, measure_image],
        )
        compare.click(
            fn=run_steps_compare,
            inputs=[input_image, colormap],
            outputs=[*(v for pair in zip(step_views, step_depths) for v in pair)],
        )
        measure_image.select(
            fn=measure,
            inputs=[state, measure_points],
            outputs=[measure_image, measure_points, measure_text],
        )
        clear.click(
            fn=lambda: (None, None, None, [], "", None, [], ""),
            outputs=[input_image, model_3d, depth_image, files, info, state, measure_points, measure_text],
        )
    return demo


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="PointDiT-B/16", choices=MODEL_CHOICES)
    parser.add_argument("--checkpoint", default=None, help="Path to a PointDiT checkpoint")
    parser.add_argument("--dinov3-weights-dir", default=str(repo_path("pretrained", "dinov3")))
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--num-sampling-steps", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    import gradio as gr

    args = parse_args(argv)
    checkpoint = Path(args.checkpoint) if args.checkpoint else Path(DEFAULT_CHECKPOINTS[args.model])
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"PointDiT checkpoint not found: {checkpoint}\n"
            "Download it from https://huggingface.co/haofeixu/pointdit (see MODELS.md)."
        )
    print(f"Loading {args.model} from {checkpoint} on {args.device} ...")
    model_args, model = build_model(
        args.model,
        args.img_size,
        args.num_sampling_steps,
        checkpoint,
        Path(args.dinov3_weights_dir),
        args.device,
    )
    app = build_app(args, model_args, model, args.device)
    app.queue().launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
