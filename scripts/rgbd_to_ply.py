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

"""Create a colored PLY point cloud from a registered RGB-D pair.

Example:
    python scripts/rgbd_to_ply.py \
        --rgb rgb.png --depth depth.png --intrinsics intrinsics.json \
        --depth-scale 0.001 --min-depth 0.2 --max-depth 3.0 \
        --output cloud.ply
"""

import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from util.rgbd_pointcloud import (  # noqa: E402
    intrinsics_matrix,
    load_depth,
    load_intrinsics,
    rgbd_to_point_cloud,
    save_ply,
    voxel_downsample,
)


def load_rgb(path):
    rgb = np.asarray(imageio.imread(path))
    if rgb.ndim == 2:
        rgb = np.stack([rgb] * 3, axis=-1)
    if rgb.shape[2] == 4:
        rgb = rgb[..., :3]
    if rgb.shape[2] != 3:
        raise ValueError(f"RGB image {path} must have 3 channels, got {rgb.shape}")
    return rgb


def load_mask(path):
    mask = np.asarray(imageio.imread(path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask > 0


def load_extrinsics(path):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return np.load(path)
    return np.asarray(json.loads(path.read_text(encoding="utf-8")), dtype=np.float64)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb", type=Path, required=True, help="RGB image (png/jpg)")
    parser.add_argument("--depth", type=Path, required=True, help="Depth map (.npy, .npz or 16-bit png)")
    parser.add_argument("--output", type=Path, required=True, help="Destination .ply file")
    parser.add_argument("--intrinsics", type=Path, help="3x3 K in .npy/.json/.yaml, or fx/fy/cx/cy mapping")
    parser.add_argument("--fx", type=float, help="Focal length x (pixels)")
    parser.add_argument("--fy", type=float, help="Focal length y (pixels)")
    parser.add_argument("--cx", type=float, help="Principal point x (pixels)")
    parser.add_argument("--cy", type=float, help="Principal point y (pixels)")
    parser.add_argument("--depth-scale", type=float, default=1.0, help="Factor turning raw depth into meters")
    parser.add_argument("--min-depth", type=float, help="Drop points closer than this (meters)")
    parser.add_argument("--max-depth", type=float, help="Drop points farther than this (meters)")
    parser.add_argument("--mask", type=Path, help="Optional mask image; zero pixels are dropped")
    parser.add_argument("--extrinsics", type=Path, help="Camera-to-world matrix (.npy/.json, 4x4 or 3x4)")
    parser.add_argument("--voxel-size", type=float, help="Optional voxel downsampling size (meters)")
    parser.add_argument("--flip-yz", action="store_true", help="Flip Y/Z for Y-up viewers")
    parser.add_argument("--ascii", action="store_true", help="Write an ASCII PLY instead of binary")
    return parser.parse_args(argv)


def resolve_intrinsics(args):
    if args.intrinsics is not None:
        return load_intrinsics(args.intrinsics)
    values = (args.fx, args.fy, args.cx, args.cy)
    if any(value is None for value in values):
        raise SystemExit("Provide --intrinsics or all of --fx/--fy/--cx/--cy")
    return intrinsics_matrix(*values)


def main(argv=None):
    args = parse_args(argv)
    intrinsics = resolve_intrinsics(args)
    rgb = load_rgb(args.rgb)
    depth = load_depth(args.depth, depth_scale=args.depth_scale)
    mask = load_mask(args.mask) if args.mask is not None else None
    extrinsics = load_extrinsics(args.extrinsics) if args.extrinsics is not None else None

    points, colors = rgbd_to_point_cloud(
        rgb,
        depth,
        intrinsics,
        depth_scale=1.0,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        mask=mask,
        extrinsics=extrinsics,
    )
    if args.voxel_size is not None:
        points, colors = voxel_downsample(points, colors, args.voxel_size)
    if len(points) == 0:
        raise SystemExit("No valid points after filtering; check depth range, scale and mask")

    save_ply(points, colors, args.output, binary=not args.ascii, transform_to_gl=args.flip_yz)
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    print(f"Wrote {len(points)} points to {args.output}")
    print(f"Bounds min={np.round(lower, 4)} max={np.round(upper, 4)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
