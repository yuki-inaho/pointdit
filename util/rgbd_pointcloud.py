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

"""RGB-D back-projection and PLY export utilities.

The functions here turn a registered RGB image and a metric depth map into a
colored point cloud.  Depth is handled as a plain 2D array: float arrays are
assumed to be in meters, integer arrays are converted with ``depth_scale``
(for example ``0.001`` for a 16-bit PNG that stores millimeters).
"""

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import trimesh
import yaml


def intrinsics_matrix(fx, fy, cx, cy):
    """Build a 3x3 pinhole intrinsics matrix."""
    return np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
    )


def load_intrinsics(path):
    """Load a 3x3 intrinsics matrix from .npy, .json or .yaml.

    Accepted layouts: a 3x3 matrix, a flat list of 9 values, or a mapping with
    ``fx``/``fy``/``cx``/``cy`` keys.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        payload = np.load(path)
    elif suffix in (".json", ".yaml", ".yml"):
        text = path.read_text(encoding="utf-8")
        payload = yaml.safe_load(text) if suffix != ".json" else json.loads(text)
    else:
        raise ValueError(f"Unsupported intrinsics format: {path}")

    if isinstance(payload, dict):
        if "K" in payload:
            payload = payload["K"]
        else:
            try:
                return intrinsics_matrix(
                    payload["fx"], payload["fy"], payload["cx"], payload["cy"]
                )
            except KeyError as exc:
                raise ValueError(
                    f"Intrinsics mapping in {path} needs fx/fy/cx/cy or K"
                ) from exc

    matrix = np.asarray(payload, dtype=np.float64)
    if matrix.shape == (9,):
        matrix = matrix.reshape(3, 3)
    if matrix.shape != (3, 3):
        raise ValueError(f"Intrinsics in {path} must be 3x3, got {matrix.shape}")
    return matrix


def load_depth(path, depth_scale=1.0):
    """Load a depth map from .npy/.npz or an image file and convert to meters."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        depth = np.load(path)
    elif suffix == ".npz":
        with np.load(path) as archive:
            depth = archive[archive.files[0]]
    else:
        depth = imageio.imread(path)
    if depth.ndim == 3:
        depth = depth[..., 0]
    return depth.astype(np.float64) * float(depth_scale)


def backproject_depth(depth, intrinsics, depth_scale=1.0, min_depth=None, max_depth=None):
    """Back-project a depth map to camera coordinates.

    Returns ``(points, valid)`` where ``points`` is ``(H, W, 3)`` in the camera
    frame (x right, y down, z forward, OpenCV convention) and ``valid`` is a
    ``(H, W)`` boolean mask selecting usable pixels.  Invalid points are zeroed.
    """
    depth = np.asarray(depth)
    if depth.ndim != 2:
        raise ValueError(f"Depth must be 2D, got shape {depth.shape}")
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    if intrinsics.shape != (3, 3):
        raise ValueError(f"Intrinsics must be 3x3, got {intrinsics.shape}")

    z = depth.astype(np.float64) * float(depth_scale)
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    height, width = z.shape
    u, v = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    points = np.stack([x, y, z], axis=-1)

    valid = np.isfinite(z) & (z > 0.0)
    if min_depth is not None:
        valid &= z >= float(min_depth)
    if max_depth is not None:
        valid &= z <= float(max_depth)
    points = np.where(valid[..., None], points, 0.0)
    return points, valid


def rgbd_to_point_maps(rgb, depth, intrinsics, depth_scale=1.0, min_depth=None, max_depth=None):
    """Dense back-projection keeping the image layout.

    Returns ``(points, colors, valid)`` with shapes ``(H, W, 3)``, ``(H, W, 3)``
    and ``(H, W)``.  Colors are uint8, ready for the PLY helpers.
    """
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(f"RGB must be (H, W, 3), got shape {rgb.shape}")
    depth = np.asarray(depth)
    if depth.ndim != 2 or depth.shape != rgb.shape[:2]:
        raise ValueError(
            f"Depth shape {depth.shape} must match RGB shape {rgb.shape[:2]}"
        )

    points, valid = backproject_depth(
        depth, intrinsics, depth_scale=depth_scale, min_depth=min_depth, max_depth=max_depth
    )
    colors = colors_to_uint8(rgb[..., :3])
    return points, colors, valid


def rgbd_to_point_cloud(
    rgb,
    depth,
    intrinsics,
    depth_scale=1.0,
    min_depth=None,
    max_depth=None,
    mask=None,
    extrinsics=None,
):
    """Convert a registered RGB-D pair into a colored point cloud.

    Args:
        rgb: (H, W, 3) uint8 or float [0, 1] image.
        depth: (H, W) depth map; scaled by ``depth_scale`` into meters.
        intrinsics: 3x3 pinhole intrinsics.
        depth_scale: Factor applied to the raw depth values.
        min_depth/max_depth: Optional metric clipping range in meters.
        mask: Optional (H, W) array; pixels that are 0 are dropped.
        extrinsics: Optional (4, 4) or (3, 4) camera-to-world matrix.

    Returns:
        (points, colors) with shapes (N, 3) float64 and (N, 3) uint8.
    """
    points, colors, valid = rgbd_to_point_maps(
        rgb, depth, intrinsics, depth_scale=depth_scale, min_depth=min_depth, max_depth=max_depth
    )
    if mask is not None:
        mask = np.asarray(mask)
        if mask.shape != valid.shape:
            raise ValueError(f"Mask shape {mask.shape} must match depth shape {valid.shape}")
        valid &= mask.astype(bool)

    flat_points = points[valid]
    flat_colors = colors[valid]
    if extrinsics is not None:
        flat_points = transform_points(flat_points, extrinsics)
    return flat_points, flat_colors


def colors_to_uint8(colors):
    """Normalize colors to uint8; floats in [0, 1] are scaled by 255."""
    colors = np.asarray(colors)
    if colors.dtype == np.uint8:
        return colors
    if colors.dtype.kind == "f":
        scale = 255.0 if colors.size == 0 or colors.max() <= 1.0 + 1e-5 else 1.0
        return np.clip(np.round(colors * scale), 0.0, 255.0).astype(np.uint8)
    return np.clip(colors, 0, 255).astype(np.uint8)


def transform_points(points, extrinsics):
    """Apply a (4, 4) or (3, 4) camera-to-world matrix to (N, 3) points."""
    points = np.asarray(points, dtype=np.float64)
    extrinsics = np.asarray(extrinsics, dtype=np.float64)
    if extrinsics.shape == (3, 4):
        extrinsics = np.vstack([extrinsics, np.array([[0.0, 0.0, 0.0, 1.0]])])
    if extrinsics.shape != (4, 4):
        raise ValueError(f"Extrinsics must be 4x4 or 3x4, got {extrinsics.shape}")
    rotation = extrinsics[:3, :3]
    translation = extrinsics[:3, 3]
    return points @ rotation.T + translation


def voxel_downsample(points, colors, voxel_size):
    """Average points and colors inside a regular voxel grid."""
    if voxel_size <= 0.0:
        raise ValueError("voxel_size must be positive")
    points = np.asarray(points, dtype=np.float64)
    colors = colors_to_uint8(colors)
    if len(points) == 0:
        return points, colors

    keys = np.floor(points / float(voxel_size)).astype(np.int64)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    summed_points = np.zeros((len(counts), 3), dtype=np.float64)
    np.add.at(summed_points, inverse, points)
    mean_points = summed_points / counts[:, None]
    summed_colors = np.zeros((len(counts), 3), dtype=np.float64)
    np.add.at(summed_colors, inverse, colors.astype(np.float64))
    mean_colors = np.clip(np.round(summed_colors / counts[:, None]), 0, 255).astype(np.uint8)
    return mean_points, mean_colors


def save_ply(points, colors, filename, binary=True, transform_to_gl=False):
    """Write (N, 3) points with uint8 colors to a PLY file via trimesh.

    Args:
        transform_to_gl: Flip Y and Z so the cloud is Y-up/Z-out, matching
            ``util.viz_pointcloud.save_single_point_cloud``.
    """
    points = np.asarray(points, dtype=np.float64)
    colors = colors_to_uint8(colors)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Points must be (N, 3), got {points.shape}")
    if len(points) == 0:
        raise ValueError("Refusing to write an empty point cloud")
    if colors.shape != points.shape:
        raise ValueError(f"Colors shape {colors.shape} must match points {points.shape}")

    if transform_to_gl:
        points = points.copy()
        points[:, 1] = -points[:, 1]
        points[:, 2] = -points[:, 2]

    cloud = trimesh.PointCloud(points, colors=colors)
    cloud.export(filename, encoding="binary" if binary else "ascii")
