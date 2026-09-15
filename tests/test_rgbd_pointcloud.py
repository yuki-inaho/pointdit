import json

import imageio.v2 as imageio
import numpy as np
import pytest
import trimesh

from scripts import rgbd_to_ply
from util.rgbd_pointcloud import (
    backproject_depth,
    colors_to_uint8,
    intrinsics_matrix,
    load_intrinsics,
    rgbd_to_point_cloud,
    save_ply,
    transform_points,
    voxel_downsample,
)

K = intrinsics_matrix(fx=100.0, fy=100.0, cx=1.0, cy=1.0)


def test_backproject_center_pixel():
    depth = np.full((3, 3), 2.0)
    points, valid = backproject_depth(depth, K)
    assert valid.all()
    assert np.allclose(points[1, 1], [0.0, 0.0, 2.0])
    assert np.allclose(points[0, 0], [-0.02, -0.02, 2.0])
    assert np.allclose(points[2, 2], [0.02, 0.02, 2.0])


def test_invalid_depth_is_masked():
    depth = np.array([[0.0, np.nan], [1.5, -1.0]])
    points, valid = backproject_depth(depth, intrinsics_matrix(1.0, 1.0, 0.0, 0.0))
    assert valid.tolist() == [[False, False], [True, False]]
    assert np.allclose(points[0], 0.0)


def test_depth_range_and_scale_filters():
    depth_mm = np.array([[500, 1500], [2500, 4000]], dtype=np.uint16)
    points, valid = backproject_depth(
        depth_mm, intrinsics_matrix(1.0, 1.0, 0.0, 0.0), depth_scale=0.001, min_depth=1.0, max_depth=3.0
    )
    assert valid.tolist() == [[False, True], [True, False]]
    assert points[0, 1, 2] == pytest.approx(1.5)


def test_rgbd_to_point_cloud_applies_mask_and_extrinsics():
    rgb = np.zeros((2, 2, 3), dtype=np.float32)
    rgb[..., 0] = 1.0
    rgb[1, 1, 1] = 0.5
    depth = np.ones((2, 2), dtype=np.float32)
    mask = np.array([[False, False], [False, True]])
    extrinsics = np.eye(4)
    extrinsics[:3, 3] = [1.0, 2.0, 3.0]

    points, colors = rgbd_to_point_cloud(
        rgb, depth, intrinsics_matrix(1.0, 1.0, 0.0, 0.0), mask=mask, extrinsics=extrinsics
    )
    assert points.shape == (1, 3)
    assert np.allclose(points[0], [2.0, 3.0, 4.0])
    assert colors.dtype == np.uint8
    assert np.allclose(colors[0], [255, 128, 0], atol=1)


def test_colors_to_uint8_variants():
    assert colors_to_uint8(np.array([[1.0, 0.0, 0.5]])).tolist() == [[255, 0, 128]]
    assert colors_to_uint8(np.array([[255, 300, -4]])).tolist() == [[255, 255, 0]]
    assert colors_to_uint8(np.zeros((1, 3), dtype=np.uint8)).dtype == np.uint8


def test_transform_points_accepts_3x4():
    points = np.array([[1.0, 0.0, 0.0]])
    extrinsics = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 2.0]])
    assert np.allclose(transform_points(points, extrinsics), [[1.0, 0.0, 2.0]])


def test_voxel_downsample_averages():
    points = np.array([[0.0, 0.0, 0.0], [0.01, 0.01, 0.01], [0.5, 0.5, 0.5]])
    colors = np.array([[0, 0, 0], [10, 20, 30], [255, 255, 255]], dtype=np.uint8)
    mean_points, mean_colors = voxel_downsample(points, colors, voxel_size=0.1)
    assert len(mean_points) == 2
    assert np.allclose(mean_points[0], [0.005, 0.005, 0.005])
    assert mean_colors[0].tolist() == [5, 10, 15]


def test_save_ply_roundtrip(tmp_path):
    points = np.array([[0.0, 0.0, 1.0], [0.1, 0.2, 1.3], [-0.2, 0.4, 0.9]])
    colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
    binary_path = tmp_path / "cloud.ply"
    ascii_path = tmp_path / "cloud_ascii.ply"

    save_ply(points, colors, binary_path)
    save_ply(points, colors, ascii_path, binary=False)

    for path in (binary_path, ascii_path):
        cloud = trimesh.load(path)
        assert len(cloud.vertices) == 3
        assert np.allclose(np.asarray(cloud.vertices), points, atol=1e-6)
        assert np.allclose(np.asarray(cloud.colors)[:, :3], colors, atol=1)


def test_save_ply_rejects_empty(tmp_path):
    with pytest.raises(ValueError):
        save_ply(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8), tmp_path / "empty.ply")


def test_load_intrinsics_json_variants(tmp_path):
    matrix_path = tmp_path / "K.json"
    matrix_path.write_text(json.dumps([[100.0, 0.0, 1.0], [0.0, 100.0, 1.0], [0.0, 0.0, 1.0]]))
    assert np.allclose(load_intrinsics(matrix_path), K)

    params_path = tmp_path / "params.json"
    params_path.write_text(json.dumps({"fx": 100.0, "fy": 100.0, "cx": 1.0, "cy": 1.0}))
    assert np.allclose(load_intrinsics(params_path), K)


def test_cli_end_to_end(tmp_path):
    depth = np.full((2, 2), 1000, dtype=np.uint16)
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    rgb[..., 1] = 200
    rgb_path = tmp_path / "rgb.png"
    depth_path = tmp_path / "depth.png"
    output_path = tmp_path / "cloud.ply"
    imageio.imwrite(rgb_path, rgb)
    imageio.imwrite(depth_path, depth)

    exit_code = rgbd_to_ply.main(
        [
            "--rgb", str(rgb_path),
            "--depth", str(depth_path),
            "--fx", "100",
            "--fy", "100",
            "--cx", "0",
            "--cy", "0",
            "--depth-scale", "0.001",
            "--min-depth", "0.5",
            "--max-depth", "1.5",
            "--voxel-size", "0.005",
            "--flip-yz",
            "--output", str(output_path),
        ]
    )
    assert exit_code == 0
    cloud = trimesh.load(output_path)
    assert len(cloud.vertices) == 4
    assert np.allclose(np.asarray(cloud.vertices)[:, 2], -1.0, atol=1e-6)
