"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""
from __future__ import annotations

import os
import json
import torch
import struct
import open3d as o3d

import numpy as np
import imageio.v2 as imageio
import torch.nn.functional as F

from dataclasses import dataclass, field
from typing import Type, Literal
from pathlib import Path

from gnerf.datasets.utils import Rays
from gnerf.configs.base_configs import BaseDatasetConfig, BaseDataset


def read_cameras_binary(path):
    with open(path, "rb") as f:
        num_cameras = struct.unpack("<Q", f.read(8))[0]
        cameras = {}
        for _ in range(num_cameras):
            camera_id = struct.unpack("<I", f.read(4))[0]
            model_id = struct.unpack("<i", f.read(4))[0]
            width = struct.unpack("<Q", f.read(8))[0]
            height = struct.unpack("<Q", f.read(8))[0]
            params_num = {0: 4, 1: 4, 2: 4, 3: 5, 4: 8}.get(model_id, 4)
            params = struct.unpack("<" + "d" * params_num, f.read(8 * params_num))
            cameras[camera_id] = {
                "model_id": model_id,
                "width": width,
                "height": height,
                "params": params,
            }
    return cameras


def read_images_binary(path):
    with open(path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        images = {}
        for _ in range(num_images):
            image_id = struct.unpack("<I", f.read(4))[0]
            qw, qx, qy, qz = struct.unpack("<dddd", f.read(8*4))
            tx, ty, tz = struct.unpack("<ddd", f.read(8*3))
            camera_id = struct.unpack("<I", f.read(4))[0]

            # Read null-terminated string safely
            name_bytes = bytearray()
            while True:
                byte = f.read(1)
                if byte == b'':
                    raise EOFError("Unexpected end of file while reading image name")
                if byte == b'\x00':
                    break
                name_bytes += byte
            try:
                name = name_bytes.decode('utf-8')
            except UnicodeDecodeError:
                print("Warning: failed to decode image name, skipping.")
                continue

            images[image_id] = {
                "qw": qw, "qx": qx, "qy": qy, "qz": qz,
                "tx": tx, "ty": ty, "tz": tz,
                "camera_id": camera_id,
                "name": name,
            }

            # skip 2D points (track) for now — they follow here, but aren't needed
            num_points2D = struct.unpack("<Q", f.read(8))[0]
            f.read(num_points2D * 3 * 8)  # x, y, point3D_id (each double)

    return images


def qvec2rotmat(qw, qx, qy, qz):
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    n = np.dot(q, q)
    if n < np.finfo(float).eps:
        return np.eye(3)
    q *= np.sqrt(2.0 / n)
    q = np.outer(q, q)
    rot = np.array([
        [1.0 - q[2, 2] - q[3, 3],       q[1, 2] - q[3, 0],       q[1, 3] + q[2, 0]],
        [      q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3],       q[2, 3] - q[1, 0]],
        [      q[1, 3] - q[2, 0],       q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2]],
    ])
    return rot


def cameras_to_json(sparse_folder):
    cameras = read_cameras_binary(os.path.join(sparse_folder, "cameras.bin"))
    images = read_images_binary(os.path.join(sparse_folder, "images.bin"))

    frames = []
    for image_id in sorted(images.keys()):
        img = images[image_id]
        cam = cameras[img["camera_id"]]

        R = qvec2rotmat(img["qw"], img["qx"], img["qy"], img["qz"])
        t = np.array([img["tx"], img["ty"], img["tz"]]).reshape(3,1)

        # Compose 4x4 transform matrix (camera-to-world)
        transform = np.eye(4)
        transform[:3, :3] = R
        transform[:3, 3:] = t

        # Extract intrinsics assuming PINHOLE camera model (model_id==1)
        if cam["model_id"] == 1:
            fx, fy, cx, cy = cam["params"][:4]
            intrinsics = {"fx": fx, "fy": fy, "cx": cx, "cy": cy}
        else:
            intrinsics = {}

        frame = {
            "file_path": img["name"],
            "transform_matrix": transform.tolist(),
            "intrinsics": intrinsics
        }
        frames.append(frame)

    return {"frames": frames}


def read_points3D_binary(path):
    """Load points3D.bin from COLMAP."""
    with open(path, "rb") as f:
        num_points = struct.unpack("<Q", f.read(8))[0]
        points = []
        for _ in range(num_points):
            point_id = struct.unpack("<Q", f.read(8))[0]
            x, y, z = struct.unpack("<ddd", f.read(8*3))
            r, g, b = struct.unpack("<BBB", f.read(3))
            error = struct.unpack("<d", f.read(8))[0]
            track_length = struct.unpack("<Q", f.read(8))[0]
            track = []
            for _ in range(track_length):
                image_id = struct.unpack("<I", f.read(4))[0]
                point2d_idx = struct.unpack("<I", f.read(4))[0]
                track.append((image_id, point2d_idx))

            points.append({
                "id": point_id,
                "xyz": (x, y, z),
                "rgb": (r, g, b),
                "error": error,
                "track": track
            })
    return points


def normalize_points(points):
    """
    Normalize points to fit inside unit sphere.
    Returns:
        - normalized points
        - translation vector
        - scale factor
    """
    xyz = np.array(points)
    centroid = np.mean(xyz, axis=0)
    xyz_centered = xyz - centroid

    scale = np.max(np.linalg.norm(xyz_centered, axis=1))
    xyz_normalized = xyz_centered / scale

    return xyz_normalized, centroid, scale


def _load_renderings(root_fp: Path, scene: str):
    """Load images and camera data, and normalize the scene."""

    data_dir = root_fp / scene
    sparse_dir = data_dir / "sparse/0"

    # 1. Load camera data from COLMAP and images
    meta = cameras_to_json(sparse_dir)

    images = []
    camtoworlds = []
    focals = []
    centers = []
    scale_down_factor = 4  # images_4 are 4x smaller than original

    for frame in meta["frames"]:
        fname = os.path.join(data_dir, "images_4", frame["file_path"])
        rgba = imageio.imread(fname)

        camtoworlds.append(np.array(frame["transform_matrix"]))
        images.append(rgba)

        intr = frame["intrinsics"]
        if "fx" in intr and "fy" in intr:
            focals.append((intr["fx"] / scale_down_factor, intr["fy"] / scale_down_factor))
        else:
            focals.append(None)

        if "cx" in intr and "cy" in intr:
            centers.append(np.array([intr["cx"] / scale_down_factor, intr["cy"] / scale_down_factor]))
        else:
            centers.append(None)

    images = np.stack(images, axis=0)
    camtoworlds = np.stack(camtoworlds, axis=0)

    # 2. Normalize using COLMAP point cloud
    points = read_points3D_binary(sparse_dir / "points3D.bin")
    xyz = np.array([pt['xyz'] for pt in points])

    # centroid = xyz.mean(axis=0)
    # max_dist = np.max(np.linalg.norm(xyz - centroid, axis=1))

    # camtoworlds[:, :3, 3] -= centroid
    camtoworlds[:, :3, 3] -= np.array([0.0, -1.5, 3.0])  # Centering at origin
    camtoworlds[:, :3, 3] /= 1.0

    # 3. Normalize intrinsics as usual
    h, w = images.shape[1:3]
    focal = focals[0] if all(f == focals[0] for f in focals) else np.array(focals)
    center = centers[0] if all(c is not None and np.array_equal(c, centers[0]) for c in centers) else np.array(centers)

    return images, camtoworlds, h, w, focal, center


@dataclass
class MipNeRFDatasetConfig(BaseDatasetConfig):
    """Configuration for the NeRFSyntheticDataset."""

    _target: Type = field(default_factory=lambda: MipNeRFDataset)
    """Target class for the NeRFSyntheticDataset."""
    data_root: Path = Path("data/mip_nerf_360_dataset")
    """Path to the dataset."""
    scene: Literal["bicycle", "bonsai", "counter", "flowers", "garden", "kitchen", "room", "stump", "treehill"] = "garden"
    """Scene name."""
    opengl_camera: bool = True
    """Use OpenGL camera convention."""
    width: int = 800
    """Width of the images."""
    height: int = 800
    """Height of the images."""


class MipNeRFDataset(BaseDataset):
    """Dataset for NeRF synthetic scenes."""

    def __init__(self, 
                 config: MipNeRFDatasetConfig,
                 split: Literal["train", "val", "trainval"] = "train",
                 num_rays: int = None,
                 batch_over_images: bool = True,
                 device: torch.device = torch.device("cpu")
                ):
        super().__init__(config)

        self.num_rays = num_rays
        self.training = (num_rays is not None) and (
            split in ["train", "trainval"]
        )

        self.batch_over_images = batch_over_images
        if split == "trainval":
            _images_train, _camtoworlds_train, self.config.height, self.config.width, _focal_train, _centers_train = _load_renderings(
                self.config.data_root, self.config.scene, "train"
            )
            _images_val, _camtoworlds_val, self.config.height, self.config.width, _focal_val, _centers_val = _load_renderings(
                self.config.data_root, self.config.scene, "val"
            )
            self.images = np.concatenate([_images_train, _images_val])
            self.camtoworlds = np.concatenate(
                [_camtoworlds_train, _camtoworlds_val]
            )
            self.focal = _focal_train
            self.centers = _centers_train
        else:
            self.images, self.camtoworlds, self.config.height, self.config.width, self.focal, self.centers = _load_renderings(
                self.config.data_root, self.config.scene
            )

        self.images = torch.from_numpy(self.images).to(torch.uint8)
        self.camtoworlds = torch.from_numpy(self.camtoworlds).to(torch.float32)
        self.K = torch.tensor(
            [
                [self.focal[0], 0, self.centers[0]],
                [0, self.focal[1], self.centers[1]],
                [0, 0, 1],
            ],
            dtype=torch.float32,
        )  # (3, 3)

        self.images = self.images.to(device)
        self.camtoworlds = self.camtoworlds.to(device)
        self.K = self.K.to(device)
        assert self.images.shape[1:3] == (self.config.height, self.config.width)
        self.g = torch.Generator(device=device)
        self.g.manual_seed(42)

    def __len__(self):
        return len(self.images)

    @torch.no_grad()
    def __getitem__(self, index):
        data = self.fetch_data(index)
        data = self.preprocess(data)
        return data
    
    def get_points(self) -> np.ndarray:

        ply_path = f"{self.config.data_root}/{self.config.scene}/sparse/0/points3D.bin"
        pts = read_points3D_binary(ply_path)
        pts = np.array([p["xyz"] for p in pts], dtype=np.float64)

        pts, self.centroid, self.scale = normalize_points(pts)
        
        return None
    
    def get_points_eval(self, path: Path) -> np.ndarray:
        
        # Load .ply file and initialize means
        ply_files = list(Path(path).glob("*.ply"))
        if not ply_files:
            print(f"No .ply files found in {path}.")
            return None
        ply_path = str(ply_files[0])
        try:
            pcd = o3d.io.read_point_cloud(ply_path)
            pts = np.asarray(pcd.points)

            print(f"Loaded pointclud with {pts.shape} points.")
        except:
            print(f"Failed to load point cloud from {ply_path}.")
            return None

        return pts

    def preprocess(self, data):
        """Process the fetched / cached data with randomness."""
        rgba, rays = data["rgba"], data["rays"]
        pixels, alpha = torch.split(rgba, [3, 1], dim=-1)

        return {
            "pixels": pixels,  # [n_rays, 3] or [h, w, 3]
            "rays": rays,  # [n_rays,] or [h, w]
            "color_bkgd": torch.zeros(3, device=self.images.device),  # [3,]
            **{k: v for k, v in data.items() if k not in ["rgba", "rays"]},
        }

    def update_num_rays(self, num_rays):
        self.num_rays = num_rays

    def fetch_data(self, index):
        """Fetch the data (it maybe cached for multiple batches)."""
        num_rays = self.num_rays

        if self.training:
            if self.batch_over_images:
                image_id = torch.randint(
                    0,
                    len(self.images),
                    size=(num_rays,),
                    device=self.images.device,
                    generator=self.g,
                )
            else:
                image_id = [index] * num_rays
            x = torch.randint(
                0,
                self.config.width,
                size=(num_rays,),
                device=self.images.device,
                generator=self.g,
            )
            y = torch.randint(
                0,
                self.config.height,
                size=(num_rays,),
                device=self.images.device,
                generator=self.g,
            )
        else:
            image_id = [index]
            x, y = torch.meshgrid(
                torch.arange(self.config.width, device=self.images.device),
                torch.arange(self.config.height, device=self.images.device),
                indexing="xy",
            )
            x = x.flatten()
            y = y.flatten()

        # generate rays
        rgb = self.images[image_id, y, x] / 255.0  # (num_rays, 3)
        if rgb.shape[-1] == 3:
            alpha = torch.ones_like(rgb[..., :1])
            rgba = torch.cat([rgb, alpha], dim=-1)
        else:
            rgba = rgb  # already has alpha

        c2w = self.camtoworlds[image_id]  # (num_rays, 3, 4)
        camera_dirs = F.pad(
            torch.stack(
                [
                    (x - self.K[0, 2] + 0.5) / self.K[0, 0],
                    (y - self.K[1, 2] + 0.5)
                    / self.K[1, 1]
                    * (-1.0 if self.config.opengl_camera else 1.0),
                ],
                dim=-1,
            ),
            (0, 1),
            value=(-1.0 if self.config.opengl_camera else 1.0),
        )  # [num_rays, 3]

        # [n_cams, height, width, 3]
        directions = (camera_dirs[:, None, :] * c2w[:, :3, :3]).sum(dim=-1)
        origins = torch.broadcast_to(c2w[:, :3, -1], directions.shape)
        viewdirs = directions / torch.linalg.norm(
            directions, dim=-1, keepdims=True
        )

        if self.training:
            origins = torch.reshape(origins, (num_rays, 3))
            viewdirs = torch.reshape(viewdirs, (num_rays, 3))
            rgba = torch.reshape(rgba, (num_rays, 4))
        else:
            origins = torch.reshape(origins, (self.config.height, self.config.width, 3))
            viewdirs = torch.reshape(viewdirs, (self.config.height, self.config.width, 3))
            rgba = torch.reshape(rgba, (self.config.height, self.config.width, 4))

        rays = Rays(origins=origins, viewdirs=viewdirs)

        return {
            "rgba": rgba,  # [h, w, 4] or [num_rays, 4]
            "rays": rays,  # [h, w, 3] or [num_rays, 3]
        }
    
    def get_weight_decay(self) -> float:
        """Get the weight decay for the dataset."""
        return 0.0