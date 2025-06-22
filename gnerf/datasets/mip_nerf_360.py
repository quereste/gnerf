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
import scipy.spatial.transform.rotation as rot

from dataclasses import dataclass, field
from typing import Type, Literal
from pathlib import Path

from gnerf.datasets.utils import Rays
from gnerf.configs.base_configs import BaseDatasetConfig, BaseDataset

from typing import Dict, Optional, Any
from enum import Enum
import collections


class CameraModel(Enum):
    """Enum for camera types."""

    OPENCV = "OPENCV"
    OPENCV_FISHEYE = "OPENCV_FISHEYE"
    EQUIRECTANGULAR = "EQUIRECTANGULAR"
    PINHOLE = "PINHOLE"
    SIMPLE_PINHOLE = "SIMPLE_PINHOLE"

def parse_colmap_camera_params(camera) -> Dict[str, Any]:
    """
    Parses all currently supported COLMAP cameras into the transforms.json metadata

    Args:
        camera: COLMAP camera
    Returns:
        transforms.json metadata containing camera's intrinsics and distortion parameters

    """
    out: Dict[str, Any] = {
        "w": camera.width,
        "h": camera.height,
    }

    # Parameters match https://github.com/colmap/colmap/blob/dev/src/base/camera_models.h
    camera_params = camera.params
    if camera.model == "SIMPLE_PINHOLE":
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[0])
        out["cx"] = float(camera_params[1])
        out["cy"] = float(camera_params[2])
        out["k1"] = 0.0
        out["k2"] = 0.0
        out["p1"] = 0.0
        out["p2"] = 0.0
        camera_model = CameraModel.OPENCV
    elif camera.model == "PINHOLE":
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[1])
        out["cx"] = float(camera_params[2])
        out["cy"] = float(camera_params[3])
        out["k1"] = 0.0
        out["k2"] = 0.0
        out["p1"] = 0.0
        out["p2"] = 0.0
        camera_model = CameraModel.OPENCV
    elif camera.model == "SIMPLE_RADIAL":
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[0])
        out["cx"] = float(camera_params[1])
        out["cy"] = float(camera_params[2])
        out["k1"] = float(camera_params[3])
        out["k2"] = 0.0
        out["p1"] = 0.0
        out["p2"] = 0.0
        camera_model = CameraModel.OPENCV
    elif camera.model == "RADIAL":
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[0])
        out["cx"] = float(camera_params[1])
        out["cy"] = float(camera_params[2])
        out["k1"] = float(camera_params[3])
        out["k2"] = float(camera_params[4])
        out["p1"] = 0.0
        out["p2"] = 0.0
        camera_model = CameraModel.OPENCV
    elif camera.model == "OPENCV":
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[1])
        out["cx"] = float(camera_params[2])
        out["cy"] = float(camera_params[3])
        out["k1"] = float(camera_params[4])
        out["k2"] = float(camera_params[5])
        out["p1"] = float(camera_params[6])
        out["p2"] = float(camera_params[7])
        camera_model = CameraModel.OPENCV
    elif camera.model == "OPENCV_FISHEYE":
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[1])
        out["cx"] = float(camera_params[2])
        out["cy"] = float(camera_params[3])
        out["k1"] = float(camera_params[4])
        out["k2"] = float(camera_params[5])
        out["k3"] = float(camera_params[6])
        out["k4"] = float(camera_params[7])
        camera_model = CameraModel.OPENCV_FISHEYE
    elif camera.model == "FULL_OPENCV":
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[1])
        out["cx"] = float(camera_params[2])
        out["cy"] = float(camera_params[3])
        out["k1"] = float(camera_params[4])
        out["k2"] = float(camera_params[5])
        out["p1"] = float(camera_params[6])
        out["p2"] = float(camera_params[7])
        out["k3"] = float(camera_params[8])
        out["k4"] = float(camera_params[9])
        out["k5"] = float(camera_params[10])
        out["k6"] = float(camera_params[11])
        raise NotImplementedError(f"{camera.model} camera model is not supported yet!")
    elif camera.model == "FOV":
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[1])
        out["cx"] = float(camera_params[2])
        out["cy"] = float(camera_params[3])
        out["omega"] = float(camera_params[4])
        raise NotImplementedError(f"{camera.model} camera model is not supported yet!")
    elif camera.model == "SIMPLE_RADIAL_FISHEYE":
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[0])
        out["cx"] = float(camera_params[1])
        out["cy"] = float(camera_params[2])
        out["k1"] = float(camera_params[3])
        out["k2"] = 0.0
        out["k3"] = 0.0
        out["k4"] = 0.0
        camera_model = CameraModel.OPENCV_FISHEYE
    elif camera.model == "RADIAL_FISHEYE":
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[0])
        out["cx"] = float(camera_params[1])
        out["cy"] = float(camera_params[2])
        out["k1"] = float(camera_params[3])
        out["k2"] = float(camera_params[4])
        out["k3"] = 0
        out["k4"] = 0
        camera_model = CameraModel.OPENCV_FISHEYE
    else:
        # THIN_PRISM_FISHEYE not supported!
        raise NotImplementedError(f"{camera.model} camera model is not supported yet!")

    out["camera_model"] = camera_model.value
    return out


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

CameraModel2 = collections.namedtuple("CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple("Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])

class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)

CAMERA_MODELS = {
    CameraModel2(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel2(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel2(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel2(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel2(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel2(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel2(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel2(model_id=7, model_name="FOV", num_params=5),
    CameraModel2(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel2(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel2(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12),
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model) for camera_model in CAMERA_MODELS])
CAMERA_MODEL_NAMES = dict([(camera_model.model_name, camera_model) for camera_model in CAMERA_MODELS])


def read_cameras_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, num_bytes=8 * num_params, format_char_sequence="d" * num_params)
            cameras[camera_id] = Camera(
                id=camera_id, model=model_name, width=width, height=height, params=np.array(params)
            )
        assert len(cameras) == num_cameras
    return cameras


def read_images_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = b""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":  # look for the ASCII 0 entry
                image_name += current_char
                current_char = read_next_bytes(fid, 1, "c")[0]
            image_name = image_name.decode("utf-8")
            num_points2D = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24 * num_points2D, format_char_sequence="ddq" * num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])), tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=image_name,
                xys=xys,
                point3D_ids=point3D_ids,
            )
    return images


def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def colmap_to_json(
    recon_dir: Path,
    camera_mask_path: Optional[Path] = None,
    image_id_to_depth_path: Optional[Dict[int, Path]] = None,
    image_rename_map: Optional[Dict[str, str]] = None,
    ply_filename="sparse_pc.ply",
    keep_original_world_coordinate: bool = False,
    use_single_camera_mode: bool = True,
) -> int:
    """Converts COLMAP's cameras.bin and images.bin to a JSON file.

    Args:
        recon_dir: Path to the reconstruction directory, e.g. "sparse/0"
        output_dir: Path to the output directory.
        camera_model: Camera model used.
        camera_mask_path: Path to the camera mask.
        image_id_to_depth_path: When including sfm-based depth, embed these depth file paths in the exported json
        image_rename_map: Use these image names instead of the names embedded in the COLMAP db
        keep_original_world_coordinate: If True, no extra transform will be applied to world coordinate.
                    Colmap optimized world often have y direction of the first camera pointing towards down direction,
                    while nerfstudio world set z direction to be up direction for viewer.
    Returns:
        The number of registered images.
    """

    # TODO(1480) use pycolmap
    # recon = pycolmap.Reconstruction(recon_dir)
    # cam_id_to_camera = recon.cameras
    # im_id_to_image = recon.images
    cam_id_to_camera = read_cameras_binary(recon_dir / "cameras.bin")
    im_id_to_image = read_images_binary(recon_dir / "images.bin")
    if set(cam_id_to_camera.keys()) != {1}:
        print(f"[bold yellow]Warning: More than one camera is found in {recon_dir}")
        print(cam_id_to_camera)
        use_single_camera_mode = False  # update bool: one camera per frame
        out = {}  # out = {"camera_model": parse_colmap_camera_params(cam_id_to_camera[1])["camera_model"]}
    else:  # one camera for all frames
        out = parse_colmap_camera_params(cam_id_to_camera[1])

    frames = []
    for im_id, im_data in im_id_to_image.items():
        # NB: COLMAP uses Eigen / scalar-first quaternions
        # * https://colmap.github.io/format.html
        # * https://github.com/colmap/colmap/blob/bf3e19140f491c3042bfd85b7192ef7d249808ec/src/base/pose.cc#L75
        # the `rotation_matrix()` handles that format for us.

        # TODO(1480) BEGIN use pycolmap API
        # rotation = im_data.rotation_matrix()
        rotation = qvec2rotmat(im_data.qvec)

        translation = im_data.tvec.reshape(3, 1)
        w2c = np.concatenate([rotation, translation], 1)
        w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]])], 0)
        c2w = np.linalg.inv(w2c)
        # Convert from COLMAP's camera coordinate system (OpenCV) to ours (OpenGL)
        c2w[0:3, 1:3] *= -1
        if not keep_original_world_coordinate:
            c2w = c2w[np.array([0, 2, 1, 3]), :]
            c2w[2, :] *= -1

        name = im_data.name
        if image_rename_map is not None:
            name = image_rename_map[name]
        name = Path(f"./images/{name}")

        frame = {
            "file_path": name.as_posix(),
            "transform_matrix": c2w.tolist(),
            "colmap_im_id": im_id,
        }
        if camera_mask_path is not None:
            frame["mask_path"] = camera_mask_path.relative_to(camera_mask_path.parent.parent).as_posix()
        if image_id_to_depth_path is not None:
            depth_path = image_id_to_depth_path[im_id]
            frame["depth_file_path"] = str(depth_path.relative_to(depth_path.parent.parent))

        if not use_single_camera_mode:  # add the camera parameters for this frame
            frame.update(parse_colmap_camera_params(cam_id_to_camera[im_data.camera_id]))

        frames.append(frame)

    out["frames"] = frames

    applied_transform = None
    if not keep_original_world_coordinate:
        applied_transform = np.eye(4)[:3, :]
        applied_transform = applied_transform[np.array([0, 2, 1]), :]
        applied_transform[2, :] *= -1
        out["applied_transform"] = applied_transform.tolist()

    # # create ply from colmap
    # assert ply_filename.endswith(".ply"), f"ply_filename: {ply_filename} does not end with '.ply'"
    # create_ply_from_colmap(
    #     ply_filename,
    #     recon_dir,
    #     output_dir,
    #     torch.from_numpy(applied_transform).float() if applied_transform is not None else None,
    # )
    # out["ply_file_path"] = ply_filename

    # with open(output_dir / "transforms.json", "w", encoding="utf-8") as f:
    #     json.dump(out, f, indent=4)

    return out


def _load_renderings(root_fp: Path, scene: str):
    """Load images and camera data, and normalize the scene."""

    data_dir = root_fp / scene
    sparse_dir = data_dir / "sparse/0"

    # 1. Load camera data from COLMAP and images
    meta = colmap_to_json(
        recon_dir=sparse_dir,
        camera_mask_path=None,
        image_id_to_depth_path=None,
        image_rename_map=None,
        keep_original_world_coordinate=False,
    )

    images = []
    camtoworlds = []
    focals = []
    centers = []
    scale_down_factor = 4  # images_4 are 4x smaller than original

    for frame in meta["frames"]:
        fname = os.path.join(data_dir, "images_4", frame["file_path"].replace("images/", ""))
        rgba = imageio.imread(fname)

        camtoworlds.append(np.array(frame["transform_matrix"]))
        images.append(rgba)

        if "fl_x" in meta and "fl_y" in meta:
            focals.append((meta["fl_x"] / scale_down_factor, meta["fl_y"] / scale_down_factor))
        else:
            focals.append(None)

        if "cx" in meta and "cy" in meta:
            centers.append(np.array([meta["cx"] / scale_down_factor, meta["cy"] / scale_down_factor]))
        else:
            centers.append(None)

    images = np.stack(images, axis=0)
    camtoworlds = np.stack(camtoworlds, axis=0)

    # # 2. Normalize using COLMAP point cloud
    # points = read_points3D_binary(sparse_dir / "points3D.bin")
    # xyz = np.array([pt['xyz'] for pt in points])

    # centroid = xyz.mean(axis=0)
    # max_dist = np.max(np.linalg.norm(xyz - centroid, axis=1))

    # camtoworlds[:, :3, 3] -= centroid
    # camtoworlds[:, :3, 3] -= np.array([0.0, -1.5, 3.0])  # Centering at origin
    # camtoworlds[:, :3, 3] *= 2.0

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

        # ply_path = f"{self.config.data_root}/{self.config.scene}/sparse/0/points3D.bin"
        # pts = read_points3D_binary(ply_path)
        # pts = np.array([p["xyz"] for p in pts], dtype=np.float64)

        # pts, self.centroid, self.scale = normalize_points(pts)
        
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