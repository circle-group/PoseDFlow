import math
import os

os.environ["PYOPENGL_PLATFORM"] = "egl"

import torch
import smplx
import trimesh
import pyrender

import numpy as np

from pytorch_lightning.callbacks import Callback
from PIL import Image
from scipy.spatial.transform import Rotation as R

import manifm.angles_utils


class BodyModelCallback(Callback):
    def __init__(self, wandb_logger, cfg):
        super().__init__()
        self._wandb_logger = wandb_logger
        self.cfg = cfg
        self.body_model = BodyModel(
            smpl_folder=cfg.body_model.smpl_folder,
            smpl_type=cfg.body_model.smpl_type,
            ext=cfg.body_model.ext,
            gender=cfg.body_model.gender,
        )
        self.counter = 0
        self.log_every = cfg.body_model.log_imgs_every

    def on_train_epoch_end(self, trainer, pl_module):
        self.counter += 1
        if self.counter % self.log_every == 0:
            n_poses = 20
            poses = pl_module.sample_all(
                n_samples=n_poses, device=pl_module.device
            )
            poses_first = poses[0, ::]
            poses_last = poses[-1, ::]
            meshes = self.body_model(body_pose=poses_first, n_shapes=n_poses)
            rend = self.body_model.render_all(meshes)
            self._wandb_logger.log_image(key="initial_noise", images=rend)

            meshes = self.body_model(body_pose=poses_last, n_shapes=n_poses)
            rend = self.body_model.render_all(meshes)
            self._wandb_logger.log_image(key="sample_images", images=rend)


class BodyModel:
    def __init__(
        self,
        smpl_folder,
        smpl_type="smplh",
        ext="npz",
        gender="neutral",
        num_betas=10,
        num_expression_coeffs=10,
        use_face_contour=False,
    ):
        self._model = smplx.create(
            smpl_folder,
            model_type=smpl_type,
            gender=gender,
            use_face_contour=use_face_contour,
            num_betas=num_betas,
            num_expression_coeffs=num_expression_coeffs,
            ext=ext,
        )

    def __call__(
        self,
        betas=None,
        expression=None,
        global_orient=None,
        body_pose=None,
        transl=None,
        n_shapes=1,
    ):
        if betas is None:
            betas = torch.randn([1, self._model.num_betas], dtype=torch.float32)
        if expression is None:
            expression = torch.randn(
                [1, self._model.num_expression_coeffs], dtype=torch.float32
            )
        if global_orient is None:
            # Compose the rotations
            Rx = R.from_rotvec([math.radians(30), 0, 0])
            Ry = R.from_rotvec([0, math.radians(0), 0])
            Rz = R.from_rotvec([0, 0, math.radians(90)])
            R_combined = Rz * Ry * Rx

            # Convert to axis-angle (shape: [3])
            axis_angle = torch.tensor(R_combined.as_rotvec(), dtype=torch.float32)

            # Expand to [n_shapes, 3]
            global_orient = axis_angle.repeat(n_shapes, 1)

        if body_pose.shape[-1] == 189:
            body_pose = manifm.angles_utils.so3_to_axis_angle(
                body_pose.reshape(-1, 21, 3, 3)
            )
            # pad last two joints that are not used
            body_pose = torch.cat(
                [body_pose, torch.zeros([n_shapes, 2, 3]).to(body_pose)],
                dim=1,
            ).reshape(n_shapes, -1)

        output = self._model(
            betas=betas.cpu(),
            expression=expression.cpu(),
            global_orient=global_orient,
            body_pose=body_pose.cpu(),
            transl=transl,
        )

        meshes = []
        for i in range(n_shapes):
            meshes.append(
                {
                    "mesh": trimesh.Trimesh(
                        vertices=output.vertices[i].detach().cpu().numpy(),
                        faces=self._model.faces,
                    ),
                    "joints": output.joints[i].detach().cpu().numpy(),
                }
            )

        return meshes

    def render_all(self, meshes, plot_joints=True):
        renderings = []
        for mesh in meshes:
            renderings.append(self.render_mesh(mesh, plot_joints=plot_joints))
        return renderings

    def render_mesh(self, mesh_with_joints, plot_joints=True):
        m = mesh_with_joints["mesh"]
        joints = mesh_with_joints["joints"]

        vertex_colors = np.ones([m.vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]
        tri_mesh = trimesh.Trimesh(
            m.vertices, m.faces, vertex_colors=vertex_colors
        )

        mesh = pyrender.Mesh.from_trimesh(tri_mesh)

        scene = pyrender.Scene()
        pc = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.414)
        s = np.sqrt(2) / 2
        camera_pose = np.array(
            [
                [0.0, -s, s, 1.3],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, s, s, 1.35],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        scene.add(mesh)
        scene.add(pc, pose=camera_pose)

        if plot_joints:
            sm = trimesh.creation.uv_sphere(radius=0.005)
            sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]
            tfs = np.tile(np.eye(4), (len(joints), 1, 1))
            tfs[:, :3, 3] = joints
            joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
            scene.add(joints_pcl)

        r = pyrender.OffscreenRenderer(
            viewport_width=640, viewport_height=480, point_size=1.0
        )
        flags = (
            pyrender.RenderFlags.RGBA | pyrender.RenderFlags.SHADOWS_DIRECTIONAL
        )
        color, _ = r.render(scene, flags=flags)
        r.delete()
        return color

    def save_rendering(self, rendered_img, path):
        img = Image.fromarray(rendered_img)
        img.save(path)

    def save_renderings(self, rendered_images, dir_path, suffix=""):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        for i, mesh in enumerate(rendered_images):
            self.save_rendering(
                mesh, os.path.join(dir_path, f"pose_{i}_{suffix}.png")
            )
