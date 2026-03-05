import torch
import numpy as np
from tqdm import tqdm

from geoopt.tensor import ManifoldParameter
from geoopt.optim import RiemannianAdam

from manifm.solvers import projx_integrator
from manifm.angles_utils import axis_angle_to_so3, so3_to_axis_angle, matrix_to_axis_angle



def flip_orientations(global_orient):
    """
    Flips the global orientation vectors around the Y axis by 180 degrees, mirroring the direction.

    The function converts rotation vectors to rotation matrices, applies a 180-degree rotation around the Y axis,
    and then converts the matrices back to rotation vectors.

    Shape:
        - Input: :Torch:`(N, 3)` where `N` is the batch size, and each row is a 3D rotation vector.
        - Output: :Torch:`(N, 3)` where each row is the flipped 3D rotation vector corresponding to the input.

    Parameters:
        - global_orient (:Torch:`Tensor`): The input batch of 3D rotation vectors.

    Returns:
        - :Torch:`Tensor`: A tensor of the same shape as `global_orient`, representing the flipped orientations.
    """
    # Convert rotation vectors to rotation matrices
    rotation_matrices = axis_angle_to_so3(global_orient)

    # Define 180-degree rotation matrix around Y axis
    cos_angle, sin_angle = torch.cos(torch.tensor(np.pi)), torch.sin(torch.tensor(np.pi))
    y_rot_matrix = torch.tensor([[cos_angle.item(), 0, sin_angle.item(), 0],
                                 [0, 1, 0, 0],
                                 [-sin_angle.item(), 0, cos_angle.item(), 0],
                                 [0, 0, 0, 1]], device=global_orient.device)

    # Expand Y rotation matrix for batch size
    y_rot_matrix_batch = y_rot_matrix.unsqueeze(0).repeat(global_orient.size(0), 1, 1)

    # Apply Y rotation to each rotation matrix
    flipped_matrices = torch.matmul(rotation_matrices, y_rot_matrix_batch)[:, :3, :]

    # Convert rotation matrices back to rotation vectors
    flipped_orient = so3_to_axis_angle(flipped_matrices)

    # Return the adjusted rotation vectors
    return flipped_orient[:, :3]


def perspective_projection(points,
                           rotation,
                           translation,
                           focal_length,
                           camera_center):
    """
    This function computes the perspective projection of a set of points.

    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        focal_length (bs,) or scalar: Focal length
        camera_center (bs, 2): Camera center
    """

    batch_size = points.shape[0]
    K = torch.zeros([batch_size, 3, 3], device=points.device)
    K[:, 0, 0] = focal_length
    K[:, 1, 1] = focal_length
    K[:, 2, 2] = 1.
    K[:, :-1, -1] = camera_center

    # Transform points
    points = torch.einsum('bij,bkj->bki', rotation, points)

    # Apply perspective distortion
    projected_points = points / points[:, :, -1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)

    return projected_points[:, :, :-1]


def gmof(x, sigma):
    """
    Geman-McClure error function
    """
    x_squared = x ** 2
    sigma_squared = sigma ** 2
    return (sigma_squared * x_squared) / (sigma_squared + x_squared)


def angle_prior(pose):
    """
    Angle prior that penalizes unnatural bending of the knees and elbows
    """
    # We subtract 3 because pose does not include the global rotation of the model
    return torch.exp(
            pose[:, [55 - 3, 58 - 3, 12 - 3, 15 - 3]] * torch.tensor([1., -1., -1, -1.], device=pose.device)) ** 2


def camera_fitting_loss(model_joints, camera_t, camera_t_est, camera_center, joints_2d, joints_conf,
                        focal_length=5000, depth_loss_weight=100, part="body"):
    """
    Loss function for camera optimization.
    """

    # Project model joints
    batch_size = model_joints.shape[0]
    rotation = torch.eye(3, device=model_joints.device).unsqueeze(0).expand(batch_size, -1, -1)
    projected_joints = perspective_projection(model_joints, rotation, camera_t,
                                              focal_length, camera_center)
    if part == "body":
        op_joints = ['RShoulder', 'LShoulder', 'R_Hip', 'L_Hip',]
        op_joints_ind = [2, 5, 9, 12,]
    elif part == "rhand":
        op_joints = ['R_Wrist_Hand', 'R_Thumb_1', 'R_Index_1', 'R_Ring_1', 'R_Pinky_1',]
        op_joints_ind = [0, 1, 5, 9, 13, 17]
    elif part == "lhand":
        op_joints = ['L_Wrist_Hand', 'L_Thumb_1', 'L_Index_1', 'L_Ring_1', 'L_Pinky_1',]
        op_joints_ind = [0, 1, 5, 9, 13, 17]
    else:
        raise ValueError(f"Unknown part: {part}")
    reprojection_error_op = (joints_2d[:, op_joints_ind] -
                             projected_joints[:, op_joints_ind]) ** 2

    reprojection_loss = (joints_conf[:, op_joints_ind] ** 2) * reprojection_error_op.sum(dim=-1)  # sum along x-y

    # Loss that penalizes deviation from depth estimate
    depth_loss = (depth_loss_weight ** 2) * (camera_t[:, 2] - camera_t_est[:, 2]) ** 2

    total_loss = reprojection_loss.sum(dim=-1) + depth_loss  # sum along different joints
    return total_loss.sum()


class SMPLify:
    """Implementation of single-stage SMPLify."""

    def __init__(self,
                 body_model,
                 manifold,
                 dim,
                 vecfield,
                 step_size=1e-2,
                 batch_size=32,
                 num_iters=100,
                 focal_length=5000,
                 side_view_thsh=25.0,
                 device='cuda:0',
                 seed=42):
        self.smpl = body_model
        self.manifold = manifold
        self.dim = dim
        self.vecfield = vecfield

        # Store options
        self.device = device
        self.focal_length = focal_length
        self.side_view_thsh = side_view_thsh
        self.step_size = step_size
        self.seed = seed

        # Ignore the the following joints for the fitting process
        ign_joints = ['OP Neck', 'OP RHip', 'OP LHip', 'Right Hip', 'Left Hip']
        self.ign_joints = [1, 9, 12, 27, 28]
        self.num_iters = num_iters

    def __call__(self, init_pose, init_betas, init_cam_t, camera_center, keypoints_2d):
        """Perform body fitting.
        Input:
            init_pose: SMPL pose estimate
            init_betas: SMPL betas estimate
            init_cam_t: Camera translation estimate
            camera_center: Camera center location
            keypoints_2d: Keypoints used for the optimization
        Returns:
            vertices: Vertices of optimized shape
            joints: 3D joints of optimized shape
            pose: SMPL pose parameters of optimized shape
            betas: SMPL beta parameters of optimized shape
            camera_translation: Camera translation
            reprojection_loss: Final joint reprojection loss
        """
        # Make camera translation a learnable parameter
        camera_translation = init_cam_t.clone()

        # Get joint confidence
        joints_2d = keypoints_2d[:, :, :2].clone()
        joints_conf = keypoints_2d[:, :, -1].clone()

        # Split SMPL pose to body pose and global orientation
        body_pose = init_pose[:, 3:].detach().clone()
        global_orient = init_pose[:, :3].detach().clone()
        betas = init_betas.detach().clone()

        # Step 1: Optimize camera translation and body orientation
        # Optimize only camera translation and body orientation
        body_pose.requires_grad = False
        betas.requires_grad = False
        global_orient.requires_grad = True
        camera_translation.requires_grad = True

        camera_opt_params = [global_orient, camera_translation]
        camera_optimizer = torch.optim.Adam(camera_opt_params, lr=self.step_size, betas=(0.9, 0.999))

        for i in range(self.num_iters):
            smpl_output = self.smpl(betas=betas,
                                    body_pose=body_pose,
                                    global_orient=global_orient,
                                    pose2rot=True,
                                    transl=camera_translation)

            model_joints = smpl_output.joints
            loss = camera_fitting_loss(model_joints, camera_translation,
                                       init_cam_t, camera_center,
                                       joints_2d, joints_conf, focal_length=self.focal_length, part='body')

            camera_optimizer.zero_grad()
            loss.backward()
            camera_optimizer.step()

        left_shoulder_idx, right_shoulder_idx = 2, 5
        shoulder_dist = torch.dist(joints_2d[:, left_shoulder_idx],
                                   joints_2d[:, right_shoulder_idx])
        try_both_orient = shoulder_dist.item() < self.side_view_thsh

        # Step 2: Optimize body joints
        # For joints ignored during fitting, set the confidence to 0
        joints_conf[:, self.ign_joints] = 0.

        inputs = (global_orient, body_pose, betas, camera_translation, camera_center, joints_2d, joints_conf)
        if try_both_orient:
            pose, betas, camera_translation, reprojection_loss = self.optimize_and_compare(*inputs)
        else:
            reprojection_loss, (pose, betas, camera_translation, _) = self.optimize_body(*inputs)

        return pose, betas, camera_translation, reprojection_loss

    def optimize_and_compare(self, global_orient, body_pose, betas, camera_translation, camera_center, joints_2d,
                             joints_conf):
        original_loss, original_results = self.optimize_body(global_orient.detach(), body_pose, betas, camera_translation,
                                                             camera_center, joints_2d, joints_conf)
        flipped_loss, flipped_results = self.optimize_body(flip_orientations(global_orient).detach(), body_pose, betas,
                                                           camera_translation, camera_center, joints_2d, joints_conf)

        min_loss_indices = original_loss < flipped_loss  # [N,]

        pose = torch.where(min_loss_indices.unsqueeze(-1), original_results[0], flipped_results[0])
        betas = torch.where(min_loss_indices.unsqueeze(-1), original_results[1], flipped_results[1])
        camera_translation = torch.where(min_loss_indices.unsqueeze(-1), original_results[2], flipped_results[2])
        reprojection_loss = torch.where(min_loss_indices, original_loss, flipped_loss)

        return pose, betas, camera_translation, reprojection_loss

    def optimize_body(self, global_orient, body_pose, betas, camera_translation, camera_center, joints_2d, joints_conf):
        """
        Optimize only the body pose and global orientation of the body
        """
        batch_size = global_orient.shape[0]

        body_pose.requires_grad = False
        betas.requires_grad = True
        global_orient.requires_grad = True
        camera_translation.requires_grad = False
        body_opt_params = [betas, global_orient]
        num_steps = 100

        time_grid = torch.linspace(0, 1, num_steps+1).to(self.device)
        rotation = torch.eye(3, device=self.device).unsqueeze(0).expand(batch_size, -1, -1)
        body_pose_so3 = axis_angle_to_so3(body_pose.view(-1, 21, 3)).view(-1, self.dim)

        torch.manual_seed(self.seed)
        init_noise = (
            self.manifold.random_combined(batch_size, self.dim)
            .reshape(batch_size, self.dim)
            .to(self.device)
        )

        x0_ = projx_integrator(
            self.manifold,
            self.vecfield,
            body_pose_so3,
            time_grid,
            method="euler",
            projx=False,
            local_coords=False,
        )[0][-1]
        x0 = x0_ * np.sqrt(0.75) + init_noise * np.sqrt(0.25)
        x0 = self.manifold.projx(x0)

        x0 = ManifoldParameter(x0, manifold=self.manifold, requires_grad=True)
        pose_optimizer = RiemannianAdam([x0], lr=self.step_size, weight_decay=1e-4)

        body_optimizer = torch.optim.Adam(body_opt_params, lr=self.step_size, betas=(0.9, 0.999))
        num = 400

        for i in tqdm(range(num)):
            body_optimizer.zero_grad()
            pose_optimizer.zero_grad()

            traj = projx_integrator(
                self.manifold,
                self.vecfield,
                x0,
                time_grid,
                method="euler",
                projx=False,
                local_coords=False,
                enable_grad=True
            )[0]
            x1 = self.manifold.projx(traj[-1])
            body_pose = matrix_to_axis_angle(x1.view(-1, 21, 3, 3)).view(-1, self.dim//3)

            smpl_output = self.smpl(betas=betas,
                                    body_pose=body_pose,
                                    global_orient=global_orient,
                                    pose2rot=True,
                                    transl=camera_translation)

            model_joints = smpl_output.joints
            projected_joints = perspective_projection(model_joints, rotation, camera_translation,
                                                      self.focal_length, camera_center)

            reprojection_error = gmof(projected_joints - joints_2d, sigma=100)
            reprojection_loss = ((joints_conf ** 2) * reprojection_error.sum(dim=-1)).sum(dim=-1).mean()

            # Angle prior for knees and elbows
            angle_prior_loss = angle_prior(body_pose).sum(dim=-1).mean()

            # Regularizer to prevent betas from taking large values
            shape_prior_loss = (betas ** 2).sum(dim=-1).mean()

            traj = traj.reshape(num_steps+1, batch_size, -1, 3, 3)
            running_cost = (3 - torch.einsum("...ii", traj)).sum(-1)
            running_cost = running_cost.mean(1).sum() / (num_steps+1)

            if i % 20 == 0 or i == num-1:
                print(f"Step {i}, Loss: {reprojection_loss.item():.4f}, Angle prior: {angle_prior_loss.item():.4f}, Shape prior: {shape_prior_loss.item():.4f}, Running cost: {running_cost.item():.4f}")

            loss = reprojection_loss + 50*angle_prior_loss + 50*shape_prior_loss + 50*running_cost
            loss.backward()

            body_optimizer.step()
            pose_optimizer.step()
            

        pose = torch.cat([global_orient, body_pose], dim=-1).detach()
        betas = betas.detach()
        return reprojection_loss, (pose, betas, camera_translation, reprojection_loss)
