import os
import json
import torch
import numpy as np


# From https://github.com/vchoutas/smplx/blob/main/smplx/joint_names.py
# Please see license for usage restrictions.
#
# SMPLX body poses (exclude "left_hand" and "right_hand" in SMPL, 21 local poses)
BODY_JOINT_NAMES = [
    "pelvis",  # global_orient actually
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
]

body_name_to_index = {name: index - 1 for index, name in enumerate(BODY_JOINT_NAMES)}


class BodyPartIndices:
    left_leg = sorted([body_name_to_index[name] for name in ["left_hip", "left_knee", "left_ankle", "left_foot"]])
    right_leg = sorted([body_name_to_index[name] for name in ["right_hip", "right_knee", "right_ankle", "right_foot"]])
    left_arm = sorted(
        [body_name_to_index[name] for name in ["left_collar", "left_shoulder", "left_elbow", "left_wrist"]])
    right_arm = sorted(
        [body_name_to_index[name] for name in ["right_collar", "right_shoulder", "right_elbow", "right_wrist"]])
    trunk = sorted(
        [body_name_to_index[name] for name in ["spine1", "spine2", "spine3", "left_shoulder", "right_shoulder"]])
    hands = sorted([body_name_to_index[name] for name in ["left_wrist", "right_wrist"]])
    legs = sorted(left_leg + right_leg)
    arms = sorted(left_arm + right_arm)
    left_body = sorted(left_leg + left_arm)
    right_body = sorted(right_leg + right_arm)

    @staticmethod
    def get_indices(part_name):
        return getattr(BodyPartIndices, part_name)

    @staticmethod
    def get_joint_indices(part_name):
        return [x+1 for x in getattr(BodyPartIndices, part_name)]   # +1 to account for global_orient
    

class BodySegIndices:
    current_file_path = os.path.abspath(__file__)
    segmentdata = json.load(
        open('examples/smplx_vert_segmentation.json')
    )

    left_leg = sorted(list(
        set(segmentdata['leftLeg'] + segmentdata['leftUpLeg'] + segmentdata['leftFoot'] + segmentdata['leftToeBase'])))
    right_leg = sorted(list(set(
        segmentdata['rightLeg'] + segmentdata['rightUpLeg'] + segmentdata['rightFoot'] + segmentdata['rightToeBase'])))
    left_arm = sorted(list(set(segmentdata['leftArm'] + segmentdata['leftForeArm'])))
    right_arm = sorted(list(set(segmentdata['rightArm'] + segmentdata['rightForeArm'])))
    trunk = sorted(list(set(
        segmentdata['spine1'] + segmentdata['spine2'] + segmentdata['leftShoulder'] + segmentdata['rightShoulder'])))
    hands = sorted(list(set(segmentdata['leftHand'] + segmentdata['rightHand'])))
    legs = sorted(list(set(left_leg + right_leg)))
    arms = sorted(list(set(left_arm + right_arm)))


@torch.no_grad()
def average_pairwise_distance(joints3d, relative_root=False):
    """
    Calculate Average Pairwise Distance (APD) for a batch of poses.

    Parameters:
    - joints3d (torch.Tensor): A tensor of shape (batch_size, num_joints, 3)
    - relative_root (bool): If True, the root joint is subtracted from all joints before calculating APD.

    Returns:
    - APD (torch.Tensor): Average Pairwise Distance (unit: meter)
    """
    batch_size, num_joints, _ = joints3d.shape
    if relative_root:
        joints3d = joints3d - joints3d[:, 0:1, :]

    # Initialize tensor to store pairwise distances between samples in the batch
    pairwise_distances = torch.zeros(batch_size, batch_size)

    for i in range(batch_size):
        for j in range(i + 1, batch_size):
            # Calculate the pairwise distance between sample i and sample j
            dist = torch.mean(torch.norm(joints3d[i, :, :] - joints3d[j, :, :], dim=-1))
            pairwise_distances[i, j] = dist
            pairwise_distances[j, i] = dist  # Distance is symmetric

    # The diagonal is zero as the distance between a sample and itself is zero
    pairwise_distances.fill_diagonal_(0)

    # Calculate the mean over all the pairwise distances in the batch to get APD
    APD = torch.sum(pairwise_distances) / (batch_size * (batch_size - 1))

    return APD
    

class Evaler:
    def __init__(self, body_model, part=None):
        self.body_model = body_model
        self.part = part


        if self.part is not None:
            self.joint_idx = np.array(getattr(BodyPartIndices, self.part)) + 1  # skip pelvis
            self.vert_idx = np.array(getattr(BodySegIndices, self.part))
        else:
            self.joint_idx = slice(None)
            self.vert_idx = slice(None)

    def eval_bodys(self, outs, gts):
        '''
        :param outs: [b, j*3] axis-angle results of body poses
        :param gts:  [b, j*3] axis-angle groundtruth of body poses
        :return: result dict for every sample [b,]
        '''
        sample_num = len(outs)
        betas = torch.zeros((sample_num, 10)).to(gts.device)
        eval_result = {'mpvpe': [], 'mpjpe': []}
        body_gt = self.body_model(body_pose=gts, betas=betas)
        body_out = self.body_model(body_pose=outs, betas=betas)

        # MPVPE from all vertices
        mesh_gt = body_gt.vertices.detach().cpu().numpy()[:, self.vert_idx]
        mesh_out = body_out.vertices.detach().cpu().numpy()[:, self.vert_idx]
        eval_result['mpvpe'].extend((np.sqrt(np.sum((mesh_out - mesh_gt) ** 2, 2)).mean(1) * 1000).tolist())

        joint_gt_body = body_gt.joints.detach().cpu().numpy()[:, self.joint_idx]
        joint_out_body = body_out.joints.detach().cpu().numpy()[:, self.joint_idx]
        eval_result['mpjpe'].extend(
            (np.sqrt(np.sum((joint_out_body - joint_gt_body) ** 2, 2)).mean(1) * 1000).tolist())

        return eval_result

    def multi_eval_bodys(self, outs, gts):
        '''
        :param outs: [b, hypo, j*3] axis-angle results of body poses, multiple hypothesis
        :param gts:  [b, j*3] axis-angle groundtruth of body poses
        :return: result dict [b,]
        '''
        hypo_num = outs.shape[1]
        eval_result = {f'mpvpe': [], f'mpjpe': []}
        for hypo in range(hypo_num):
            result = self.eval_bodys(outs[:, hypo], gts)
            eval_result['mpvpe'].append(result['mpvpe'])
            eval_result['mpjpe'].append(result['mpjpe'])

        eval_result['mpvpe'] = np.min(eval_result['mpvpe'], axis=0)
        eval_result['mpjpe'] = np.min(eval_result['mpjpe'], axis=0)

        return eval_result

    def multi_eval_bodys_all(self, outs, gts):
        '''
        :param outs: [b, hypo, j*3] axis-angle results of body poses, multiple hypothesis
        :param gts:  [b, j*3] axis-angle groundtruth of body poses
        :return: result dict [b,]
        '''
        hypo_num = outs.shape[1]
        eval_collector = {f'mpvpe': [], f'mpjpe': []}
        eval_result = {f'mpvpe_best': [], f'mpjpe_best': [],
                       f'mpvpe_mean': [], f'mpjpe_mean': [],
                       f'mpvpe_std': [], f'mpjpe_std': []}
        for hypo in range(hypo_num):
            result = self.eval_bodys(outs[:, hypo], gts)
            eval_collector['mpvpe'].append(result['mpvpe'])
            eval_collector['mpjpe'].append(result['mpjpe'])

        eval_result['mpvpe_best'] = np.min(eval_collector['mpvpe'], axis=0)
        eval_result['mpjpe_best'] = np.min(eval_collector['mpjpe'], axis=0)
        eval_result['mpvpe_mean'] = np.mean(eval_collector['mpvpe'], axis=0)
        eval_result['mpjpe_mean'] = np.mean(eval_collector['mpjpe'], axis=0)
        eval_result['mpvpe_std'] = np.std(eval_collector['mpvpe'], axis=0)
        eval_result['mpjpe_std'] = np.std(eval_collector['mpjpe'], axis=0)

        return eval_result

    def print_eval_result(self, eval_result):
        print('MPVPE: %.2f mm' % np.mean(eval_result['mpvpe']))
        print('MPJPE: %.2f mm' % np.mean(eval_result['mpjpe']))

    def print_eval_result_all(self, eval_result):
        # complete this
        print('MPVPE best: %.2f mm' % np.mean(eval_result['mpvpe_best']))
        print('MPJPE best: %.2f mm' % np.mean(eval_result['mpjpe_best']))
        print('MPVPE mean: %.2f mm' % np.mean(eval_result['mpvpe_mean']))
        print('MPJPE mean: %.2f mm' % np.mean(eval_result['mpjpe_mean']))
        print('MPVPE std: %.2f mm' % np.mean(eval_result['mpvpe_std']))
        print('MPJPE std: %.2f mm' % np.mean(eval_result['mpjpe_std']))

    def print_multi_eval_result(self, eval_result, hypo_num):
        print(f'multihypo {hypo_num} MPVPE: %.2f mm' % np.mean(eval_result['mpvpe']))
        print(f'multihypo {hypo_num} MPJPE: %.2f mm' % np.mean(eval_result['mpjpe']))
