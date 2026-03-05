import cv2
import json
from glob import glob
from torch.utils.data import DataLoader

import os
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path

import torch

from manifm.eval_utils import load_model
from manifm.amass_dataset import get_product_manifold
from experiment.hmr_utils import cam_crop2full, SMPLXNew, MocapDataset, compute_bbox
from experiment.smplify import SMPLify



joint_set_coco = {
    'joint_num': 135,  # body 25 (23 + pelvis + neck), lhand 21, rhand 21, face 68
    'joints_name': \
        ('Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist',
         'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Pelvis', 'Neck',
         'L_Big_toe', 'L_Small_toe', 'L_Heel', 'R_Big_toe', 'R_Small_toe', 'R_Heel',  # body part
         'L_Wrist_Hand', 'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3', 'L_Thumb_4', 'L_Index_1', 'L_Index_2',
         'L_Index_3', 'L_Index_4', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Middle_4', 'L_Ring_1',
         'L_Ring_2', 'L_Ring_3', 'L_Ring_4', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Pinky_4',  # left hand
         'R_Wrist_Hand', 'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3', 'R_Thumb_4', 'R_Index_1', 'R_Index_2',
         'R_Index_3', 'R_Index_4', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Middle_4', 'R_Ring_1',
         'R_Ring_2', 'R_Ring_3', 'R_Ring_4', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Pinky_4',  # right hand
         *['Face_' + str(i) for i in range(56, 73)],  # face contour
         *['Face_' + str(i) for i in range(5, 36)],  # eyebrow, nose, eyes
         *['Face_' + str(i) for i in range(36, 56)],  # outer mouth, inner mouth
         ),
    'flip_pairs': \
        ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (18, 21), (19, 22), (20, 23),
         # body part
         (24, 45), (25, 46), (26, 47), (27, 48), (28, 49), (29, 50), (30, 51), (31, 52), (32, 53), (33, 54),
         (34, 55), (35, 56), (36, 57), (37, 58), (38, 59), (39, 60), (40, 61), (41, 62), (42, 63), (43, 64),
         (44, 65),  # hand part
         (66, 82), (67, 81), (68, 80), (69, 79), (70, 78), (71, 77), (72, 76), (73, 75),  # face contour
         (83, 92), (84, 91), (85, 90), (86, 89), (87, 88),  # face eyebrow
         (97, 101), (98, 100),  # face below nose
         (102, 111), (103, 110), (104, 109), (105, 108), (106, 113), (107, 112),  # face eyes
         (114, 120), (115, 119), (116, 118), (121, 125), (122, 124),  # face mouth
         (126, 130), (127, 129), (131, 133)  # face lip
         )
}

joint_set_openpose = {
    'joint_num': 135,  # body 25, lhand 21, rhand 21, face 68
    'joints_name':
        ("Nose", "Neck", "R_Shoulder", "R_Elbow", "R_Wrist", "L_Shoulder", "L_Elbow", "L_Wrist",
         "Pelvis", "R_Hip", "R_Knee", "R_Ankle", "L_Hip", "L_Knee", "L_Ankle", "R_Eye", "L_Eye", "R_Ear", "L_Ear",
         'L_Big_toe', 'L_Small_toe', 'L_Heel', 'R_Big_toe', 'R_Small_toe', 'R_Heel',  # body part
         'L_Wrist_Hand', 'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3', 'L_Thumb_4', 'L_Index_1', 'L_Index_2',
         'L_Index_3', 'L_Index_4', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Middle_4', 'L_Ring_1',
         'L_Ring_2', 'L_Ring_3', 'L_Ring_4', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Pinky_4',  # left hand
         'R_Wrist_Hand', 'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3', 'R_Thumb_4', 'R_Index_1', 'R_Index_2',
         'R_Index_3', 'R_Index_4', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Middle_4', 'R_Ring_1',
         'R_Ring_2', 'R_Ring_3', 'R_Ring_4', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Pinky_4',  # right hand
         *['Face_' + str(i) for i in range(56, 73)],  # face contour
         *['Face_' + str(i) for i in range(5, 36)],  # eyebrow, nose, eyes
         *['Face_' + str(i) for i in range(36, 56)],  # outer mouth, inner mouth
         # 'L_Eyeball', 'R_Eyeball',  # eye ball, abandoned in practice
         ),
}

# creating index map from coco to openpose
def coco_to_openpose():
    # Initialize a list with -1 for each OpenPose keypoint
    mapping = [-1] * len(joint_set_openpose['joints_name'])

    # Loop through the COCO keypoints for every keypoints in OpenPose
    for i, openpose_keypoint_name in enumerate(joint_set_openpose['joints_name']):
        if openpose_keypoint_name in joint_set_coco['joints_name']:
            # Find the index of the COCO keypoint in the OpenPose list
            coco_index = joint_set_coco['joints_name'].index(openpose_keypoint_name)
            # Update the mapping for the found keypoint
            mapping[i] = coco_index
        else:
            # Handle the case where the COCO keypoint doesn't have a direct mapping in OpenPose
            print(f"No direct mapping for OpenPose keypoint: {openpose_keypoint_name}")

    return np.array(mapping)


# processing original coco whole-body annotations (133 joints)
def merge_joint(joint_img, feet_img, lhand_img, rhand_img, face_img):
    # pelvis
    lhip_idx = joint_set_coco['joints_name'].index('L_Hip')
    rhip_idx = joint_set_coco['joints_name'].index('R_Hip')
    pelvis = (joint_img[lhip_idx, :] + joint_img[rhip_idx, :]) * 0.5
    pelvis[2] = joint_img[lhip_idx, 2] * joint_img[rhip_idx, 2]  # joint_valid
    pelvis = pelvis.reshape(1, 3)

    lshoud_idx = joint_set_coco['joints_name'].index('L_Shoulder')
    rshoud_idx = joint_set_coco['joints_name'].index('R_Shoulder')
    neck = (joint_img[lshoud_idx, :] + joint_img[rshoud_idx, :]) * 0.5
    neck[2] = joint_img[lshoud_idx, 2] * joint_img[rshoud_idx, 2]  # joint_valid
    neck = neck.reshape(1, 3)

    # feet
    lfoot = feet_img[:3, :]
    rfoot = feet_img[3:, :]

    joint_img = np.concatenate((joint_img, pelvis, neck, lfoot, rfoot, lhand_img, rhand_img, face_img)).astype(
        np.float32)  # [135, 3]
    return joint_img


def vitpose_to_openpose(keypoints):
    # array: [133, 3] -> array: [135, 3]
    assert len(keypoints) == 133, f"Expected 133 keypoints, got {len(keypoints)}"

    # Now, split this array into parts for body, feet, left hand, right hand, and face
    body_kpts = keypoints[:17]  # 17 body keypoints
    feet_kpts = keypoints[17:23]  # 6 foot keypoints
    face_kpts = keypoints[23:91]  # 68 face keypoints
    left_hand_kpts = keypoints[91:112]  # 21 left hand keypoints
    right_hand_kpts = keypoints[112:133]  # 21 right hand keypoints

    # Call merge_joint with the prepared parts
    merged_joints = merge_joint(body_kpts, feet_kpts, left_hand_kpts, right_hand_kpts, face_kpts)
    idx_mapping = coco_to_openpose()

    return merged_joints[idx_mapping]



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-md", "--model_dir", type=str,
                        default="checkpoints/epoch-009_step-0_loss-0.0000.ckpt",
                        help="Path to the pretrained RFM model")
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument("--cliff", action='store_true', help="whether to use CLIFF initialization")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    img_paths = sorted(glob("data/ehf/EHF/*_img.jpg"))
    gt_ply_paths = sorted(glob("data/ehf/EHF/*_align.ply"))
    json_paths = sorted(glob("data/ehf/EHF_keypoints/vitpose_keypoints/predictions/*_img.json"))

    batch_size = 16
    N_POSES = 22

    smpl = SMPLXNew(
        'data/human_model_files/smplx',
        batch_size=batch_size,
    ).to(device)
    betas = torch.zeros((batch_size, 10)).to(device)

    cfg, model = load_model(args.model_dir)
    manifold, dim = get_product_manifold(cfg)

    vecfield = model.vecfield
    vecfield.eval()
    vecfield.to(device)
    vecfield_compiled = torch.compile(vecfield)

    img_names = [Path(path).stem for path in img_paths]
    total_length = len(img_paths)

    current_idx, batch_keypoints, batch_img, batch_ply = 0, [], [], []
    all_eval_results = {'pa_mpjpe': [], 'pa_mpvpe': [], 'pck50': []}
    
    if args.cliff:
        cliff_poses = np.load('data/ehf/CLIFF/smpl_poses_cliff.npz')

    for img_path, json_path, gt_ply_path in tqdm(zip(img_paths, json_paths, gt_ply_paths), desc='Dataset',
                                                 total=total_length):
        base_name = os.path.basename(img_path)
        img_name, _ = os.path.splitext(base_name)
        # load image and 2D keypoints
        img_bgr = cv2.imread(img_path)
        json_data = json.load(open(json_path))

        vit_keypoints = np.array(json_data[0]['keypoints'])
        keypoints = vitpose_to_openpose(vit_keypoints)[:25]

        batch_keypoints.append(keypoints)
        batch_img.append(img_bgr)
        batch_ply.append(gt_ply_path)
        if len(batch_keypoints) < batch_size:
            continue

        bboxes = compute_bbox(batch_keypoints)
        keypoints = np.array(batch_keypoints)
        print('batch keypoints:', keypoints.shape)
        # [batch_id, min_x, min_y, max_x, max_y]
        bend_init = torch.tensor([bboxes[batch_id, 2] > 400 for batch_id in range(batch_size)], device=device)
        bboxes = [np.array([batch_id, 400, 100, 1000, 1200]) for batch_id in range(batch_size)]

        assert len(bboxes) == batch_size
        mocap_db = MocapDataset(batch_img, bboxes, batch_size, device, 'data/human_model_files')
        mocap_data_loader = DataLoader(mocap_db, batch_size=batch_size, num_workers=0)

        for batch in mocap_data_loader:
            img_h = batch["img_h"].to(device).float()
            img_w = batch["img_w"].to(device).float()

            focal_length = batch["focal_length"].to(device).float()
            camera_center = torch.hstack((img_w[:, None], img_h[:, None])) / 2

            kpts = np.zeros((batch_size, 49, 3))
            kpts[:, :25, :] = keypoints
            keypoints_tensor = torch.from_numpy(kpts).to(device)

            if not args.cliff:
                smpl_poses = smpl.mean_poses[:N_POSES * 3].unsqueeze(0).repeat(batch_size, 1).to(device)  # N*66
                bend_pose = torch.from_numpy(
        np.load('examples/bend_pose.npz')['pose'][:, :N_POSES * 3]
    ).to(smpl_poses.device)
                smpl_poses[bend_init, 3:] = bend_pose[:, 3:]
                init_betas = smpl.mean_shape.unsqueeze(0).repeat(batch_size, 1).to(device)  # N*10

                # Convert the camera parameters from the crop camera to the full camera
                center = batch["center"].to(device).float()
                scale = batch["scale"].to(device).float()
                full_img_shape = torch.stack((img_h, img_w), dim=-1)
                pred_cam_crop = torch.tensor([[0.9, 0, 0]], device=device).repeat(batch_size, 1)
                init_cam_t = cam_crop2full(pred_cam_crop, center, scale, full_img_shape, focal_length)
            else:
                smpl_poses = torch.from_numpy(cliff_poses['pose'][current_idx:current_idx+batch_size, :66]).to(device)
                init_betas = torch.from_numpy(cliff_poses['shape'][current_idx:current_idx+batch_size]).to(device)
                init_cam_t = torch.from_numpy(cliff_poses['global_t'][current_idx:current_idx+batch_size]).to(device)
                focal_length = torch.from_numpy(cliff_poses['focal_l'][current_idx:current_idx+batch_size]).to(device)

            init_vertices = smpl(betas=init_betas,
                             body_pose=smpl_poses[:, 3:],
                             global_orient=smpl_poses[:, :3],
                             transl=init_cam_t).vertices

            # be careful: the estimated focal_length should be used here instead of the default constant
            smplify = SMPLify(smpl, manifold, dim, vecfield_compiled, step_size=args.lr, batch_size=batch_size,
                              num_iters=100, focal_length=focal_length, device=device, seed=current_idx//batch_size)
            results = smplify(smpl_poses.detach(),
                              init_betas.detach(),
                              init_cam_t.detach(),
                              camera_center,
                              keypoints_tensor)

            batch_results = mocap_db.eval_EHF(results, batch_ply)
            all_eval_results['pa_mpjpe'].extend(batch_results['pa_mpjpe'])
            all_eval_results['pa_mpvpe'].extend(batch_results['pa_mpvpe'])
            all_eval_results['pck50'].extend(batch_results['pck50'])
        
        batch_keypoints, batch_img, batch_ply = [], [], []  # clear for the next batch
        current_idx += batch_size

    print('results on whole dataset:')
    mocap_db.print_eval_result(all_eval_results)
