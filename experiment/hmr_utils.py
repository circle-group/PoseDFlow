import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from plyfile import PlyData

import smplx
from smplx import SMPLX as _SMPLX
from manifm.angles_utils import so3_to_axis_angle
try:
    from smplx.body_models import ModelOutput as SMPLOutput
except Exception as e:
    from smplx.body_models import SMPLOutput


CROP_IMG_HEIGHT = 256
CROP_IMG_WIDTH = 192
CROP_ASPECT_RATIO = CROP_IMG_HEIGHT / float(CROP_IMG_WIDTH)

# Mean and standard deviation for normalizing input image
IMG_NORM_MEAN = [0.485, 0.456, 0.406]
IMG_NORM_STD = [0.229, 0.224, 0.225]


def compute_bbox(keypoints_list):
    all_keypoints = keypoints_list
    bbox_list = []

    for batch_id, keypoints in enumerate(all_keypoints):
        visible_keypoints = keypoints[keypoints[:, 2] > 0]

        if len(visible_keypoints) == 0:
            continue

        x_coords = visible_keypoints[:, 0]
        y_coords = visible_keypoints[:, 1]

        min_x = np.min(x_coords)
        min_y = np.min(y_coords)
        max_x = np.max(x_coords)
        max_y = np.max(y_coords)

        # [batch_id, min_x, min_y, max_x, max_y]
        bbox = [batch_id, min_x, min_y, max_x, max_y]
        bbox_list.append(bbox)

    bbox_array = np.array(bbox_list)
    return bbox_array


def cam_crop2full(crop_cam, center, scale, full_img_shape, focal_length):
    """
    convert the camera parameters from the crop camera to the full camera
    :param crop_cam: shape=(N, 3) weak perspective camera in cropped img coordinates (s, tx, ty)
    :param center: shape=(N, 2) bbox coordinates (c_x, c_y)
    :param scale: shape=(N, 1) square bbox resolution  (b / 200)
    :param full_img_shape: shape=(N, 2) original image height and width
    :param focal_length: shape=(N,)
    :return:
    """
    img_h, img_w = full_img_shape[:, 0], full_img_shape[:, 1]
    cx, cy, b = center[:, 0], center[:, 1], scale * 200
    w_2, h_2 = img_w / 2., img_h / 2.
    bs = b * crop_cam[:, 0] + 1e-9
    tz = 2 * focal_length / bs
    tx = (2 * (cx - w_2) / bs) + crop_cam[:, 1]
    ty = (2 * (cy - h_2) / bs) + crop_cam[:, 2]
    full_cam = torch.stack([tx, ty, tz], dim=-1)
    return full_cam


def estimate_focal_length(img_h, img_w):
    return (img_w * img_w + img_h * img_h) ** 0.5  # fov: 55 degree


def rot6d_to_axis_angle(rot6d):
    """Convert 6d rotation representation to 3d vector of axis-angle rotation.
    Shape:
        - Input: :Torch:`(N, 6)`
        - Output: :Torch:`(N, 3)`
    """
    a1, a2 = rot6d[..., :3], rot6d[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    rot_mat = torch.stack((b1, b2, b3), dim=-2)
    axis_angle = so3_to_axis_angle(rot_mat).reshape(-1, 3)  # axis-angle
    return axis_angle


def rigid_transform_3D(A, B):
    n, dim = A.shape
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    H = np.dot(np.transpose(A - centroid_A), B - centroid_B) / n
    U, s, V = np.linalg.svd(H)
    R = np.dot(np.transpose(V), np.transpose(U))
    if np.linalg.det(R) < 0:
        s[-1] = -s[-1]
        V[2] = -V[2]
        R = np.dot(np.transpose(V), np.transpose(U))

    varP = np.var(A, axis=0).sum()
    c = 1 / varP * np.sum(s)

    t = -np.dot(c * R, np.transpose(centroid_A)) + np.transpose(centroid_B)
    return c, R, t


def rigid_align(A, B):
    c, R, t = rigid_transform_3D(A, B)
    A2 = np.transpose(np.dot(c * R, np.transpose(A))) + t
    return A2


def load_ply(file_name):
    plydata = PlyData.read(file_name)
    x = plydata['vertex']['x']
    y = plydata['vertex']['y']
    z = plydata['vertex']['z']
    v = np.stack((x, y, z), 1)
    return v


def get_transform(center, scale, res, rot=0):
    """Generate transformation matrix."""
    # res: (height, width), (rows, cols)
    crop_aspect_ratio = res[0] / float(res[1])
    h = 200 * scale
    w = h / crop_aspect_ratio
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / w
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / w + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot  # To match direction of rotation from cropping
        rot_mat = np.zeros((3, 3))
        rot_rad = rot * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
        rot_mat[2, 2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0, 2] = -res[1] / 2
        t_mat[1, 2] = -res[0] / 2
        t_inv = t_mat.copy()
        t_inv[:2, 2] *= -1
        t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
    return t


def transform(pt, center, scale, res, invert=0, rot=0):
    """Transform pixel location to different reference."""
    t = get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.]).T
    new_pt = np.dot(t, new_pt)
    return np.array([round(new_pt[0]), round(new_pt[1])], dtype=int) + 1


def crop(img, center, scale, res):
    """
    Crop image according to the supplied bounding box.
    res: [rows, cols]
    """
    # Upper left point
    ul = np.array(transform([1, 1], center, scale, res, invert=1)) - 1
    # Bottom right point
    br = np.array(transform([res[1] + 1, res[0] + 1], center, scale, res, invert=1)) - 1

    # Padding so that when rotated proper amount of context is included
    pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape, dtype=np.float32)

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])
    try:
        new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], old_x[0]:old_x[1]]
    except Exception as e:
        print(e)

    new_img = cv2.resize(new_img, (res[1], res[0]))  # (cols, rows)

    return new_img, ul, br


def bbox_from_detector(bbox, rescale=1.1):
    """
    Get center and scale of bounding box from bounding box.
    The expected format is [min_x, min_y, max_x, max_y].
    """
    # center
    center_x = (bbox[0] + bbox[2]) / 2.0
    center_y = (bbox[1] + bbox[3]) / 2.0
    center = torch.tensor([center_x, center_y])

    # scale
    bbox_w = bbox[2] - bbox[0]
    bbox_h = bbox[3] - bbox[1]
    bbox_size = max(bbox_w * CROP_ASPECT_RATIO, bbox_h)
    scale = bbox_size / 200.0
    # adjust bounding box tightness
    scale *= rescale
    return center, scale


def process_image(orig_img_rgb, bbox,
                  crop_height=CROP_IMG_HEIGHT,
                  crop_width=CROP_IMG_WIDTH):
    """
    Read image, do preprocessing and possibly crop it according to the bounding box.
    If there are bounding box annotations, use them to crop the image.
    If no bounding box is specified but openpose detections are available, use them to get the bounding box.
    """
    try:
        center, scale = bbox_from_detector(bbox)
    except Exception as e:
        print("Error occurs in person detection", e)
        # Assume that the person is centered in the image
        height = orig_img_rgb.shape[0]
        width = orig_img_rgb.shape[1]
        center = np.array([width // 2, height // 2])
        scale = max(height, width * crop_height / float(crop_width)) / 200.

    img, ul, br = crop(orig_img_rgb, center, scale, (crop_height, crop_width))
    crop_img = img.copy()

    img = img / 255.
    mean = np.array(IMG_NORM_MEAN, dtype=np.float32)
    std = np.array(IMG_NORM_STD, dtype=np.float32)
    norm_img = (img - mean) / std
    norm_img = np.transpose(norm_img, (2, 0, 1))

    return norm_img, center, scale, ul, br, crop_img


# from CLIFF (https://github.com/haofanwang/CLIFF), for fitting 2d keypoints only
JOINT_NAMES = [
    # 25 OpenPose joints (in the order provided by OpenPose)
    'OP Nose',
    'OP Neck',
    'OP RShoulder',
    'OP RElbow',
    'OP RWrist',
    'OP LShoulder',
    'OP LElbow',
    'OP LWrist',
    'OP MidHip',
    'OP RHip',
    'OP RKnee',
    'OP RAnkle',
    'OP LHip',
    'OP LKnee',
    'OP LAnkle',
    'OP REye',
    'OP LEye',
    'OP REar',
    'OP LEar',
    'OP LBigToe',
    'OP LSmallToe',
    'OP LHeel',
    'OP RBigToe',
    'OP RSmallToe',
    'OP RHeel',
    # 24 Ground Truth joints (superset of joints from different datasets)
    'Right Ankle',
    'Right Knee',
    'Right Hip',
    'Left Hip',
    'Left Knee',
    'Left Ankle',
    'Right Wrist',
    'Right Elbow',
    'Right Shoulder',
    'Left Shoulder',
    'Left Elbow',
    'Left Wrist',
    'Neck (LSP)',
    'Top of Head (LSP)',
    'Pelvis (MPII)',
    'Thorax (MPII)',
    'Spine (H36M)',
    'Jaw (H36M)',
    'Head (H36M)',
    'Nose',
    'Left Eye',
    'Right Eye',
    'Left Ear',
    'Right Ear'
]

# Dict containing the joints in numerical order
JOINT_IDS = {JOINT_NAMES[i]: i for i in range(len(JOINT_NAMES))}

# Map joints to SMPL joints
JOINT_MAP = {
    'OP Nose': 24, 'OP Neck': 12, 'OP RShoulder': 17,
    'OP RElbow': 19, 'OP RWrist': 21, 'OP LShoulder': 16,
    'OP LElbow': 18, 'OP LWrist': 20, 'OP MidHip': 0,
    'OP RHip': 2, 'OP RKnee': 5, 'OP RAnkle': 8,
    'OP LHip': 1, 'OP LKnee': 4, 'OP LAnkle': 7,
    'OP REye': 25, 'OP LEye': 26, 'OP REar': 27,
    'OP LEar': 28, 'OP LBigToe': 29, 'OP LSmallToe': 30,
    'OP LHeel': 31, 'OP RBigToe': 32, 'OP RSmallToe': 33, 'OP RHeel': 34,
    'Right Ankle': 8, 'Right Knee': 5, 'Right Hip': 45,
    'Left Hip': 46, 'Left Knee': 4, 'Left Ankle': 7,
    'Right Wrist': 21, 'Right Elbow': 19, 'Right Shoulder': 17,
    'Left Shoulder': 16, 'Left Elbow': 18, 'Left Wrist': 20,
    'Neck (LSP)': 47, 'Top of Head (LSP)': 48,
    'Pelvis (MPII)': 49, 'Thorax (MPII)': 50,
    'Spine (H36M)': 51, 'Jaw (H36M)': 52,
    'Head (H36M)': 53, 'Nose': 24, 'Left Eye': 26,
    'Right Eye': 25, 'Left Ear': 28, 'Right Ear': 27
}

class SMPLXNew(torch.nn.Module):
    def __init__(self, model_path: str, **kwargs):
        super(SMPLXNew, self).__init__()
        self.bm = _SMPLX(model_path, **kwargs)
        joints = [JOINT_MAP[i] for i in JOINT_NAMES]
        # a hack from SMPL to SMPLX openpose joints
        joints[:25] = [55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                       8, 1, 4, 7, 56, 57, 58, 59, 60, 61, 62,
                       63, 64, 65]
        # borrow from smpl, as simple initialization
        smpl_mean_params = np.load('examples/smpl_mean_params.npz')
        rot6d_poses = torch.tensor(smpl_mean_params['pose'], dtype=torch.float32)
        axis_poses = rot6d_to_axis_angle(rot6d_poses.reshape(-1, 6)).reshape(-1)
        self.register_buffer('mean_poses', axis_poses)  # [72]
        self.register_buffer('mean_shape', torch.tensor(smpl_mean_params['shape'], dtype=torch.float32))  # [10]
        self.faces = self.bm.faces_tensor.numpy()
        self.J_regressor_idx = {'pelvis': 0, 'lwrist': 20, 'rwrist': 21, 'neck': 12}
        self.joint_map = torch.tensor(joints, dtype=torch.long)

    def forward(self, *args, **kwargs):
        kwargs['get_skin'] = True
        smpl_output = self.bm(*args, **kwargs)
        joints = smpl_output.joints[:, self.joint_map, :]
        output = SMPLOutput(vertices=smpl_output.vertices,
                            global_orient=smpl_output.global_orient,
                            body_pose=smpl_output.body_pose,
                            joints=joints,
                            betas=smpl_output.betas,
                            full_pose=smpl_output.full_pose)
        return output


class MocapDataset(Dataset):
    def __init__(self, img_bgr_list, detection_list, batchsize=1, device='cuda:0', body_model_path=None):
        self.img_bgr_list = img_bgr_list
        self.detection_list = detection_list
        self.device = device

        # To evaluate EHF
        self.cam_param = {'R': [-2.98747896, 0.01172457, -0.05704687]}
        self.cam_param['R'], _ = cv2.Rodrigues(np.array(self.cam_param['R']))
        if body_model_path is not None:
            self.smplx = smplx.create(body_model_path,
                                   num_betas=10,
                                   batch_size=batchsize,
                                   model_type='smplx',
                                   gender='neutral').to(device)
            #self.smplx = _SMPLX(body_model_path,
            #                    model_type='smplx',
            #                    num_betas=10,
            #                    batch_size=batchsize,
            #                    num_expression_coeffs=100,
            #                    use_pca=False,
            #                    flat_hand_mean=False,
            #                    use_face_contour=True).to(device)


    def __len__(self):
        return len(self.detection_list)

    def __getitem__(self, idx):
        """
        bbox: [batch_id, min_x, min_y, max_x, max_y]
        :param idx:
        :return:
        """
        item = {}
        img_idx = int(self.detection_list[idx][0].item())
        img_bgr = self.img_bgr_list[img_idx]
        img_rgb = img_bgr[:, :, ::-1]
        img_h, img_w, _ = img_rgb.shape
        focal_length = estimate_focal_length(img_h, img_w)

        bbox = self.detection_list[idx][1:5]
        norm_img, center, scale, crop_ul, crop_br, _ = process_image(img_rgb, bbox)

        item["norm_img"] = norm_img
        item["center"] = center
        item["scale"] = scale
        item["crop_ul"] = crop_ul
        item["crop_br"] = crop_br
        item["img_h"] = img_h
        item["img_w"] = img_w
        item["focal_length"] = focal_length
        return item

    def eval_EHF(self, pred_results, gt_ply_path):
        eval_result = {'pa_mpjpe': [], 'pa_mpvpe': [], 'pck50': []}
        batchsize = pred_results[0].shape[0]
        if batchsize > 1:
            assert isinstance(gt_ply_path, list) and len(gt_ply_path) == batchsize
            gt_ply_path_list = gt_ply_path
        else:
            gt_ply_path_list = [gt_ply_path]
        pose, betas, camera_translation, reprojection_loss = pred_results
        mesh_out = self.smplx(betas=betas,
                              body_pose=pose[:, 3:66],
                              global_orient=pose[:, :3],
                              trans=camera_translation).vertices.detach().cpu().numpy()
        for idx, gt_ply in enumerate(gt_ply_path_list):
            mesh_gt = load_ply(gt_ply)
            mesh_gt = np.dot(self.cam_param['R'], mesh_gt.transpose(1, 0)).transpose(1, 0)
            # MPJPE from body joints
            joint_gt_body = np.dot(self.smplx.J_regressor.cpu().numpy(), mesh_gt)[:22]
            joint_out_body = np.dot(self.smplx.J_regressor.cpu().numpy(), mesh_out[idx])[:22]
            joint_out_body_align = rigid_align(joint_out_body, joint_gt_body)
            eval_result['pa_mpjpe'].append(
                np.sqrt(np.sum((joint_out_body_align - joint_gt_body) ** 2, axis=1)).mean() * 1000)

            mesh_out_align = rigid_align(mesh_out[idx], mesh_gt)
            eval_result['pa_mpvpe'].append(
                np.sqrt(np.sum((mesh_out_align - mesh_gt) ** 2, axis=1)).mean() * 1000)

            diff_3d = np.sqrt(np.sum((joint_out_body_align - joint_gt_body) ** 2, axis=1))
            eval_result['pck50'].append(np.mean(diff_3d < (50/1000.)) * 100)
            #joint_out_body_align = joint_out_body - joint_out_body[0, None, :] + \
            #                       joint_gt_body[0, None, :]
            #eval_result['mpjpe_body'].append(
            #    np.sqrt(np.sum((joint_out_body_align - joint_gt_body) ** 2, axis=1)).mean() * 1000)


        return eval_result

    def print_eval_result(self, eval_result):
        print('PA MPJPE: %.2f mm' % np.mean(eval_result['pa_mpjpe']))
        print('PA MPVPE: %.2f mm' % np.mean(eval_result['pa_mpvpe']))
        print('PCK@50mm: %.2f' % np.mean(eval_result['pck50']))
