import argparse
import torch
import numpy as np
import os.path as osp
import smplx
from scipy import stats
from tqdm import tqdm

from manifm.angles_utils import so3_to_axis_angle
from manifm.eval_utils import load_model
from experiment.utils import calculate_dnn, calculate_frechet_distance, calculate_apd


n_joints = 21


def eval_poses(cfg, poses):
    device = torch.device(args.device)
    bs, _ = poses.shape

    body_model = smplx.create(
        args.bm_path,
        model_type='smpl',
        gender='neutral',
        num_betas=10,
        batch_size=bs
    ).to(device)

    poses = so3_to_axis_angle(poses.reshape(-1, n_joints, 3, 3)) # [bs, 21, 3, 3] -> [bs, 21, 3]
    body_pose = torch.cat([poses, torch.zeros((bs, 2, 3)).to(poses)], dim=1).reshape(bs, -1) # [bs, 21, 3] -> [bs, 69]

    cdata = np.load(osp.join('examples', 'gt_mean_var_v2.npz'))
    muj, covj = cdata['muj'], cdata['covj']

    body_params = {}
    body_params['body_pose'] = body_pose.clone().detach().float().to(
        device=device)
    body_params['betas'] = torch.zeros((bs, 10)).to(device=device)
    body_out = body_model(return_verts=True, **body_params)
    pts = body_out.joints[:, :22].detach().cpu().numpy()
    pts_r = pts.reshape(-1, 66)

    apd = calculate_apd(pts)
    mu_wj, cov_wj = np.mean(pts_r, axis=0), np.cov(pts_r, rowvar=False)

    fid = calculate_frechet_distance(muj, covj, mu_wj, cov_wj)

    dnn = calculate_dnn(cfg, poses)

    return fid, apd * 100, dnn


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Sample and evaluate poses using RFM")

    parser.add_argument("-ns", "--num_samples", type=int, default=500, help="Number of generated samples")
    parser.add_argument("-md", "--model_dir", type=str,
                        default="checkpoints/epoch-009_step-0_loss-0.0000.ckpt",
                        help="Path to the pretrained RFM model")
    parser.add_argument("-bp", "--bm_path", type=str, default="data/human_model_files/",
                        help="SMPL body model path")
    parser.add_argument("-nsteps", "--num_steps", type=int, default=1000, help="Number of optimization steps")
    parser.add_argument("-px", "--projx", action="store_true")
    parser.add_argument("-lc", "--local_coords", action="store_true")
    parser.add_argument("-so", "--solver", type=str, default="euler", help="ODE solver: euler, midpoint, rk4")
    parser.add_argument("-d", "--device", type=str, default="cuda:0", help="CUDA or CPU")


    args = parser.parse_args()

    device = args.device
    n_samples = args.num_samples
    n_steps = args.num_steps
    num_exp = 20

    cfg, model = load_model(args.model_dir)
    model = model.to(device=device)

    metrics = []
    for seed in tqdm(range(num_exp)):
        sampled_poses = model.sample(n_samples, device, n_steps, args.projx, args.local_coords, args.solver)
        metrics.append(eval_poses(cfg, sampled_poses))

    metrics = np.array(metrics)
    means = np.mean(metrics, axis=0)
    sems = stats.sem(metrics, axis=0)
    h = sems * stats.t.ppf(0.975, num_exp-1)

    print(f"FID: {means[0]:.3f} ± {h[0]:.3f}")
    print(f"APD: {means[1]:.3f} ± {h[1]:.3f} (in cm)")
    print(f"dNN: {means[2]:.3f} ± {h[2]:.3f} (in rad)")
