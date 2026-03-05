import os
import argparse
import glob
import random

import numpy as np
from tqdm import tqdm

import torch
import smplx
from geoopt.tensor import ManifoldParameter
from geoopt.optim import RiemannianAdam

from manifm.eval_utils import load_model
from manifm.amass_dataset import get_product_manifold
from manifm.solvers import projx_integrator
from manifm.angles_utils import axis_angle_to_so3, matrix_to_axis_angle


def geodesic_loss(preds, targets, eps=1e-6):
    R_rel = preds.mT @ targets
    trace = torch.einsum("...ii", R_rel)
    theta = torch.acos(torch.clamp((trace - 1) / 2, -1+eps, 1-eps))
    return theta.mean()


def rdflow_opt(smpl_model, manifold, dim, vecfield, device, joints3d, seed, lr, N, temp_weight=1e-2):
    n_samples = len(joints3d)
    vecfield_compiled = torch.compile(vecfield)

    num_steps = 100
    time_grid = torch.linspace(0, 1, num_steps+1).to(device)

    torch.manual_seed(seed)
    init_noise = (
        manifold.random_combined(n_samples, dim)
        .reshape(n_samples, dim)
        .to(device)
    )

    t_pose = axis_angle_to_so3(torch.zeros([1, 21, 3])).reshape(1, -1).repeat(n_samples, 1).to(device)
    x0_ = projx_integrator(
        manifold,
        vecfield_compiled,
        t_pose,
        time_grid,
        method="euler",
        projx=False,
        local_coords=False,
    )[0][-1]
    x0 = x0_ * np.sqrt(0.75) + init_noise * np.sqrt(0.25)
    x0 = manifold.projx(x0)

    x0 = ManifoldParameter(x0, manifold=manifold, requires_grad=True)
    optimizer = RiemannianAdam([x0], lr=lr, weight_decay=1e-4)
    mse_loss = torch.nn.MSELoss()

    for i in range(N):
        optimizer.zero_grad()
        traj = projx_integrator(
            manifold,
            vecfield_compiled,
            x0,
            time_grid,
            method="euler",
            projx=False,
            local_coords=False,
            enable_grad=True
        )[0]
        x1 = manifold.projx(traj[-1])

        temp_loss = geodesic_loss(
            x1.view(n_samples, -1, 3, 3)[:-1],
            x1.view(n_samples, -1, 3, 3)[1:],
        )

        x1_smpl = smpl_model(body_pose=matrix_to_axis_angle(x1.view(-1, 21, 3, 3)).view(-1, dim//3),
                            betas=torch.zeros((n_samples, 10), device=device))
        data_loss = mse_loss(x1_smpl.joints[:, :22], joints3d)

        traj = traj.reshape(num_steps+1, n_samples, -1, 3, 3)
        running_cost = (3 - torch.einsum("...ii", traj)).sum(-1)
        running_cost = running_cost.mean(1).sum() / (num_steps+1)


        loss = data_loss + temp_weight * temp_loss + 1e-5 * running_cost
        loss.backward()
        optimizer.step()

        if i % 25 == 0 or i == N-1:
            print(f"Step {i}, Loss: {loss.item()}, Temp Loss: {temp_loss.item()}, Running Cost: {running_cost.item()}")

    return x1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-md", "--model_dir", type=str,
                        default="checkpoints/epoch-009_step-0_loss-0.0000.ckpt",
                        help="Path to the pretrained RFM model")
    parser.add_argument("--lr", type=float, default=0.25, help="learning rate")
    parser.add_argument("--noise", type=float, default=0.04, help="noise standard deviation")
    parser.add_argument("--dataset", type=str, default="AMASS", choices=["AMASS", "HPS"])
    parser.add_argument("--num_steps", type=int, default=300, help="number of optimization steps")
    parser.add_argument("--temp_weight", type=float, default=1e-2, help="temporal consistency weight")

    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    std = args.noise

    if args.dataset == "AMASS":
        clean_seqs = sorted(glob.glob("data/HumanEva_60frame/*/*.npz"))
    elif args.dataset == "HPS":
        data_dir = 'data/HPS_60frame'
        clean_seqs = []
        for root, dirs, files in os.walk(data_dir):
            npz_files = [file for file in files if file.endswith('.npz')]
            if len(npz_files) >= 50:  # Sample 50 if a directory has 50+ sequences
                selected_files = random.sample(npz_files, 50)
                for file in selected_files:
                    full_file_path = os.path.join(root, file)
                    clean_seqs.append(full_file_path)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    print(len(clean_seqs))

    cfg, model = load_model(args.model_dir)
    model.eval().to(device)
    manifold, dim = get_product_manifold(cfg)

    all_batch_results = {}
    for seed, seq in tqdm(enumerate(clean_seqs), total=len(clean_seqs)):
        cdata = np.load(seq)["pose_body"].astype(np.float32)
        cdata = torch.from_numpy(cdata).to(device)
        batch_size = len(cdata)
        body_model = smplx.create(
            'data/human_model_files',
            model_type='smplx',
            gender='neutral',
            batch_size=batch_size,
        ).to(device)
        betas = torch.zeros((batch_size, 10)).to(device)

        with torch.no_grad():
            smpl_gt = body_model(body_pose=cdata, betas=betas)

        joints3d = smpl_gt.joints[:, :22]
        torch.manual_seed(seed)
        noisy_joints3d = joints3d + std * torch.randn(*joints3d.shape, device=joints3d.device)
        pred = rdflow_opt(body_model, manifold, dim, model.vecfield, device, noisy_joints3d, seed, args.lr, args.num_steps, args.temp_weight)

        with torch.no_grad():
            smpl_pred = body_model(body_pose=matrix_to_axis_angle(pred.view(-1, 21, 3, 3)).view(-1, dim//3), betas=betas)

        joint_error = smpl_pred.joints[:, :22] - smpl_gt.joints[:, :22]
        vert_error = smpl_pred.vertices - smpl_gt.vertices
        MPJPE = torch.mean(torch.sqrt(torch.sum(joint_error * joint_error, dim=2)), dim=1) * 1000
        MPVPE = torch.mean(torch.sqrt(torch.sum(vert_error * vert_error, dim=2)), dim=1) * 1000
        results_dict = {'MPJPE': MPJPE.detach().cpu().numpy(),
                        'MPVPE': MPVPE.detach().cpu().numpy(),}

        for key, value in results_dict.items():
            if key in all_batch_results:
                all_batch_results[key] = np.concatenate([all_batch_results[key], value])
            else:
                all_batch_results[key] = value
    
    for key, value in all_batch_results.items():
        average_value = np.nanmean(value)
        print(f"The average of {key} is {average_value}")
