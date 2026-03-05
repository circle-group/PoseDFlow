import argparse

import numpy as np
from tqdm import tqdm

import torch
import smplx
from geoopt.tensor import ManifoldParameter
from geoopt.optim import  RiemannianAdam

from manifm.eval_utils import load_model
from manifm.amass_dataset import AMASSDataset, get_product_manifold
from manifm.solvers import projx_integrator
from manifm.angles_utils import axis_angle_to_so3, so3_to_axis_angle
from experiment.inverse_utils import BodyPartIndices, Evaler, average_pairwise_distance


def geodesic_loss(preds, targets, eps=1e-6):
    R_rel = preds.mT @ targets
    trace = torch.einsum("...ii", R_rel)
    theta = torch.acos(torch.clamp((trace - 1) / 2, -1+eps, 1-eps))
    return theta.mean()


def rdflow_opt(manifold, dim, vecfield, device, target_pts, mask, seed, lr, N):
    n_samples = len(target_pts)
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
    y0 = t_pose * (~mask) + target_pts * mask
    x0_ = projx_integrator(
        manifold,
        vecfield_compiled,
        y0,
        time_grid,
        method="euler",
        projx=False,
        local_coords=False,
    )[0][-1]
    x0 = x0_ * np.sqrt(0.25) + init_noise * np.sqrt(0.75)
    x0 = manifold.projx(x0)
    x0 = ManifoldParameter(x0, manifold=manifold, requires_grad=True)

    optimizer = RiemannianAdam([x0], lr=lr, weight_decay=1e-4)

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

        inv_loss = geodesic_loss(
            x1[mask].view(n_samples, -1, 3, 3),
            target_pts[mask].view(n_samples, -1, 3, 3)
        )

        traj = traj.reshape(num_steps+1, n_samples, -1, 3, 3)
        running_cost = (3 - torch.einsum("...ii", traj)).sum(-1)
        running_cost = running_cost.mean(1).sum() / (num_steps+1)

        loss = inv_loss + 1e-3 * running_cost
        loss.backward()
        optimizer.step()

        if i % 25 == 0 or i == N-1:
            print(f"Step {i}, Loss: {inv_loss.item()}, Running Cost: {running_cost.item()}")

    x_out = x1 * (~mask) + target_pts * mask
    return x_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-md", "--model_dir", type=str,
                        default="checkpoints/epoch-009_step-0_loss-0.0000.ckpt",
                        help="Path to the pretrained RFM model")
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument("--num_steps", type=int, default=200, help="number of optimization steps")
    parser.add_argument("--occ_type", type=str, default="left_leg", choices=['left_leg', 'right_leg', 'left_arm',
                                                                         'right_arm', 'trunk', 'hands',
                                                                         'legs', 'arms'])
    parser.add_argument("--num_hypo", type=int, default=10, help="number of hypotheses")
    parser.add_argument("--sample", type=int, default=10, help="sampling interval")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    batch_size = 500

    cfg, model = load_model(args.model_dir)
    model.eval().to(device)
    cfg.optim.val_batch_size = batch_size
    manifold, dim = get_product_manifold(cfg)

    test_dataset = AMASSDataset(cfg, subset='test', sample_interval=args.sample)
    test_loader = test_dataset.get_loader(drop_last=True)

    mask_joints = BodyPartIndices.get_indices(args.occ_type)
    mask = torch.ones((batch_size, dim), device=device, dtype=torch.bool)
    mask_indices = torch.tensor(mask_joints, dtype=torch.long).view(-1, 1) * 9 + torch.arange(9).view(1, -1)
    mask_indices = mask_indices.flatten().to(device)
    mask[:, mask_indices] = 0

    body_model = smplx.create(
        'data/human_model_files',
        model_type='smplx',
        gender='neutral',
        batch_size=batch_size,
    ).to(device)

    body_model_apd = smplx.create(
        'data/human_model_files',
        model_type='smplx',
        gender='neutral',
        batch_size=args.num_hypo,
    ).to(device)

    betas = torch.zeros((batch_size, 10)).to(device)
    evaler = Evaler(body_model, part=args.occ_type)

    all_results = []
    joint_idx = evaler.joint_idx

    for batch in tqdm(test_loader):
        pred_poses = []
        batch = batch.to(device)

        for seed in tqdm(range(args.num_hypo)):
            pred = rdflow_opt(manifold, dim, model.vecfield, device, batch, mask, seed, args.lr, args.num_steps)
            pred_poses.append(pred)

        pred_poses = torch.stack(pred_poses, dim=1)
        pred_poses = so3_to_axis_angle(pred_poses.view(-1, args.num_hypo, 21, 3, 3)).view(-1, args.num_hypo, dim//3)
        batch = so3_to_axis_angle(batch.view(-1, 21, 3, 3)).view(-1, dim//3)

        batch_APD = []
        with torch.no_grad():
            for b in range(batch_size):
                samples = pred_poses[b]
                body_out = body_model_apd(body_pose=samples, betas=betas[:args.num_hypo])
                joints3d = body_out.joints
                body_joints3d = joints3d[:, joint_idx, :]
                APD = average_pairwise_distance(body_joints3d) * 100.0
                batch_APD.append(APD)

        all_results.append({'APD': batch_APD})
        eval_results = evaler.multi_eval_bodys_all(pred_poses, batch)  # [batch_size, ]
        all_results.append(eval_results)

    collected_dict = {}
    for single_process_results in all_results:
        for key, value in single_process_results.items():
            if key not in collected_dict:
                collected_dict[key] = []
            collected_dict[key].extend(value)

    for key, value in collected_dict.items():
        average_value = np.mean(np.array(value))
        print(f"The average of {key} is {average_value}")
