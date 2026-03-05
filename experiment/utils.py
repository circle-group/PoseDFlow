import numpy as np
from scipy import linalg
from scipy.spatial import distance
import torch
from tqdm import tqdm

from manifm.angles_utils import axis_angle_to_quaternion, so3_to_quaternion
from manifm.amass_dataset import AMASSDataset


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative dataset set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative dataset set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            # raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)

def calculate_apd(pts):
    # pts: (n, j, 3)
    _, nj, _ = pts.shape
    res = distance.pdist(pts.reshape(-1, nj*3), lambda u, v: np.sqrt(np.sum(np.power((u.reshape(nj, 3)-v.reshape(nj, 3)), 2), axis=1)).mean())  # Pairwise distances
    res = res.mean()
    return res

def cal_intrinsic_geo(a, b):
    ndim = len(a.shape)
    a_conj = torch.cat((a[..., 0].unsqueeze(-1), -a[..., 1].unsqueeze(-1), -a[..., 2].unsqueeze(-1), -a[..., 3].unsqueeze(-1)), dim=ndim-1)
    w = quaternion_hamilton_product(a_conj, b)[..., 0]
    prod = torch.abs(w)
    prod = torch.clamp(prod, max=1)
    res = torch.acos(prod)
    return res

def quaternion_hamilton_product(a, b):

    ndim = len(a.shape)

    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]

    res = torch.concat((
        (aw * bw - ax * bx - ay * by - az * bz).unsqueeze(-1),
        (aw * bx + ax * bw + ay * bz - az * by).unsqueeze(-1),
        (aw * by - ax * bz + ay * bw + az * bx).unsqueeze(-1),
        (aw * bz + ax * by - ay * bx + az * bw).unsqueeze(-1)
    ), dim=ndim-1)

    return res

def calculate_dnn(cfg, body_pose, device='cuda'):
    training_dataset = AMASSDataset(cfg, 'train')
    training_dataloader = training_dataset.get_loader()
    min_distances = torch.full((len(body_pose),), float('inf'), device=device)
    
    body_pose = axis_angle_to_quaternion(body_pose.view(-1, 21, 3)).unsqueeze(1)

    for batch in tqdm(training_dataloader):
        batch = batch.to(device).view(-1, 21, 3, 3)
        batch_quat = so3_to_quaternion(batch).unsqueeze(0)

        dists = cal_intrinsic_geo(body_pose, batch_quat)
        min_dists = dists.mean(-1).min(-1).values
        min_distances = torch.minimum(min_distances, min_dists)

    return min_distances.mean().item()
