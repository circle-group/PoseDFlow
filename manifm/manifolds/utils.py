"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import torch


def geodesic(product_manifold, start_point, end_point):
    shooting_tangent_vec = product_manifold.logmap(start_point, end_point)

    def path(t):
        """Generate parameterized function for geodesic curve.
        Parameters
        ----------
        t : array-like, shape=[n_points,]
            Times at which to compute points of the geodesics.
        """
        tangent_vecs = torch.einsum("i,...k->...ik", t, shooting_tangent_vec)
        points_at_time_t = product_manifold.expmap(start_point.unsqueeze(-2), tangent_vecs)
        return points_at_time_t

    return path
