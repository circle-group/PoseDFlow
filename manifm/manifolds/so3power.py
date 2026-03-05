import math
import geoopt
import torch
from manifm.angles_utils import axis_angle_to_so3

class SO3(geoopt.Manifold):
    def __init__(self):
        super().__init__()
        self.name = "SO3"
        self.ndim = 2

    def _check_point_on_manifold(self, x, *, atol=0.00001, rtol=0.00001):
        raise NotImplementedError
    
    def _check_vector_on_tangent(self, x, u, *, atol=0.00001, rtol=0.00001):
        raise NotImplementedError
    
    def egrad2rgrad(self, x, u):
        grad = self.proju(x, u)
        return grad
    
    def expmap(self, x, u):
        
        return x @ torch.matrix_exp(u)
        
    def inner(self, x, u, v = None, *, keepdim=False):

        if v is None:
            v = u

        return torch.sum(u * v, dim=(-2, -1), keepdim=keepdim)
    
    def proju(self, x, u):

        # Compute the relative rotation matrix
        relative_rotation = self._relative_rotation(x, u)

        # Extract the skew-symmetric part to ensure it belongs to so(3)
        return self._skew_symmetric(relative_rotation)
    
    def projx(self, x):
        q, r = torch.linalg.qr(x)

        q = q.contiguous()
        r = r.contiguous()

        s = torch.diagonal(r, dim1=-2, dim2=-1).clone()
        s[s == 0] = 1
        s = s / torch.abs(s)
        s = s.unsqueeze(-1)

        q = q * s.transpose(-2, -1)
        return q

    def retr(self, x, u):
        return self.expmap(x, u)
    
    def transp(self, x, y, v):
        return v

    def retr_transp(self, x, u, v):
        return self.retr(x, u), v

    def logmap(self, x, y):

        # Calculate the relative rotation matrix
        relative_rotation = self._relative_rotation(x, y)

        # Compute the matrix logarithm to obtain the tangent vector
        tangent_vector = self._matrix_log(relative_rotation)

        return tangent_vector

    def random(self, *size, dtype=None, device=None):
        batch_size = size[0][0]

        # Sample from normal distribution in tangent space (axis-angle)
        noise = torch.randn(batch_size, 3, dtype=dtype, device=device)

        # Map to SO(3) via exponential map (axis-angle to rotation matrix)
        return axis_angle_to_so3(noise)  # output: (batch_size, 3, 3)

    def _skew_symmetric(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the skew-symmetric part of a square matrix.

        Args:
            x (torch.Tensor): A square matrix represented as a PyTorch tensor.

        Returns:
            torch.Tensor: The skew-symmetric part of the input matrix.
        """
        return 0.5 * (x - x.mT)

    def _relative_rotation(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the relative rotation between two rotation matrices.

        Args:
            x (torch.Tensor): A tensor representing the first rotation matrix.
                              Expected shape is (..., 3, 3).
            y (torch.Tensor): A tensor representing the second rotation matrix.
                              Expected shape is (..., 3, 3).

        Returns:
            torch.Tensor: A tensor representing the relative rotation matrix.
                          The shape of the output is (..., 3, 3).
        """
        return x.mT @ y

    def _matrix_log(self, R):
        """
        Compute the matrix logarithm of a given square matrix A.
        
        Args:
            A (torch.Tensor): A square matrix of shape (n, n) for which the matrix
                logarithm is to be computed.
        Returns:
            torch.Tensor: The matrix logarithm of A, of shape (n, n).
        Raises:
            ValueError: If the matrix is not diagonalizable (i.e., its eigenvectors
                are not invertible).
        """
        theta = torch.acos(torch.clamp((torch.trace(R) - 1) / 2, -1.0 + 1e-6, 1.0 - 1e-6))
        log_R = (theta / (2 * torch.sin(theta))) * (R - R.mT)
        return log_R


class SO3Power(geoopt.ProductManifold):
    def __init__(self, n_joints: int):
        self.name = "SO3 Power"
        self.ndim = 9 * n_joints
        manifolds_with_shape = [(SO3(), (3, 3)) for _ in range(n_joints)]
        super().__init__(*manifolds_with_shape)
        self.n_joints = n_joints

    def uniform_logprob(self, x):
        return torch.full_like(x[..., 0], -self.n_joints * math.log(8 * math.pi * math.pi))

    def base_logprob(self, *args, **kwargs):
        return self.uniform_logprob(*args, **kwargs)
