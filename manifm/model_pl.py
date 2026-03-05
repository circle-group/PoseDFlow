"""Copyright (c) Meta Platforms, Inc. and affiliates."""

from typing import Any
import traceback
import os

import torch
from torchmetrics import MeanMetric, MinMetric
import pytorch_lightning as pl
from torch.func import vjp, jvp, vmap, jacrev
from torchdiffeq import odeint

from manifm.amass_dataset import get_product_manifold
from manifm.ema import EMA
from manifm.model.arch import tMLP, ProjectToTangent, Unbatch
from manifm.manifolds import geodesic

from manifm.solvers import projx_integrator_return_last, projx_integrator
from manifm.body_model import BodyModel


def div_fn(u):
    """Accepts a function u:R^D -> R^D."""
    J = jacrev(u)
    return lambda x: torch.trace(J(x))


def output_and_div(vecfield, x, v=None, div_mode="exact"):
    if div_mode == "exact":
        dx = vecfield(x)
        div = vmap(div_fn(vecfield))(x)
    else:
        dx, vjpfunc = vjp(vecfield, x)
        vJ = vjpfunc(v)[0]
        div = torch.sum(vJ * v, dim=-1)
    return dx, div


class ManifoldFMLitModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.product_manifold, self.dim = get_product_manifold(cfg)

        # Model of the vector field.
        self.model = EMA(
            Unbatch(  # Ensures vmap works.
                ProjectToTangent(  # Ensures we can just use Euclidean divergence.
                    tMLP(  # Vector field in the ambient space.
                        self.dim,
                        d_model=cfg.model.d_model,
                        num_layers=cfg.model.num_layers,
                        actfn=cfg.model.actfn,
                        fourier=cfg.model.get("fourier", None),
                    ),
                    manifold=self.product_manifold,
                    metric_normalize=self.cfg.model.get("metric_normalize", False),
                )
            ),
            cfg.optim.ema_decay,
        )

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_metric = MeanMetric()
        self.val_metric = MeanMetric()
        self.test_metric = MeanMetric()

        # for logging best so far validation accuracy
        self.val_metric_best = MinMetric()

    @property
    def vecfield(self):
        return self.model

    @property
    def device(self):
        return self.model.parameters().__next__().device

    @torch.no_grad()
    def visualize(self, batch, force=False, n_samples=20, out_dir=""):
        if not force and not self.cfg.get("visualize", False):
            return

        output_dir = os.path.join(out_dir, "renderings")

        body_model = BodyModel(
            smpl_folder=self.cfg.body_model.smpl_folder,
            smpl_type=self.cfg.body_model.smpl_type,
            ext=self.cfg.body_model.ext,
            gender=self.cfg.body_model.gender,
        )
        # GT poses
        meshes = body_model(body_pose=batch[:n_samples], n_shapes=n_samples)
        rend = body_model.render_all(meshes)
        body_model.save_renderings(rend, output_dir, "GT")

        # Generated poses
        poses = self.sample_all(n_samples=n_samples, device=self.device)

        meshes = body_model(body_pose=poses[0, ::], n_shapes=n_samples)
        rend = body_model.render_all(meshes)
        body_model.save_renderings(rend, output_dir, "first")

        meshes = body_model(body_pose=poses[-1, ::], n_shapes=n_samples)
        rend = body_model.render_all(meshes)
        body_model.save_renderings(rend, output_dir, "last")

    @torch.no_grad()
    def compute_cost(self, batch):  # it looks like this is not used
        if isinstance(batch, dict):
            x0 = batch["x0"]
        else:
            x0 = (
                self.product_manifold.random_combined(batch.shape[0], self.dim)
                .reshape(batch.shape[0], self.dim)
                .to(batch.device)
            )

        # Solve ODE.
        x1 = odeint(
            self.vecfield,
            x0,
            t=torch.linspace(0, 1, 2).to(x0.device),
            atol=self.cfg.model.atol,
            rtol=self.cfg.model.rtol,
        )[-1]

        x1 = self.product_manifold.projx(x1)

        return self.product_manifold.dist(x0, x1)

    @torch.no_grad()
    def sample(self, n_samples, device, num_steps, projx,
               local_coords, method, x0=None):
        if x0 is None:
            # Sample from base distribution.
            x0 = (
                self.product_manifold.random_combined(n_samples, self.dim)
                .reshape(n_samples, self.dim)
                .to(device)
            )
        x1 = projx_integrator_return_last(
            self.product_manifold,
            self.vecfield,
            x0,
            t=torch.linspace(0, 1, num_steps+1).to(device),
            method=method,
            projx=projx,
            local_coords=local_coords,
            pbar=False,
        )
        return x1 if (projx or local_coords) else self.product_manifold.projx(x1)


    @torch.no_grad()
    def sample_all(self, n_samples, device, x0=None):
        if x0 is None:
            # Sample from base distribution.
            x0 = (
                self.product_manifold.random_combined(n_samples, self.dim)
                .reshape(n_samples, self.dim)
                .to(device)
            )

        # Solve ODE.
        local_coords = self.cfg.get("local_coords", False)
        eval_projx = self.cfg.get("eval_projx", False)

        xs, _ = projx_integrator(
            self.product_manifold,
            self.vecfield,
            x0,
            t=torch.linspace(0, 1, 1001).to(device),
            method="euler",
            projx=eval_projx,
            local_coords=local_coords,
            pbar=False,
        )
        return self.product_manifold.projx(xs)

    @torch.no_grad()
    def compute_exact_loglikelihood(
        self,
        batch: torch.Tensor,
        t1: float = 1.0,
        return_projx_error: bool = False,
        num_steps=1000,
    ):
        """Computes the negative log-likelihood of a batch of data."""

        try:
            nfe = [0]

            div_mode = self.cfg.get("div_mode", "exact")

            with torch.inference_mode(mode=False):
                v = None
                if div_mode == "rademacher":
                    v = (
                        torch.randint(low=0, high=2, size=batch.shape).to(batch)
                        * 2
                        - 1
                    )

                def odefunc(t, tensor):
                    nfe[0] += 1
                    t = t.to(tensor)
                    x = tensor[..., : self.dim]
                    vecfield = lambda x: self.vecfield(t, x)
                    dx, div = output_and_div(
                        vecfield, x, v=v, div_mode=div_mode
                    )

                    if hasattr(self.product_manifold, "logdetG"):

                        def _jvp(x, v):
                            return jvp(self.product_manifold.logdetG, (x,), (v,))[1]

                        corr = vmap(_jvp)(x, dx)
                        div = div + 0.5 * corr.to(div)

                    div = div.reshape(-1, 1)
                    del t, x
                    return torch.cat([dx, div], dim=-1)

                # Solve ODE on the product manifold of data manifold x euclidean.
                state1 = torch.cat(
                    [batch, torch.zeros_like(batch[..., :1])], dim=-1
                )

                local_coords = self.cfg.get("local_coords", False)
                eval_projx = self.cfg.get("eval_projx", False)

                state0 = projx_integrator_return_last(
                    self.product_manifold,
                    odefunc,
                    state1,
                    t=torch.linspace(t1, 0, num_steps + 1).to(batch),
                    method="euler",
                    projx=eval_projx,
                    local_coords=local_coords,
                    pbar=False,
                )

                # log number of function evaluations
                self.log(
                    "nfe",
                    torch.tensor(nfe[0], dtype=torch.float32),
                    prog_bar=True,
                    logger=True,
                )

                x0, logdetjac = state0[..., : self.dim], state0[..., -1]
                x0_ = x0
                x0 = self.product_manifold.projx(x0)

                # log how close the final solution is to the manifold.
                integ_error = (
                    (x0[..., : self.dim] - x0_[..., : self.dim]).abs().max()
                )
                self.log("integ_error", integ_error)

                logp0 = self.product_manifold.base_logprob(x0)
                logp1 = logp0 + logdetjac

                if self.cfg.get("normalize_loglik", False):
                    logp1 = logp1 / self.dim

                # Mask out those that left the manifold
                masked_logp1 = logp1

                if return_projx_error:
                    return logp1, integ_error
                else:
                    return masked_logp1
        except:
            traceback.print_exc()
            return torch.zeros(batch.shape[0]).to(batch)

    def loss_fn(self, batch: torch.Tensor):
        return self.rfm_loss_fn(batch)

    def rfm_loss_fn(self, batch: torch.Tensor):
        if isinstance(batch, dict):
            x0 = batch["x0"]
            x1 = batch["x1"]
        else:
            x1 = batch
            x0 = self.product_manifold.random_combined(x1.shape[0], self.dim).to(x1)

        N = x1.shape[0]

        t = torch.rand(N).reshape(-1, 1).to(x1)

        def cond_u(x0, x1, t):
            path = geodesic(self.product_manifold, x0, x1)
            x_t, u_t = jvp(path, (t,), (torch.ones_like(t).to(t),))
            return x_t, u_t

        x_t, u_t = vmap(cond_u)(x0, x1, t)
        x_t = x_t.reshape(N, self.dim)
        u_t = u_t.reshape(N, self.dim)

        diff = self.vecfield(t, x_t) - u_t
        return self.product_manifold.inner(x_t, diff).mean() / self.dim

    def training_step(self, batch: Any, batch_idx: int):
        loss = self.loss_fn(batch)

        if torch.isfinite(loss):
            # log train metrics
            self.log("train/loss", loss, on_step=True, on_epoch=True)
            self.train_metric.update(loss)
        else:
            # skip step if loss is NaN.
            print(f"Skipping iteration because loss is {loss.item()}.")
            return None
        
        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss}

    def on_training_epoch_end(self):
        # `outputs` is a list of dicts returned from `training_step()`
        self.train_metric.reset()

    def validation_step(self, batch: Any, batch_idx: int):
        if isinstance(batch, dict):
            x1 = batch["x1"]
        else:
            x1 = batch

        loss = self.loss_fn(batch)
        batch_size = x1.shape[0]

        self.log(
            "val/loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.val_metric.update(loss)
        return {"loss": loss}

    def on_validation_epoch_end(self):
        val_loss = (
            self.val_metric.compute()
        )  # get val accuracy from current epoch
        self.val_metric_best.update(val_loss)
        self.log(
            "val/loss_best",
            self.val_metric_best.compute(),
            on_epoch=True,
            prog_bar=True,
        )
        self.val_metric.reset()

    def test_step(self, batch: Any, batch_idx: int):
        if isinstance(batch, dict):
            x1 = batch["x1"]
        else:
            x1 = batch

        loss = self.rfm_loss_fn(batch)
        batch_size = x1.shape[0]

        self.log("test/loss", loss, batch_size=batch_size)
        self.test_metric.update(loss)
        return {"loss": loss}

    def on_test_epoch_end(self):
        avg_test_loss = self.test_metric.compute()
        self.log("test/loss_avg", avg_test_loss, prog_bar=True)
        self.test_metric.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.optim.lr,
            weight_decay=self.cfg.optim.wd,
            eps=self.cfg.optim.eps,
        )

        if self.cfg.optim.get("scheduler") == "reduce_lr_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                patience=self.cfg.optim.patience,
                factor=0.5,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "step",
                    "frequency": self.cfg.val_every,
                },
            }
        elif self.cfg.optim.get("scheduler") == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.cfg.optim.num_iterations,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }
        else:
            return {
                "optimizer": optimizer,
            }

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if isinstance(self.model, EMA):
            self.model.update_ema()
