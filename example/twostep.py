"""Standard/Improved MeanFlow followed by frozen-MeanFlow DP refinement."""

from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Iterator, Optional, Tuple

import torch
from torch import Tensor

from deep_particle import NearIdentityDP
from meanflow import MeanflowNet, improved_meanflow_loss, ordinary_meanflow_loss, sample_meanflow
from twostep_config import TwoStepConfig
from twostep_ot import exact_emd_coupling, pairwise_squared_cost
from tqdm import tqdm

ORIGINAL_TIME_SAMPLER = False

def sample_meanflow_times(
    like: Tensor,
    logit_mean: float,
    logit_std: float,
    same_time_probability: float = 0.5,
) -> tuple[Tensor, Tensor, Tensor]:
    """Sample the paper's ordered logit-normal ``(r, t)`` pairs.

    Two times are drawn independently, the smaller is assigned to ``r`` and
    the larger to ``t``, and ``r=t`` is imposed on the requested portion of
    samples.  The returned mask identifies those diagonal samples.
    """
    if logit_std <= 0:
        raise ValueError("logit_std must be positive.")
    if not 0.0 <= same_time_probability <= 1.0:
        raise ValueError("same_time_probability must lie in [0, 1].")
    first = torch.sigmoid(logit_mean + logit_std * torch.randn_like(like))
    second = torch.sigmoid(logit_mean + logit_std * torch.randn_like(like))
    r, t = torch.minimum(first, second), torch.maximum(first, second)
    diagonal_count = round(same_time_probability * r.numel())
    same_time = torch.zeros_like(r, dtype=torch.bool)
    if diagonal_count:
        diagonal_indices = torch.randperm(r.numel(), device=r.device)[:diagonal_count]
        same_time.view(-1)[diagonal_indices] = True
    r = torch.where(same_time, t, r)
    return r, t, same_time


class TwoStepDiffusion:
    """Stage I Meanflow followed by Stage II near-identity DP refinement."""
    def __init__(self, config: TwoStepConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.meanflow = MeanflowNet(config.state_dim, config.condition_dim, config.mf_width, config.mf_depth).to(self.device)
        self.dp = NearIdentityDP(config.state_dim, config.condition_dim, config.dp_width, config.dp_depth, config.near_identity_scale).to(self.device)
        self.meanflow_history: dict[str, list[float]] = {}
        self.dp_history: dict[str, list[float]] = {}

    def _split_batch(self, batch: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        source, reference = (torch.as_tensor(value, dtype=torch.float32, device=self.device) for value in batch)
        expected = self.config.state_dim + self.config.condition_dim
        if source.ndim != 3 or reference.ndim != 3 or source.shape[-1] != expected or reference.shape[-1] != self.config.state_dim:
            raise ValueError(f"Expected source [B,N,{expected}] and reference [B,N,{self.config.state_dim}], got {tuple(source.shape)} and {tuple(reference.shape)}.")
        return source[..., :self.config.state_dim], source[..., self.config.state_dim:], reference

    @staticmethod
    def _batch(iterator: Iterator) -> Tuple[Tensor, Tensor]:
        try:
            return next(iterator)
        except StopIteration as error:
            raise RuntimeError("Dataset must yield training batches indefinitely.") from error

    def _meanflow_scheduler(self, optimizer: torch.optim.Optimizer, steps: int):
        """Stage-I cosine decay with a non-vanishing learning-rate floor."""
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(steps, 1),
            eta_min=self.config.mf_lr * self.config.mf_min_lr_ratio,
        )

    def train_meanflow(
        self,
        dataset: Iterable,
        steps: Optional[int] = None,
    ) -> list[float]:
        """Train the configured MeanFlow objective and retain the last state."""
        for parameter in self.meanflow.parameters():
            parameter.requires_grad_(True)
        optimizer, history, iterator = torch.optim.Adam(self.meanflow.parameters(), lr=self.config.mf_lr), [], iter(dataset)
        self.meanflow.train()
        training_steps = self.config.mf_steps if steps is None else steps
        if training_steps < 0:
            raise ValueError("steps must be non-negative.")
        scheduler = self._meanflow_scheduler(optimizer, training_steps)
        self.meanflow_history = {
            "adaptive_loss": [],
            "raw_mse": [],
            "raw_mse_same_time": [],
            "raw_mse_different_time": [],
            "learning_rate": [],
        }
        loss_function = improved_meanflow_loss if self.config.improved_meanflow else ordinary_meanflow_loss
        for _ in tqdm(range(training_steps)):
            source, condition, reference = self._split_batch(self._batch(iterator))
            
            if ORIGINAL_TIME_SAMPLER:
                r, t, same_time = sample_meanflow_times(
                    source[..., :1],
                    self.config.time_logit_mean,
                    self.config.time_logit_std,
                )
            else:
                t = torch.sigmoid(self.config.time_logit_mean + self.config.time_logit_std * torch.randn_like(source[..., :1]))
                r = t * torch.sigmoid(self.config.time_logit_mean + self.config.time_logit_std * torch.randn_like(t))
                same_time = torch.rand_like(r) < 0.5
                r = torch.where(same_time, t, r)
            # In the paper's time convention the terminal/reference cloud is
            # x_0 and the source/prior cloud is x_1.
            z, velocity = (1 - t) * reference + t * source, source - reference
            loss, details = loss_function(
                self.meanflow,
                z,
                r,
                t,
                condition,
                velocity,
                self.config.mf_adaptive_power,
                self.config.mf_adaptive_epsilon,
                same_time=same_time,
                adaptive_weight_min=self.config.mf_adaptive_weight_min,
                adaptive_weight_max=self.config.mf_adaptive_weight_max,
                return_details=True,
            )
            loss_value = float(loss.detach().cpu())
            for name in ("adaptive_loss", "raw_mse", "raw_mse_same_time", "raw_mse_different_time"):
                self.meanflow_history[name].append(float(details[name].detach().cpu()))
            self.meanflow_history["learning_rate"].append(float(optimizer.param_groups[0]["lr"]))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            history.append(loss_value)
        return history

    @torch.no_grad()
    def meanflow_sample(self, source: Tensor, condition: Tensor) -> Tensor:
        source = torch.as_tensor(source, dtype=torch.float32, device=self.device)
        condition = torch.as_tensor(condition, dtype=torch.float32, device=self.device)
        return sample_meanflow(self.meanflow, source, condition)

    def _refine(self, meanflow_output: Tensor, condition: Tensor) -> Tensor:
        return self.dp(meanflow_output, condition)

    def train_dp(
        self,
        dataset: Iterable,
        steps: Optional[int] = None,
    ) -> list[float]:
        """Train DP at constant learning rate with the last MeanFlow frozen."""
        self.meanflow.eval()
        for parameter in self.meanflow.parameters():
            parameter.requires_grad_(False)
        for parameter in self.dp.parameters():
            parameter.requires_grad_(True)
        optimizer = torch.optim.Adam(self.dp.parameters(), lr=self.config.dp_lr)
        history, iterator, cached, gamma = [], iter(dataset), None, None
        self.dp.train()
        training_steps = self.config.dp_steps if steps is None else steps
        if training_steps < 0:
            raise ValueError("steps must be non-negative.")
        self.dp_history = {
            "total_loss": [],
            "ot_loss": [],
            "residual_loss": [],
            "weighted_residual_loss": [],
            "learning_rate": [],
        }
        for step in tqdm(range(training_steps)):
            if cached is None or step % self.config.batch_refresh == 0:
                source, condition, reference = self._split_batch(self._batch(iterator))
                cached, gamma = (self.meanflow_sample(source, condition), condition, reference), None
            mf_output, condition, reference = cached
            refined = self._refine(mf_output, condition)
            if gamma is None or step % self.config.gamma_refresh == 0:
                gamma = exact_emd_coupling(refined, reference)
            cost = pairwise_squared_cost(refined, reference)
            ot_loss = (cost * gamma).sum(dim=(-2, -1)).mean()
            residual_loss = (refined - mf_output).square().sum(dim=-1).mean()
            weighted_residual_loss = self.config.residual_weight * residual_loss
            loss = ot_loss + weighted_residual_loss
            loss_value = float(loss.detach().cpu())
            self.dp_history["total_loss"].append(loss_value)
            self.dp_history["ot_loss"].append(float(ot_loss.detach().cpu()))
            self.dp_history["residual_loss"].append(float(residual_loss.detach().cpu()))
            self.dp_history["weighted_residual_loss"].append(float(weighted_residual_loss.detach().cpu()))
            self.dp_history["learning_rate"].append(float(optimizer.param_groups[0]["lr"]))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            history.append(loss_value)
        return history

    def fit(
        self,
        meanflow_dataset: Iterable,
        dp_dataset: Iterable,
        mf_steps: Optional[int] = None,
        dp_steps: Optional[int] = None,
    ) -> dict[str, list[float]]:
        """Train paired-endpoint Stage I, then independent-batch Stage II."""
        return {
            "meanflow": self.train_meanflow(meanflow_dataset, mf_steps),
            "dp": self.train_dp(dp_dataset, dp_steps),
        }

    @torch.no_grad()
    def sample(self, source: Tensor, condition: Optional[Tensor] = None) -> Tensor:
        self.meanflow.eval()
        self.dp.eval()
        if condition is None:
            source = torch.as_tensor(source, dtype=torch.float32, device=self.device)
            expected = self.config.state_dim + self.config.condition_dim
            if source.shape[-1] != expected: raise ValueError(f"Combined input must have {expected} features, got {source.shape[-1]}.")
            condition, source = source[..., self.config.state_dim:], source[..., :self.config.state_dim]
        else:
            source = torch.as_tensor(source, dtype=torch.float32, device=self.device)
            condition = torch.as_tensor(condition, dtype=torch.float32, device=self.device)
        return self._refine(self.meanflow_sample(source, condition), condition)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": 3,
            "config": asdict(self.config),
            "meanflow": self.meanflow.state_dict(),
            "dp": self.dp.state_dict(),
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path, device: Optional[str] = None) -> "TwoStepDiffusion":
        checkpoint = torch.load(path, map_location=device or "cpu")
        config_values = dict(checkpoint["config"])
        if "min_lr_ratio" in config_values and "mf_min_lr_ratio" not in config_values:
            config_values["mf_min_lr_ratio"] = config_values.pop("min_lr_ratio")
        config = TwoStepConfig(**config_values)
        if device is not None: config.device = device
        model = cls(config)
        model.meanflow.load_state_dict(checkpoint["meanflow"]); model.dp.load_state_dict(checkpoint["dp"])
        return model
