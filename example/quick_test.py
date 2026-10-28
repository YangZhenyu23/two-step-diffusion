"""Fast smoke tests for standard/Improved MeanFlow + frozen constant-LR DP."""

from pathlib import Path
import tempfile

import numpy as np
import torch

from twostep import TwoStepDiffusion
from twostep_config import TwoStepConfig


class SyntheticParticleBatches:
    """Infinite conditional translations with paired or independent targets."""

    def __init__(self, batch_size: int = 2, particles: int = 64, independent_reference: bool = False):
        self.batch_size = batch_size
        self.particles = particles
        self.independent_reference = independent_reference

    def __iter__(self):
        while True:
            sigma = torch.linspace(0.25, 0.75, self.batch_size).view(self.batch_size, 1, 1)
            condition = sigma.expand(-1, self.particles, -1)
            source = torch.randn(self.batch_size, self.particles, 2)
            reference_base = torch.randn_like(source) if self.independent_reference else source
            reference = reference_base + torch.cat((condition, -0.5 * condition), dim=-1)
            yield torch.cat((source, condition), dim=-1), reference


def run_mode(improved_meanflow: bool) -> None:
    torch.manual_seed(7600)
    config = TwoStepConfig(
        state_dim=2,
        device="cpu",
        mf_width=16,
        mf_depth=2,
        dp_width=16,
        dp_depth=2,
        mf_steps=20,
        dp_steps=20,
        batch_refresh=5,
        gamma_refresh=5,
        improved_meanflow=improved_meanflow,
    )
    model = TwoStepDiffusion(config)
    paired_data = SyntheticParticleBatches()
    independent_data = SyntheticParticleBatches(independent_reference=True)
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        mf_history = model.train_meanflow(paired_data)
        assert len(mf_history) == config.mf_steps
        assert len(model.meanflow_history["raw_mse"]) == config.mf_steps
        assert model.meanflow_history["learning_rate"][0] > model.meanflow_history["learning_rate"][-1]

        combined_input, _ = next(iter(paired_data))
        source = combined_input[..., :2]
        condition = combined_input[..., 2:]
        mf_output = model.meanflow_sample(source, condition)
        assert torch.equal(model.dp(mf_output, condition), mf_output), "DP must initialize as identity."
        frozen_meanflow = {name: parameter.detach().clone() for name, parameter in model.meanflow.named_parameters()}
        dp_history = model.train_dp(independent_data)
        assert len(dp_history) == config.dp_steps
        assert set(model.dp_history["learning_rate"]) == {config.dp_lr}
        assert all(torch.equal(frozen_meanflow[name], parameter) for name, parameter in model.meanflow.named_parameters())

        history = {"meanflow": mf_history, "dp": dp_history}
        prediction = model.sample(combined_input)
        assert prediction.shape == (2, 64, 2)
        assert torch.isfinite(prediction).all()
        assert all(np.isfinite(values).all() for values in history.values())

        saved_model = directory / "twostep.pt"
        model.save(saved_model)
        restored = TwoStepDiffusion.load(saved_model, device="cpu")
        restored_prediction = restored.sample(combined_input)
        assert torch.allclose(prediction, restored_prediction)

    mode_name = "Improved Mean Flow" if improved_meanflow else "standard MeanFlow"
    print(f"Quick test passed ({mode_name}, frozen DP, exact EMD)")
    print(f"MeanFlow: {len(mf_history)} steps, final loss={mf_history[-1]:.6f}")
    print(f"DP:       {len(dp_history)} steps, final loss={dp_history[-1]:.6f}")
    print(f"Prediction shape: {tuple(prediction.shape)}")


def main() -> None:
    run_mode(improved_meanflow=False)
    run_mode(improved_meanflow=True)


if __name__ == "__main__":
    main()
