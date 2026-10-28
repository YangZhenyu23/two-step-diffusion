"""Compatibility regression test: Stage II must keep MeanFlow frozen.

The historical filename is retained, but joint training is intentionally no
longer implemented.
"""

import torch

from twostep import TwoStepDiffusion
from twostep_config import TwoStepConfig


class SyntheticParticleBatches:
    def __init__(self, independent_reference: bool):
        self.independent_reference = independent_reference

    def __iter__(self):
        while True:
            condition = torch.tensor([0.25, 0.75]).view(2, 1, 1).expand(2, 32, 1)
            source = torch.randn(2, 32, 2)
            reference_base = torch.randn_like(source) if self.independent_reference else source
            reference = reference_base + torch.cat((condition, -0.5 * condition), dim=-1)
            yield torch.cat((source, condition), dim=-1), reference


def main() -> None:
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
    )
    model = TwoStepDiffusion(config)
    model.train_meanflow(SyntheticParticleBatches(independent_reference=False))
    before = {name: value.detach().clone() for name, value in model.meanflow.state_dict().items()}
    dp_history = model.train_dp(SyntheticParticleBatches(independent_reference=True))
    after = model.meanflow.state_dict()
    assert all(torch.equal(before[name], after[name]) for name in before), "MeanFlow changed during Stage II."
    assert set(model.dp_history["learning_rate"]) == {config.dp_lr}, "Stage-II learning rate changed."
    inputs, _ = next(iter(SyntheticParticleBatches(independent_reference=False)))
    prediction = model.sample(inputs)
    assert prediction.shape == (2, 32, 2) and torch.isfinite(prediction).all()
    assert len(dp_history) == config.dp_steps
    print("Frozen-MeanFlow Stage-II regression test passed")


if __name__ == "__main__":
    main()
