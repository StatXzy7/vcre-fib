from __future__ import annotations

import torch

from sfibai_b.model_cost import count_conv_linear_flops


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 2, kernel_size=3, padding=1, bias=False)
        self.linear = torch.nn.Linear(32, 4, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(self.conv(inputs).flatten(1))


def test_conv_linear_flop_counter_has_an_explicit_two_flops_per_mac_contract() -> None:
    model = _TinyModel()
    result = count_conv_linear_flops(model, torch.zeros(1, 3, 4, 4))

    assert result == 1_984
