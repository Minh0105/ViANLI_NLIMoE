from typing import List
import torch
import torch.nn as nn

class NLIExpert(nn.Module):
    """Linear -> GELU -> Dropout -> Linear."""
    def __init__(
        self,
        input_dim: int = 768,
        intermediate_dim: int = 1536,
        output_dim: int = 3,
        dropout_prob: float = 0.1
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, intermediate_dim),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(intermediate_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class MoEExpertLayer(nn.Module):
    """N experts applied in parallel."""
    def __init__(
        self,
        num_experts: int = 4,
        input_dim: int = 768,
        intermediate_dim: int = 1536,
        output_dim: int = 3,
        dropout_prob: float = 0.1
    ):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            NLIExpert(
                input_dim=input_dim,
                intermediate_dim=intermediate_dim,
                output_dim=output_dim,
                dropout_prob=dropout_prob
            )
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [expert(x) for expert in self.experts]
        return torch.stack(outputs, dim=1)

    def forward_selected(self, x: torch.Tensor, selected_mask: torch.Tensor) -> torch.Tensor:
        """Run each expert only on the samples routed to it."""
        outputs = []
        output_dim = self.experts[0].net[-1].out_features
        for expert_idx, expert in enumerate(self.experts):
            indices = selected_mask[:, expert_idx].nonzero(as_tuple=True)[0]
            expert_output = x.new_zeros((x.size(0), output_dim))
            if indices.numel():
                selected = expert(x.index_select(0, indices))
                expert_output = expert_output.index_copy(0, indices, selected.to(expert_output.dtype))
            outputs.append(expert_output)
        return torch.stack(outputs, dim=1)
