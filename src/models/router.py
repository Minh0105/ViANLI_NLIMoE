from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class NLIMoERouter(nn.Module):
    """Top-k, static, dynamic or dense routing."""
    def __init__(
        self,
        input_dim: int = 768,
        num_experts: int = 4,
        router_type: str = "linear",
        routing_mode: str = "dynamic",
        top_k: int = 2,
        static_threshold: float = 0.25,
        dynamic_scale: float = 1.0,
        threshold_gradient: str = "hard",
        threshold_temperature: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.router_type = router_type
        self.routing_mode = routing_mode
        self.top_k = top_k
        self.static_threshold = static_threshold
        self.dynamic_scale = dynamic_scale
        if threshold_gradient not in {"hard", "straight_through"}:
            raise ValueError("threshold_gradient must be hard or straight_through")
        if threshold_temperature <= 0:
            raise ValueError("threshold_temperature must be positive")
        if num_experts < 1 or not 1 <= top_k <= num_experts:
            raise ValueError("Require num_experts >= 1 and 1 <= top_k <= num_experts")
        self.threshold_gradient = threshold_gradient
        self.threshold_temperature = threshold_temperature

        if router_type == "linear":
            self.gate = nn.Linear(input_dim, num_experts, bias=False)
        elif router_type == "mlp":
            self.gate = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.GELU(),
                nn.Linear(input_dim // 2, num_experts, bias=False)
            )
        else:
            raise ValueError(f"Unsupported router type: {router_type}")

        if routing_mode == "dynamic":
            self.dynamic_thresh_head = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )

    def forward(
        self,
        representation: torch.Tensor,
        routing_mode_override: str = None,
        threshold_override: float = None
    ) -> Dict[str, Any]:
        mode = routing_mode_override or self.routing_mode
        logits = self.gate(representation)  # [B, num_experts]
        raw_probs = F.softmax(logits, dim=-1)  # [B, num_experts]
        batch_size = representation.size(0)

        if mode == "top_k":
            k = min(self.top_k, self.num_experts)
            topk_weights, topk_indices = torch.topk(raw_probs, k, dim=-1)
            selected_mask = torch.zeros_like(raw_probs, dtype=torch.bool)
            selected_mask.scatter_(1, topk_indices, True)

            sum_topk = topk_weights.sum(dim=-1, keepdim=True).clamp(min=1e-9)
            norm_topk = topk_weights / sum_topk
            routing_weights = torch.zeros_like(raw_probs)
            routing_weights.scatter_(1, topk_indices, norm_topk)

        elif mode == "static":
            tau = threshold_override if threshold_override is not None else self.static_threshold
            selected_mask = raw_probs >= tau

            empty_samples = (selected_mask.sum(dim=-1) == 0)
            if empty_samples.any():
                top1_idx = raw_probs[empty_samples].argmax(dim=-1, keepdim=True)
                selected_mask[empty_samples] = selected_mask[empty_samples].scatter(1, top1_idx, True)

            masked_weights = raw_probs * selected_mask.float()
            sum_weights = masked_weights.sum(dim=-1, keepdim=True).clamp(min=1e-9)
            routing_weights = masked_weights / sum_weights

        elif mode == "dynamic":
            pred_tau = self.dynamic_thresh_head(representation) * self.dynamic_scale
            selected_mask = raw_probs >= pred_tau

            empty_samples = (selected_mask.sum(dim=-1) == 0)
            if empty_samples.any():
                top1_idx = raw_probs[empty_samples].argmax(dim=-1, keepdim=True)
                selected_mask[empty_samples] = selected_mask[empty_samples].scatter(1, top1_idx, True)

            # straight-through: hard mask forward, sigmoid gradient backward
            selection = selected_mask.float()
            if self.training and self.threshold_gradient == "straight_through":
                soft_selection = torch.sigmoid(
                    (raw_probs - pred_tau) / self.threshold_temperature
                )
                selection = selection + (soft_selection - soft_selection.detach())
            masked_weights = raw_probs * selection
            sum_weights = masked_weights.sum(dim=-1, keepdim=True).clamp(min=1e-9)
            routing_weights = masked_weights / sum_weights

        elif mode == "dense":
            selected_mask = torch.ones_like(raw_probs, dtype=torch.bool)
            routing_weights = raw_probs

        else:
            raise ValueError(f"Unsupported routing mode: {mode}")

        f = selected_mask.float().mean(dim=0)
        P = raw_probs.mean(dim=0)
        aux_loss = self.num_experts * torch.sum(f * P)

        return {
            "routing_weights": routing_weights,
            "selected_mask": selected_mask,
            "raw_probs": raw_probs,
            "aux_loss": aux_loss
        }
