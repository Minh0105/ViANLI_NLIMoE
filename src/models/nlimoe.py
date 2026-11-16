from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

from .router import NLIMoERouter
from .moe_layers import MoEExpertLayer

class NLIMoEModel(nn.Module):
    """
    Toàn bộ kiến trúc NLIMoE:
    Backbone XLM-R -> Representation z -> Router(z) & Experts(z) -> Weighted Aggregation -> Logits.
    """
    def __init__(
        self,
        backbone_name: str = "xlm-roberta-base",
        num_labels: int = 3,
        num_experts: int = 4,
        intermediate_dim: int = 1536,
        router_type: str = "linear",
        routing_mode: str = "dynamic",
        top_k: int = 2,
        static_threshold: float = 0.25,
        dynamic_scale: float = 1.0,
        aux_loss_weight: float = 0.01,
        dropout_prob: float = 0.1,
        pooling: str = "cls",
        threshold_gradient: str = "hard",
        threshold_temperature: float = 0.1,
        sparse_inference: bool = False
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_labels = num_labels
        self.num_experts = num_experts
        self.aux_loss_weight = aux_loss_weight
        self.pooling = pooling.lower()
        self.sparse_inference = sparse_inference

        # 1. Backbone
        self.config = AutoConfig.from_pretrained(backbone_name)
        self.encoder = AutoModel.from_pretrained(backbone_name, config=self.config)
        self.hidden_size = self.config.hidden_size

        # 2. Router
        self.router = NLIMoERouter(
            input_dim=self.hidden_size,
            num_experts=num_experts,
            router_type=router_type,
            routing_mode=routing_mode,
            top_k=top_k,
            static_threshold=static_threshold,
            dynamic_scale=dynamic_scale,
            threshold_gradient=threshold_gradient,
            threshold_temperature=threshold_temperature
        )

        # 3. Experts Layer
        self.experts = MoEExpertLayer(
            num_experts=num_experts,
            input_dim=self.hidden_size,
            intermediate_dim=intermediate_dim,
            output_dim=num_labels,
            dropout_prob=dropout_prob
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def pool_representation(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            return last_hidden_state[:, 0, :]
        elif self.pooling == "mean":
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            return sum_embeddings / sum_mask
        else:
            raise ValueError(f"Unsupported pooling method: {self.pooling}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        routing_mode_override: Optional[str] = None,
        threshold_override: Optional[float] = None
    ) -> Dict[str, Any]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        z = self.pool_representation(outputs.last_hidden_state, attention_mask)

        router_out = self.router(
            z,
            routing_mode_override=routing_mode_override,
            threshold_override=threshold_override
        )
        routing_weights = router_out["routing_weights"]
        selected_mask = router_out["selected_mask"]
        raw_probs = router_out["raw_probs"]
        aux_loss = router_out["aux_loss"]

        # Dense training preserves gradients to the threshold estimator through
        # every expert. Sparse dispatch is used only at evaluation/inference.
        if self.sparse_inference and not self.training:
            expert_outputs = self.experts.forward_selected(z, selected_mask)
        else:
            expert_outputs = self.experts(z)

        weights_expanded = routing_weights.unsqueeze(-1)
        logits = torch.sum(expert_outputs * weights_expanded, dim=1)

        total_loss = None
        task_loss = None
        if labels is not None:
            task_loss = self.loss_fn(logits, labels)
            total_loss = task_loss + self.aux_loss_weight * aux_loss

        return {
            "logits": logits,
            "loss": total_loss,
            "task_loss": task_loss,
            "aux_loss": aux_loss,
            "routing_weights": routing_weights,
            "selected_mask": selected_mask,
            "raw_probs": raw_probs,
            "pooled_representation": z
        }
