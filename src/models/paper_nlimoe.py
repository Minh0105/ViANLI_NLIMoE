"""NLIMoE (arXiv:2406.17716v3, Sec. 5, Eq. 2-16).

threshold_gradient='straight_through', expert_type='bottleneck' and
expert_cost_weight (a penalty on experts used per sample) are our
modifications; the defaults follow the paper. An empty selection gives a zero
MoE vector (not specified in the paper).
"""
import torch
from torch import nn
from transformers import AutoConfig, AutoModel


class PaperRouter(nn.Module):
    def __init__(self, hidden_size, num_experts=7, routing_mode="dynamic",
                 top_k=4, static_threshold=0.1, gamma=0.1,
                 threshold_gradient="hard", threshold_temperature=0.1):
        super().__init__()
        if num_experts < 1 or not 1 <= top_k <= num_experts:
            raise ValueError("Require 1 <= top_k <= num_experts")
        self.num_experts = num_experts
        self.routing_mode = routing_mode
        self.top_k = top_k
        self.static_threshold = static_threshold
        self.gamma = gamma
        if threshold_gradient not in ("hard", "straight_through"):
            raise ValueError("threshold_gradient must be 'hard' or 'straight_through'")
        if threshold_temperature <= 0:
            raise ValueError("threshold_temperature must be positive")
        self.threshold_gradient = threshold_gradient
        self.threshold_temperature = threshold_temperature
        self.gate = nn.Linear(hidden_size, num_experts, bias=True)
        self.complexity_gate = nn.Linear(hidden_size, 1, bias=True)

    def forward(self, x, routing_mode_override=None, threshold_override=None):
        mode = routing_mode_override or self.routing_mode
        probabilities = self.gate(x).softmax(-1)
        complexity = self.complexity_gate(x).sigmoid()
        threshold = self.static_threshold + self.gamma * complexity
        if mode == "dynamic":
            mask = probabilities > threshold
        elif mode == "static":
            tau = self.static_threshold if threshold_override is None else threshold_override
            mask = probabilities > tau
        elif mode == "top_k":
            indices = probabilities.topk(self.top_k, dim=-1).indices
            mask = torch.zeros_like(probabilities, dtype=torch.bool).scatter(1, indices, True)
        elif mode == "dense":
            mask = torch.ones_like(probabilities, dtype=torch.bool)
        else:
            raise ValueError(f"Unsupported routing mode: {mode}")
        selection = mask.to(probabilities.dtype)
        if mode == "dynamic" and self.training and self.threshold_gradient == "straight_through":
            # straight-through estimator: forward = hard mask, backward = d sigmoid((P - rho) / T)
            soft = torch.sigmoid((probabilities - threshold) / self.threshold_temperature)
            selection = selection + (soft - soft.detach())
        masked = probabilities * selection
        weights = masked / masked.sum(-1, keepdim=True).clamp_min(1e-9)
        # Eq. (12): average over experts, then over the batch.
        dynamic_loss = -(probabilities * probabilities.clamp_min(1e-9).log()).sum(-1).mean() / self.num_experts
        balance_loss = self.num_experts * (mask.float().mean(0) * probabilities.mean(0)).sum()
        # experts used per sample; with STE its gradient reaches the threshold
        expected_experts = selection.sum(-1).mean()
        return {"routing_weights": weights, "selected_mask": mask, "expected_experts": expected_experts,
                "raw_probs": probabilities, "complexity": complexity,
                "threshold": threshold, "dynamic_loss": dynamic_loss,
                "aux_loss": balance_loss, "empty_selection": ~mask.any(-1)}


def build_expert(hidden_size, dropout=0.4, expert_type="linear", bottleneck_dim=256):
    """linear: Linear(d,d)-Dropout-ReLU (paper); bottleneck: Linear(d,r)-ReLU-Linear(r,d)-Dropout-ReLU."""
    if expert_type == "linear":
        return nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Dropout(dropout), nn.ReLU())
    if expert_type == "bottleneck":
        if not 0 < bottleneck_dim < hidden_size:
            raise ValueError("bottleneck_dim must be in (0, hidden_size)")
        return nn.Sequential(nn.Linear(hidden_size, bottleneck_dim), nn.ReLU(),
                             nn.Linear(bottleneck_dim, hidden_size), nn.Dropout(dropout), nn.ReLU())
    raise ValueError("expert_type must be 'linear' or 'bottleneck'")


class PaperExpertLayer(nn.Module):
    def __init__(self, hidden_size, num_experts=7, dropout=0.4,
                 expert_type="linear", bottleneck_dim=256):
        super().__init__()
        self.expert_type = expert_type
        self.experts = nn.ModuleList([
            build_expert(hidden_size, dropout, expert_type, bottleneck_dim)
            for _ in range(num_experts)
        ])

    def forward(self, x, selected_mask=None):
        if selected_mask is None:
            return torch.stack([expert(x) for expert in self.experts], dim=1)
        outputs = []
        for index, expert in enumerate(self.experts):
            rows = selected_mask[:, index].nonzero(as_tuple=True)[0]
            output = torch.zeros_like(x)
            if rows.numel():
                output = output.index_copy(0, rows, expert(x.index_select(0, rows)).to(output.dtype))
            outputs.append(output)
        return torch.stack(outputs, dim=1)


class PaperNLIMoEModel(nn.Module):
    def __init__(self, backbone_name="xlm-roberta-large", num_labels=3,
                 num_experts=7, routing_mode="dynamic", top_k=4,
                 static_threshold=0.1, gamma=0.1, dropout_prob=0.4,
                 dynamic_loss_weight=0.001, balance_loss_weight=0.01,
                 encoder_dropout=None, expected_hidden_size=1024,
                 threshold_gradient="hard", threshold_temperature=0.1,
                 expert_type="linear", bottleneck_dim=256, expert_cost_weight=0.0):
        super().__init__()
        self.config = AutoConfig.from_pretrained(backbone_name)
        if expected_hidden_size is not None and self.config.hidden_size != expected_hidden_size:
            raise ValueError(f"Paper expects hidden_size={expected_hidden_size}; backbone has {self.config.hidden_size}")
        if encoder_dropout is not None:
            self.config.hidden_dropout_prob = encoder_dropout
            self.config.attention_probs_dropout_prob = encoder_dropout
        self.encoder = AutoModel.from_pretrained(backbone_name, config=self.config)
        self.hidden_size = self.config.hidden_size
        self.num_experts = num_experts
        self.router = PaperRouter(self.hidden_size, num_experts, routing_mode, top_k, static_threshold, gamma,
                                  threshold_gradient, threshold_temperature)
        self.experts = PaperExpertLayer(self.hidden_size, num_experts, dropout_prob, expert_type, bottleneck_dim)
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.classifier = nn.Linear(self.hidden_size, num_labels)
        self.dynamic_loss_weight = dynamic_loss_weight
        self.balance_loss_weight = balance_loss_weight
        self.expert_cost_weight = expert_cost_weight

    def forward(self, input_ids, attention_mask, labels=None,
                routing_mode_override=None, threshold_override=None):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        # Eq. 2: mean over non-padding tokens
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        routed = self.router(pooled, routing_mode_override, threshold_override)
        # with STE, unselected experts also need outputs to get a gradient
        ste_training = self.training and self.router.threshold_gradient == "straight_through"
        expert_vectors = self.experts(pooled, None if ste_training else routed["selected_mask"])
        combined = (expert_vectors * routed["routing_weights"].unsqueeze(-1)).sum(1)
        logits = self.classifier(self.layer_norm(combined))
        task_loss = None if labels is None else nn.functional.cross_entropy(logits, labels)
        loss = None if labels is None else (task_loss + self.dynamic_loss_weight * routed["dynamic_loss"]
                                           + self.balance_loss_weight * routed["aux_loss"]
                                           + self.expert_cost_weight * routed["expected_experts"])
        return {**routed, "logits": logits, "loss": loss, "task_loss": task_loss,
                "pooled_representation": pooled, "moe_representation": combined}
