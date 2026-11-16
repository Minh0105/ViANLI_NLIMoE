from typing import Dict, List, Any
import numpy as np
import torch

def compute_routing_diagnostics(
    routing_weights: torch.Tensor,
    selected_mask: torch.Tensor,
    labels: torch.Tensor = None,
    label_names: List[str] = None
) -> Dict[str, Any]:
    """
    Tính toán 4 chỉ số chẩn đoán định tuyến bắt buộc cho MoE:
    1. Expert Usage (%)
    2. Collapse Detection
    3. Average Experts Per Sample
    4. Router Entropy
    5. Expert Distribution theo nhãn NLI
    """
    if isinstance(routing_weights, torch.Tensor):
        routing_weights = routing_weights.detach().cpu().numpy()
    if isinstance(selected_mask, torch.Tensor):
        selected_mask = selected_mask.detach().cpu().numpy().astype(bool)
    if labels is not None and isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    batch_size, num_experts = routing_weights.shape

    samples_per_expert = selected_mask.sum(axis=0)
    expert_usage_pct = (samples_per_expert / batch_size) * 100.0
    avg_expert_weights = routing_weights.mean(axis=0)

    total_activations = selected_mask.sum()
    if total_activations > 0:
        activation_shares = samples_per_expert / total_activations
        collapse_warning = bool(np.any(activation_shares > 0.60) or np.any(activation_shares < 0.05))
    else:
        collapse_warning = True

    experts_per_sample = selected_mask.sum(axis=1)
    avg_experts_per_sample = float(np.mean(experts_per_sample))

    eps = 1e-9
    clipped_weights = np.clip(routing_weights, eps, 1.0)
    row_sums = clipped_weights.sum(axis=1, keepdims=True)
    norm_weights = clipped_weights / np.maximum(row_sums, eps)
    entropies = -np.sum(norm_weights * np.log(norm_weights), axis=1)
    mean_routing_entropy = float(np.mean(entropies))

    diagnostics = {
        "num_experts": num_experts,
        "expert_usage_pct": {f"expert_{i}": float(expert_usage_pct[i]) for i in range(num_experts)},
        "avg_expert_weights": {f"expert_{i}": float(avg_expert_weights[i]) for i in range(num_experts)},
        "collapse_warning": collapse_warning,
        "avg_experts_per_sample": avg_experts_per_sample,
        "mean_routing_entropy": mean_routing_entropy
    }

    if labels is not None:
        if label_names is None:
            label_names = [f"class_{i}" for i in range(int(labels.max()) + 1)]
        dist_per_label = {}
        for class_idx, name in enumerate(label_names):
            class_mask = (labels == class_idx)
            if np.sum(class_mask) > 0:
                class_expert_act = selected_mask[class_mask].sum(axis=0)
                class_usage_pct = (class_expert_act / np.sum(class_mask)) * 100.0
                dist_per_label[name] = {f"expert_{i}": float(class_usage_pct[i]) for i in range(num_experts)}
            else:
                dist_per_label[name] = {f"expert_{i}": 0.0 for i in range(num_experts)}
        diagnostics["expert_distribution_per_label"] = dist_per_label

    return diagnostics
