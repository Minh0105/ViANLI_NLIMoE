from typing import Dict, Any, List, Optional
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class XLMRobertaNLI(nn.Module):
    """XLM-R encoder with a linear classification head.

    use_layer_norm=True adds LayerNorm before dropout and the classifier, so that with pooling='mean' the head
    matches NLIMoE without the MoE block (ablation A4). class_weights weights the cross-entropy per class.
    Both are off by default (E0, E5a).
    """
    def __init__(
        self,
        backbone_name: str = "xlm-roberta-base",
        num_labels: int = 3,
        dropout_prob: float = 0.1,
        pooling: str = "cls",
        encoder_dropout: Optional[float] = None,
        use_layer_norm: bool = False,
        class_weights: Optional[List[float]] = None
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_labels = num_labels
        self.pooling = pooling.lower()

        self.config = AutoConfig.from_pretrained(backbone_name)
        if encoder_dropout is not None:
            self.config.hidden_dropout_prob = encoder_dropout
            self.config.attention_probs_dropout_prob = encoder_dropout
        self.encoder = AutoModel.from_pretrained(backbone_name, config=self.config)
        self.hidden_size = self.config.hidden_size

        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(self.hidden_size, num_labels)
        if class_weights is not None and len(class_weights) != num_labels:
            raise ValueError(f"class_weights needs {num_labels} values, got {len(class_weights)}")
        # not stored in checkpoints, so E0 / E5a checkpoints load unchanged
        self.register_buffer("class_weights", None if class_weights is None
                             else torch.tensor(class_weights, dtype=torch.float), persistent=False)

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
        return_hidden_states: bool = False
    ) -> Dict[str, Any]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        pooled = self.pool_representation(last_hidden_state, attention_mask)

        normed = self.layer_norm(pooled) if self.use_layer_norm else pooled
        dropped = self.dropout(normed)
        logits = self.classifier(dropped)

        loss = None
        if labels is not None:
            weight = None if self.class_weights is None else self.class_weights.to(logits.dtype)
            loss = nn.functional.cross_entropy(logits, labels, weight=weight)

        result = {
            "logits": logits,
            "loss": loss,
            "pooled_representation": pooled
        }
        if return_hidden_states:
            result["last_hidden_state"] = last_hidden_state

        return result
