from .nlimoe import NLIMoEModel
from .paper_nlimoe import PaperNLIMoEModel


def build_moe_model(cfg, routing_type=None, top_k=None, threshold=None):
    model, moe = cfg['model'], cfg['moe']
    mode = routing_type or moe.get('routing_type', 'dynamic')
    k = top_k if top_k is not None else moe.get('top_k', 2)
    tau = threshold if threshold is not None else moe.get('threshold', 0.25)
    if model.get('architecture') == 'paper_nlimoe_v3':
        return PaperNLIMoEModel(
            backbone_name=model['backbone'], num_labels=model.get('num_labels', 3),
            num_experts=moe['num_experts'], routing_mode=mode, top_k=k,
            static_threshold=tau, gamma=moe['gamma'], dropout_prob=model['dropout'],
            dynamic_loss_weight=moe['dynamic_loss_weight'], balance_loss_weight=moe['balance_loss_weight'],
            encoder_dropout=model.get('encoder_dropout'), expected_hidden_size=model.get('expected_hidden_size', 1024),
            threshold_gradient=moe.get('threshold_gradient', 'hard'),
            threshold_temperature=moe.get('threshold_temperature', 0.1),
            expert_type=moe.get('expert_type', 'linear'), bottleneck_dim=moe.get('bottleneck_dim', 256))
    return NLIMoEModel(
        backbone_name=model['backbone'], num_labels=model.get('num_labels', 3),
        num_experts=moe.get('num_experts', 4), intermediate_dim=moe.get('intermediate_dim', 1536),
        router_type=moe.get('router_type', 'linear'), routing_mode=mode, top_k=k,
        static_threshold=tau, dynamic_scale=moe.get('dynamic_scale', 1.0),
        aux_loss_weight=moe.get('aux_loss_weight', 0.01), dropout_prob=model.get('dropout', 0.1),
        pooling=model.get('pooling', 'cls'), threshold_gradient=moe.get('threshold_gradient', 'hard'),
        threshold_temperature=moe.get('threshold_temperature', 0.1), sparse_inference=moe.get('sparse_inference', False))
