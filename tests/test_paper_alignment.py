import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from transformers import XLMRobertaConfig, XLMRobertaModel
import yaml

from src.models.factory import build_moe_model
from src.models.paper_nlimoe import PaperRouter, PaperExpertLayer, PaperNLIMoEModel
from src.utils.training import train_with_validation


class PaperAlignmentTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)

    def test_router_threshold_and_both_losses_match_equations(self):
        router = PaperRouter(8)
        x = torch.randn(12, 8)
        output = router(x)
        p = router.gate(x).softmax(-1)
        c = router.complexity_gate(x).sigmoid()
        expected_mask = p > (0.1 + 0.1 * c)
        self.assertTrue(torch.equal(output['selected_mask'], expected_mask))
        torch.testing.assert_close(output['threshold'], 0.1 + 0.1 * c)
        torch.testing.assert_close(output['dynamic_loss'], -(p * p.log()).sum(-1).mean() / 7)
        torch.testing.assert_close(output['aux_loss'], 7 * (expected_mask.float().mean(0) * p.mean(0)).sum())
        (output['dynamic_loss'] + output['aux_loss']).backward()
        # Do not disguise the unresolved hard-mask gradient issue.
        self.assertIsNone(router.complexity_gate.weight.grad)
        self.assertIsNotNone(router.gate.weight.grad)

    def test_empty_selection_convention_is_finite_and_not_top1_fallback(self):
        router = PaperRouter(8)
        with torch.no_grad():
            router.gate.weight.zero_()
            router.gate.bias.zero_()
            router.complexity_gate.weight.zero_()
            router.complexity_gate.bias.zero_()
        result = router(torch.zeros(2, 8)) # 1/7 < 0.15
        self.assertTrue(result['empty_selection'].all())
        self.assertEqual(result['selected_mask'].sum().item(), 0)
        self.assertTrue(torch.isfinite(result['routing_weights']).all())
        self.assertEqual(result['routing_weights'].sum().item(), 0)

    def test_topk_vianli_selects_four_of_five(self):
        router = PaperRouter(8, num_experts=5, top_k=4, routing_mode='top_k')
        output = router(torch.randn(12, 8))
        self.assertTrue((output['selected_mask'].sum(-1) == 4).all())
        torch.testing.assert_close(output['routing_weights'].sum(-1), torch.ones(12))

    def test_expert_order_dimension_and_sparse_outputs(self):
        layer = PaperExpertLayer(16, num_experts=7).eval()
        self.assertIsInstance(layer.experts[0][0], nn.Linear)
        self.assertIsInstance(layer.experts[0][1], nn.Dropout)
        self.assertIsInstance(layer.experts[0][2], nn.ReLU)
        self.assertEqual(layer.experts[0][0].out_features, 16)
        x = torch.randn(4, 16)
        mask = torch.rand(4, 7) > 0.5
        torch.testing.assert_close(layer(x, mask), layer(x) * mask.unsqueeze(-1))

    def test_model_loss_pooling_classifier_and_checkpoint(self):
        config = XLMRobertaConfig(vocab_size=32, hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32, max_position_embeddings=32)
        with patch('src.models.paper_nlimoe.AutoConfig.from_pretrained', return_value=config), patch('src.models.paper_nlimoe.AutoModel.from_pretrained', side_effect=lambda *a, **k: XLMRobertaModel(config)):
            model = PaperNLIMoEModel(expected_hidden_size=16)
            restored = PaperNLIMoEModel(expected_hidden_size=16)
        model.eval()
        ids = torch.randint(3, 32, (4, 6))
        mask = torch.tensor([[1, 1, 1, 1, 0, 0]] * 4)
        labels = torch.arange(4) % 3
        result = model(ids, mask, labels=labels)
        hidden = model.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        torch.testing.assert_close(result['pooled_representation'], hidden[:, :4].mean(1))
        expected = result['task_loss'] + 0.001 * result['dynamic_loss'] + 0.01 * result['aux_loss']
        torch.testing.assert_close(result['loss'], expected)
        self.assertIsInstance(model.layer_norm, nn.LayerNorm)
        self.assertEqual(model.classifier.in_features, 16)
        result['loss'].backward()
        self.assertIsNotNone(model.classifier.weight.grad)
        restored.load_state_dict(model.state_dict(), strict=True)
        restored.eval()
        torch.testing.assert_close(result['logits'], restored(ids, mask)['logits'])

    def test_training_evaluation_cadence_and_saved_best_metrics(self):
        class TinyClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(2, 3)
            def forward(self, input_ids, attention_mask, labels):
                return {'loss': nn.functional.cross_entropy(self.layer(input_ids.float()), labels)}
        model = TinyClassifier()
        batch = {'input_ids': torch.ones(2, 2), 'attention_mask': torch.ones(2, 2), 'label': torch.tensor([0, 1])}
        calls = []
        def evaluate():
            calls.append(1)
            return {'accuracy': 0.6 if len(calls) == 1 else 0.5, 'macro_f1': 0.4 if len(calls) == 1 else 0.3}
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            best = train_with_validation(model, [batch] * 20, 'cpu', torch.optim.AdamW(model.parameters(), lr=1e-5), None,
                21, {'eval_frequency': 400, 'early_stopping_patience': 0, 'max_grad_norm': None}, evaluate, Path(folder)/'best.pt')
        self.assertEqual(len(calls), 2) # Step 400 and final step 420.
        self.assertEqual(best, {'accuracy': 0.6, 'macro_f1': 0.4})

    def test_reported_config_values_and_factory(self):
        for name in ('xlmr_base', 'nlimoe_dynamic', 'nlimoe_topk'):
            cfg = yaml.safe_load(Path(f'configs/{name}.yaml').read_text(encoding='utf-8'))
            self.assertEqual(cfg['data']['max_length'], 256)
            self.assertEqual(cfg['training']['epochs'], 7)
            self.assertEqual(cfg['training']['batch_size'], 16)
            self.assertEqual(cfg['training']['eval_frequency'], 400)
            self.assertEqual(cfg['training']['weight_decay'], 0)
            self.assertEqual(cfg['model']['dropout'], 0.4)
        cfg = yaml.safe_load(Path('configs/nlimoe_topk.yaml').read_text(encoding='utf-8'))
        tiny = XLMRobertaConfig(vocab_size=32, hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32)
        cfg['model']['expected_hidden_size'] = 16
        with patch('src.models.paper_nlimoe.AutoConfig.from_pretrained', return_value=tiny), patch('src.models.paper_nlimoe.AutoModel.from_pretrained', side_effect=lambda *a, **k: XLMRobertaModel(tiny)):
            model = build_moe_model(cfg)
        self.assertEqual(model.num_experts, 5)
        self.assertEqual(model.router.top_k, 4)


if __name__ == '__main__':
    unittest.main()
