"""Offline regression tests; synthetic inputs do not measure Vietnamese NLI quality."""
import unittest
from unittest.mock import patch

import torch
from transformers import XLMRobertaConfig, XLMRobertaModel

from src.data.dataset import load_vianli_dataloaders
from src.models.moe_layers import MoEExpertLayer
from src.models.nlimoe import NLIMoEModel
from src.models.router import NLIMoERouter


class MoEPipelineTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)

    def test_dynamic_head_receives_gradients_and_updates(self):
        router = NLIMoERouter(input_dim=8, threshold_gradient="straight_through", dynamic_scale=0.5)
        experts = MoEExpertLayer(input_dim=8, intermediate_dim=16, dropout_prob=0)
        inputs = torch.randn(32, 8)
        labels = torch.arange(32) % 3
        optimizer = torch.optim.Adam(list(router.parameters()) + list(experts.parameters()), lr=0.02)
        initial_head = router.dynamic_thresh_head[0].weight.detach().clone()
        losses = []
        for step in range(120):
            optimizer.zero_grad()
            routed = router(inputs)
            logits = (experts(inputs) * routed['routing_weights'].unsqueeze(-1)).sum(1)
            loss = torch.nn.functional.cross_entropy(logits, labels) + 0.01 * routed['aux_loss']
            loss.backward()
            if step == 0:
                for name, parameter in router.dynamic_thresh_head.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
            optimizer.step()
            losses.append(loss.item())
        self.assertFalse(torch.equal(initial_head, router.dynamic_thresh_head[0].weight))
        self.assertLess(losses[-1], losses[0] * 0.25)
        self.assertGreaterEqual((logits.argmax(-1) == labels).float().mean().item(), 0.95)

    def test_straight_through_keeps_hard_forward(self):
        hard = NLIMoERouter(input_dim=8, dynamic_scale=0.5)
        experimental = NLIMoERouter(input_dim=8, dynamic_scale=0.5, threshold_gradient="straight_through")
        experimental.load_state_dict(hard.state_dict())
        inputs = torch.randn(12, 8)
        expected, actual = hard(inputs), experimental(inputs)
        self.assertTrue(torch.equal(expected['selected_mask'], actual['selected_mask']))
        torch.testing.assert_close(expected['routing_weights'], actual['routing_weights'])

    def test_sparse_dispatch_matches_dense_and_skips_unused_experts(self):
        experts = MoEExpertLayer(input_dim=8, intermediate_dim=16, dropout_prob=0).eval()
        inputs = torch.randn(6, 8)
        mask = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0]] * 3, dtype=torch.bool)
        dense = experts(inputs)
        calls = []
        handles = [expert.register_forward_pre_hook(lambda module, args: calls.append(args[0].size(0))) for expert in experts.experts]
        sparse = experts.forward_selected(inputs, mask)
        for handle in handles:
            handle.remove()
        torch.testing.assert_close(sparse, dense * mask.unsqueeze(-1))
        self.assertEqual(calls, [3, 3])
        self.assertEqual(sum(calls), mask.sum().item())

    def test_routing_invariants_and_empty_selection_fallback(self):
        inputs = torch.randn(6, 8)
        for mode in ('top_k', 'static', 'dynamic', 'dense'):
            router = NLIMoERouter(input_dim=8, routing_mode=mode, static_threshold=1.0, dynamic_scale=2.0)
            output = router(inputs)
            self.assertTrue(torch.isfinite(output['routing_weights']).all())
            self.assertTrue((output['selected_mask'].sum(-1) >= 1).all())
            torch.testing.assert_close(output['routing_weights'].sum(-1), torch.ones(6))

    def test_sanity_loader_evaluates_same_training_examples(self):
        def tokenizer(*args, **kwargs):
            return {'input_ids': torch.ones(1, 4, dtype=torch.long), 'attention_mask': torch.ones(1, 4, dtype=torch.long)}
        data = {split: [{'uid': f'{split}-{i}', 'premise': 'A', 'hypothesis': 'B', 'label': i % 3} for i in range(40)] for split in ('train', 'validation', 'test')}
        with patch('src.data.dataset.load_dataset', return_value=data), patch('src.data.dataset.AutoTokenizer.from_pretrained', return_value=tokenizer):
            train, validation, test, _ = load_vianli_dataloaders(overfit_subset_size=32)
        self.assertIs(train.dataset, validation.dataset)
        self.assertEqual(len(train.dataset), 32)
        self.assertTrue(validation.dataset[0]['uid'].startswith('train-'))
        self.assertEqual(len(test.dataset), 40)

    def test_tiny_encoder_integration_checkpoint_and_sparse_inference(self):
        config = XLMRobertaConfig(vocab_size=32, hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32, max_position_embeddings=32)
        with patch('src.models.nlimoe.AutoConfig.from_pretrained', return_value=config), patch('src.models.nlimoe.AutoModel.from_pretrained', side_effect=lambda *args, **kwargs: XLMRobertaModel(config)):
            model = NLIMoEModel(intermediate_dim=16, threshold_gradient='straight_through', dynamic_scale=0.5, pooling='mean')
            restored = NLIMoEModel(intermediate_dim=16, threshold_gradient='straight_through', dynamic_scale=0.5, pooling='mean', sparse_inference=True)
        ids = torch.randint(3, 32, (6, 8))
        mask = torch.ones_like(ids)
        output = model(ids, mask, labels=torch.arange(6) % 3)
        output['loss'].backward()
        self.assertTrue(all(p.grad is not None for p in model.router.dynamic_thresh_head.parameters()))
        restored.load_state_dict(model.state_dict(), strict=True)
        model.eval()
        restored.eval()
        with torch.no_grad():
            torch.testing.assert_close(model(ids, mask)['logits'], restored(ids, mask)['logits'])
            # Same dynamic checkpoint can be evaluated under static routing.
            sweep_output = restored(ids, mask, routing_mode_override='static', threshold_override=0.25)
        self.assertEqual(sweep_output['logits'].shape, (6, 3))


if __name__ == '__main__':
    unittest.main()
