"""Tests for the proposed improvements B2 (STE threshold) and B3 (bottleneck experts),
the training-history log and the experiment queue."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import yaml
from transformers import XLMRobertaConfig, XLMRobertaModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.factory import build_moe_model
from src.models.paper_nlimoe import build_expert
from src.utils.training import train_with_validation

TINY = XLMRobertaConfig(vocab_size=32, hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32,
                        hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)


def tiny_model(config_name, **moe_overrides):
    cfg = yaml.safe_load(Path(f'configs/{config_name}.yaml').read_text(encoding='utf-8'))
    cfg['model'].update(expected_hidden_size=16, dropout=0.0, encoder_dropout=0.0)
    cfg['moe'].update(moe_overrides)
    with patch('src.models.paper_nlimoe.AutoConfig.from_pretrained', return_value=TINY), \
            patch('src.models.paper_nlimoe.AutoModel.from_pretrained', side_effect=lambda *a, **k: XLMRobertaModel(TINY)):
        return build_moe_model(cfg)


def batch(size=8, length=12, seed=0):
    generator = torch.Generator().manual_seed(seed)
    ids = torch.randint(3, 32, (size, length), generator=generator)
    mask = torch.ones(size, length, dtype=torch.long)
    mask[0, -4:] = 0
    return ids, mask, torch.randint(0, 3, (size,), generator=generator)


class ImprovementTests(unittest.TestCase):
    def gate_gradient(self, model):
        ids, mask, labels = batch()
        model.train()
        model(ids, mask, labels=labels)['loss'].backward()
        return sum(float(p.grad.abs().sum()) for p in model.router.complexity_gate.parameters() if p.grad is not None)

    def test_hard_mask_blocks_gradient_and_ste_restores_it(self):
        torch.manual_seed(0)
        self.assertEqual(self.gate_gradient(tiny_model('nlimoe_dynamic')), 0.0)
        torch.manual_seed(0)
        self.assertGreater(self.gate_gradient(tiny_model('nlimoe_ste')), 0.0)

    def test_ste_keeps_the_paper_forward_pass(self):
        torch.manual_seed(0)
        hard = tiny_model('nlimoe_dynamic')
        ste = tiny_model('nlimoe_ste')
        ste.load_state_dict(hard.state_dict())
        ids, mask, labels = batch()
        for train_mode in (True, False):
            hard.train(train_mode)
            ste.train(train_mode)
            a, b = hard(ids, mask, labels=labels), ste(ids, mask, labels=labels)
            torch.testing.assert_close(a['logits'], b['logits'])
            self.assertTrue(torch.equal(a['selected_mask'], b['selected_mask']))

    def test_bottleneck_expert_parameter_count(self):
        count = lambda module: sum(p.numel() for p in module.parameters())
        self.assertEqual(count(build_expert(1024, expert_type='linear')), 1_049_600)
        self.assertEqual(count(build_expert(1024, expert_type='bottleneck', bottleneck_dim=256)), 525_568)
        model = tiny_model('nlimoe_ste_bottleneck', bottleneck_dim=4)
        self.assertEqual(model.experts.expert_type, 'bottleneck')
        self.assertEqual(count(model.experts), 7 * ((16 * 4 + 4) + (4 * 16 + 16)))

    def test_configs_change_only_one_component(self):
        load = lambda name: yaml.safe_load(Path(f'configs/{name}.yaml').read_text(encoding='utf-8'))
        b1, b2, b3 = load('nlimoe_dynamic'), load('nlimoe_ste'), load('nlimoe_ste_bottleneck')
        for section in ('data', 'model', 'training'):
            self.assertEqual(b1[section], b2[section])
            self.assertEqual(b2[section], b3[section])
        changed_b2 = {k for k in set(b1['moe']) | set(b2['moe']) if b1['moe'].get(k) != b2['moe'].get(k)}
        changed_b3 = {k for k in set(b2['moe']) | set(b3['moe']) if b2['moe'].get(k) != b3['moe'].get(k)}
        self.assertEqual(changed_b2, {'threshold_gradient', 'threshold_temperature', 'expert_type'})
        self.assertEqual(b2['moe']['expert_type'], 'linear')
        self.assertEqual(changed_b3, {'expert_type', 'bottleneck_dim'})

    def test_history_log_records_training_and_validation(self):
        torch.manual_seed(0)
        model = tiny_model('nlimoe_ste')
        batches = [dict(zip(('input_ids', 'attention_mask', 'label'), batch(4, seed=i))) for i in range(6)]
        with tempfile.TemporaryDirectory() as folder:
            history = os.path.join(folder, 'run.jsonl')
            probe = lambda net: {'grad_norm_wc': float(net.router.complexity_gate.weight.grad.norm())}
            train_with_validation(model, batches, 'cpu', torch.optim.AdamW(model.parameters(), lr=1e-3), None, 1,
                                  {'gradient_accumulation_steps': 2, 'log_every': 1, 'eval_frequency': 0},
                                  lambda: {'accuracy': 0.5, 'macro_f1': 0.4}, os.path.join(folder, 'best.pt'),
                                  history_path=history, grad_probe=probe)
            events = [json.loads(line) for line in Path(history).read_text(encoding='utf-8').splitlines()]
        kinds = [e['type'] for e in events]
        self.assertEqual(kinds[0], 'start')
        self.assertEqual(kinds.count('train'), 3)  # 6 batches / accumulation 2
        self.assertEqual(kinds[-1], 'val')
        train = [e for e in events if e['type'] == 'train']
        self.assertTrue(all(e['grad_norm_wc'] > 0 for e in train))
        self.assertTrue(all({'loss', 'avg_experts', 'mean_complexity', 'mean_threshold'} <= set(e) for e in train))

    def test_queue_recognises_finished_runs(self):
        import pandas as pd
        from scripts.run_queue import is_done
        from src.utils.viz import variant_of
        finished = pd.DataFrame({'experiment_name': [
            'xlmr_large_paper_v3_seed42_20261120_101010_000001',
            'nlimoe_paper_v3_ste_bottleneck_dynamic_seed42_20261120_101010_000002']})
        self.assertTrue(is_done('B0', 42, finished))
        self.assertFalse(is_done('B0', 123, finished))
        self.assertTrue(is_done('B3', 42, finished))
        self.assertFalse(is_done('B2', 42, finished))
        self.assertEqual(variant_of(finished['experiment_name'][1]), 'B3')
        self.assertEqual(variant_of('nlimoe_paper_v3_ste_dynamic_seed7_20261120_101010_000003'), 'B2')
        self.assertEqual(variant_of('nlimoe_paper_v3_dynamic_dynamic_seed7_20261120_101010_000004'), 'B1')


if __name__ == '__main__':
    unittest.main()
