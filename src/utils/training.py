"""Shared training loop with validation cadence in optimizer steps.

Supports mixed precision (bf16 / fp16 autocast) and gradient accumulation, so the
paper's effective batch size can be kept on GPUs with less memory, e.g.
batch_size 8 x gradient_accumulation_steps 2 = effective batch 16.
"""
import contextlib
import json
import math
import time
from pathlib import Path
import torch
from tqdm import tqdm

PRECISIONS = ('fp32', 'bf16', 'fp16', 'auto')


def resolve_precision(precision, device):
    """Map 'auto' to bf16 when the GPU supports it, else fp16; CPU always uses fp32."""
    precision = (precision or 'fp32').lower()
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")
    if torch.device(device).type != 'cuda':
        return 'fp32'
    if precision == 'auto':
        return 'bf16' if torch.cuda.is_bf16_supported() else 'fp16'
    if precision == 'bf16' and not torch.cuda.is_bf16_supported():
        raise ValueError("This GPU does not support bf16; use precision fp16 or auto")
    return precision


def autocast_context(device, precision):
    """Autocast context for forward passes; a no-op for fp32."""
    if precision == 'fp32':
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if precision == 'bf16' else torch.float16
    return torch.autocast(device_type=torch.device(device).type, dtype=dtype)


def optimizer_steps_per_epoch(loader, accumulation_steps):
    return math.ceil(len(loader) / max(1, accumulation_steps))


def append_history(path, record):
    """Append one JSON line to a run's history file (read by the Colab monitor)."""
    if not path:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    record = {'time': time.time(), **record}
    with open(path, 'a', encoding='utf-8') as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=float) + '\n')


def _gpu_memory_gb(device):
    if torch.device(device).type != 'cuda':
        return 0.0
    return round(torch.cuda.max_memory_allocated(device) / 1024 ** 3, 2)


def train_with_validation(model, loader, device, optimizer, scheduler, epochs,
                          training, evaluate, checkpoint_path, sanity=False,
                          history_path=None, grad_probe=None):
    """`history_path`: JSONL file receiving train/val events.
    `grad_probe(model) -> dict`: extra gradient statistics, measured right
    before each logged optimizer step (e.g. the norm of dL/dw_c)."""
    best = None
    stale = 0
    global_step = 0
    frequency = training.get('eval_frequency', 0)
    patience = training.get('early_stopping_patience', 0)
    accumulation = max(1, int(training.get('gradient_accumulation_steps', 1)))
    precision = resolve_precision(training.get('precision', 'fp32'), device)
    scaler = torch.amp.GradScaler('cuda') if precision == 'fp16' else None
    clip = training.get('max_grad_norm')
    log_every = max(1, int(training.get('log_every', 20)))
    steps_per_epoch = optimizer_steps_per_epoch(loader, accumulation)
    total_steps = steps_per_epoch * epochs
    window = {'loss': 0.0, 'batches': 0, 'experts': 0.0, 'complexity': 0.0, 'threshold': 0.0, 'routed': 0}
    batch_size = getattr(loader, 'batch_size', None)
    effective = batch_size * accumulation if batch_size else 'n/a'
    print(f"[Train] precision={precision} | grad_accumulation={accumulation} | effective batch={effective}")

    def validate():
        nonlocal best, stale
        with autocast_context(device, precision):
            metrics = evaluate()
        print(f"Step {global_step}: Val Acc: {metrics['accuracy']:.4f} | Val Macro-F1: {metrics['macro_f1']:.4f}")
        improved = best is None or metrics['accuracy'] > best['accuracy']
        if improved:
            best = dict(metrics)
            stale = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> Saved best checkpoint to {checkpoint_path}")
        else:
            stale += 1
        append_history(history_path, {
            'type': 'val', 'step': global_step, 'total_steps': total_steps, 'epoch': epoch,
            'best': improved, 'gpu_mem_gb': _gpu_memory_gb(device),
            **{k: v for k, v in metrics.items() if isinstance(v, (int, float))}})
        model.train()
        return (sanity and metrics['accuracy'] >= 0.95) or (not sanity and patience > 0 and stale >= patience)

    append_history(history_path, {'type': 'start', 'total_steps': total_steps, 'epochs': epochs,
                                  'precision': precision, 'accumulation': accumulation,
                                  'batch_size': getattr(loader, 'batch_size', None)})
    epoch = 0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        num_batches = len(loader)
        for index, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch}/{epochs}")):
            # The last group of an epoch may hold fewer than `accumulation` batches.
            group_start = (index // accumulation) * accumulation
            group_size = min(accumulation, num_batches - group_start)
            with autocast_context(device, precision):
                output = model(input_ids=batch['input_ids'].to(device),
                               attention_mask=batch['attention_mask'].to(device),
                               labels=batch['label'].to(device))
            loss = output['loss']
            scaled = loss / group_size
            if scaler is not None:
                scaler.scale(scaled).backward()
            else:
                scaled.backward()
            total_loss += loss.item()
            window['loss'] += loss.item()
            window['batches'] += 1
            if 'selected_mask' in output and output['selected_mask'] is not None:
                window['experts'] += output['selected_mask'].float().sum(-1).mean().item()
                window['routed'] += 1
                if output.get('complexity') is not None:
                    window['complexity'] += output['complexity'].float().mean().item()
                if output.get('threshold') is not None:
                    window['threshold'] += output['threshold'].float().mean().item()

            if index - group_start + 1 < group_size:
                continue
            if scaler is not None:
                scaler.unscale_(optimizer)
            logging_step = history_path and (global_step + 1) % log_every == 0
            probe = grad_probe(model) if (logging_step and grad_probe is not None) else {}
            if clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            if logging_step:
                record = {'type': 'train', 'step': global_step, 'total_steps': total_steps, 'epoch': epoch,
                          'loss': window['loss'] / max(1, window['batches']),
                          'lr': optimizer.param_groups[0]['lr'], 'gpu_mem_gb': _gpu_memory_gb(device), **probe}
                if window['routed']:
                    record.update(avg_experts=window['experts'] / window['routed'],
                                  mean_complexity=window['complexity'] / window['routed'],
                                  mean_threshold=window['threshold'] / window['routed'])
                append_history(history_path, record)
                window = dict.fromkeys(window, 0)
            if not sanity and frequency > 0 and global_step % frequency == 0:
                if validate():
                    return best
        print(f"Epoch {epoch}: Train Loss: {total_loss / num_batches:.4f}")
        # Final validation also covers datasets shorter than eval_frequency.
        if sanity or frequency == 0 or (epoch == epochs and global_step % frequency != 0):
            if validate():
                return best
    return best


def save_run_config(cfg, checkpoint_path):
    path = Path(checkpoint_path).with_suffix('.config.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding='utf-8')
