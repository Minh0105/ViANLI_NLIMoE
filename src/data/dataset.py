import os
from typing import Dict, Any, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from datasets import load_dataset
from transformers import AutoTokenizer

LABEL2ID = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

class ViANLIDataset(Dataset):
    """
    Dataset wrapper cho bộ dữ liệu uitnlp/ViANLI.
    Hỗ trợ cả nhãn chuỗi (string) và nhãn số (integer).
    """
    def __init__(self, hf_split_dataset, tokenizer, max_length: int = 256, dynamic_padding: bool = False):
        self.data = hf_split_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dynamic_padding = dynamic_padding

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        premise = str(item.get("premise", ""))
        hypothesis = str(item.get("hypothesis", ""))
        raw_label = item.get("label")

        # Map label chuẩn về [0, 1, 2]
        if isinstance(raw_label, str):
            label_id = LABEL2ID.get(raw_label.lower().strip(), -1)
        elif isinstance(raw_label, int):
            label_id = raw_label
        else:
            label_id = int(raw_label)

        encoding = self.tokenizer(
            premise,
            hypothesis,
            max_length=self.max_length,
            padding=False if self.dynamic_padding else "max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "uid": item.get("uid", idx),
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label_id, dtype=torch.long),
            "premise": premise,
            "hypothesis": hypothesis
        }

def make_dynamic_padding_collate(pad_token_id: int):
    """Pad each batch only to its longest sequence (attention_mask = 0 on padding)."""
    def collate(items):
        width = max(item["input_ids"].size(0) for item in items)
        ids = torch.full((len(items), width), pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(items), width), dtype=torch.long)
        for row, item in enumerate(items):
            length = item["input_ids"].size(0)
            ids[row, :length] = item["input_ids"]
            mask[row, :length] = item["attention_mask"]
        return {
            "uid": [item["uid"] for item in items],
            "input_ids": ids,
            "attention_mask": mask,
            "label": torch.stack([item["label"] for item in items]),
            "premise": [item["premise"] for item in items],
            "hypothesis": [item["hypothesis"] for item in items],
        }
    return collate


def load_vianli_dataloaders(
    dataset_name: str = "uitnlp/ViANLI",
    tokenizer_name: str = "xlm-roberta-base",
    max_length: int = 256,
    batch_size: int = 32,
    num_workers: int = 0,
    overfit_subset_size: Optional[int] = None,
    dynamic_padding: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    """
    Tải dataset uitnlp/ViANLI và chuẩn bị DataLoader cho train, validation, test.
    Hỗ trợ chế độ overfit_subset_size (32-64 mẫu) cho sanity check.
    """
    print(f"[Data] Loading dataset {dataset_name}...")
    dataset = load_dataset(dataset_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    train_data = ViANLIDataset(dataset["train"], tokenizer, max_length=max_length, dynamic_padding=dynamic_padding)
    val_data = ViANLIDataset(dataset["validation"], tokenizer, max_length=max_length, dynamic_padding=dynamic_padding)
    test_data = ViANLIDataset(dataset["test"], tokenizer, max_length=max_length, dynamic_padding=dynamic_padding)

    if overfit_subset_size is not None and overfit_subset_size > 0:
        print(f"[Data] SANITY CHECK MODE: Creating overfit subset of {overfit_subset_size} samples...")
        indices = list(range(min(overfit_subset_size, len(train_data))))
        train_data = Subset(train_data, indices)
        # Overfit checks evaluate the exact training examples, not unseen val data.
        val_data = train_data

    collate_fn = make_dynamic_padding_collate(tokenizer.pad_token_id) if dynamic_padding else None
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn
    )

    metadata = {
        "train_size": len(train_data),
        "val_size": len(val_data),
        "test_size": len(test_data),
        "tokenizer": tokenizer,
        "label2id": LABEL2ID,
        "id2label": ID2LABEL
    }

    print(f"[Data] Loaded {len(train_data)} train, {len(val_data)} val, {len(test_data)} test samples.")
    return train_loader, val_loader, test_loader, metadata
