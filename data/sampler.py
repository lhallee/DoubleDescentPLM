import torch
import collections
import random
from typing import Any, Mapping, Optional, List, Dict, Tuple
from datasets import load_dataset
from transformers import EsmTokenizer
from torch.utils.data import Dataset


DEFAULT_REPO_ID = "Synthyra/PDB-Chain-Complex-Benchmark-Rigor"


class PDBClusteredDataset(Dataset):
    """One randomly selected row per sequence-cluster group per epoch."""

    def __init__(
        self,
        repo_id: str = DEFAULT_REPO_ID,
        mode: str = "chains",
        split: str = "train",
        seed: int = 0,
        epoch: int = 0,
        cluster_key: Optional[str] = None,
        complex_grouping: str = "member_cluster",
        load_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        mode_to_config = {
            "single": "chains",
            "chains": "chains",
            "complex": "complexes",
            "complexes": "complexes",
        }
        self.config = mode_to_config[mode]
        self.seed = int(seed)
        self.cluster_key = cluster_key or (
            "sequence_cluster_30" if self.config == "chains" else "sequence_clusters_30"
        )
        self.complex_grouping = complex_grouping
        self.dataset = load_dataset(
            repo_id,
            self.config,
            split=split,
            streaming=False,
            **dict(load_kwargs or {}),
        )
        self.groups = self._build_groups()
        self.group_keys = sorted(self.groups)
        self.set_epoch(epoch)

    def _group_keys_for_value(self, value: Any) -> tuple[tuple[str, ...], ...]:
        if self.config == "chains":
            return ((str(value),),)

        clusters = tuple(sorted({str(cluster) for cluster in value if str(cluster)}))
        if not clusters:
            raise ValueError("complex row has no sequence cluster labels")
        if self.complex_grouping == "cluster_set":
            return (clusters,)
        if self.complex_grouping == "member_cluster":
            return tuple((cluster,) for cluster in clusters)
        raise ValueError("complex_grouping must be 'member_cluster' or 'cluster_set'")

    def _build_groups(self) -> dict[tuple[str, ...], list[int]]:
        groups = collections.defaultdict(list)
        for row_idx, value in enumerate(self.dataset[self.cluster_key]):
            for key in self._group_keys_for_value(value):
                groups[key].append(row_idx)
        if not groups:
            raise ValueError("dataset split contains no rows")
        return dict(groups)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        rng = random.Random(self.seed + self.epoch)
        self.indices = [rng.choice(self.groups[key]) for key in self.group_keys]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        return self.dataset[self.indices[index]]


class TokenizeCollator:
    def __init__(
            self,
            max_length: int = 512,
            device: torch.device = "cuda",
            mask_rate: float = 0.15
        ):
        self.tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D')
        self.max_length = max_length
        self.device = device
        self.mask_rate = mask_rate
        self.mask_token = self.tokenizer.mask_token_id

    def mask(self, tokenized: Mapping[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        input_ids = tokenized['input_ids'].clone()
        labels = input_ids.clone()
        eligible = tokenized['attention_mask'].bool() & ~tokenized['special_tokens_mask'].bool()
        masked = (torch.rand(input_ids.shape) < self.mask_rate) & eligible
        if not masked.any(): # make sure at least one token is masked
            eligible_positions = eligible.nonzero()
            row, col = eligible_positions[torch.randint(eligible_positions.size(0), (1,)).item()]
            masked[row, col] = True
        input_ids[masked] = self.mask_token
        labels[~masked] = -100
        return input_ids, labels

    def __call__(self, batch: List[Mapping[str, Any]]) -> Dict[str, torch.Tensor]:
        seqs = [sample['seqs'].strip().upper() for sample in batch]
        tokenized = self.tokenizer(
            seqs,
            padding='max-length',
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt',
            return_special_tokens_mask=True,
        )
        input_ids, labels = self.mask(tokenized)
        return {
            'input_ids': input_ids.to(self.device),
            'attention_mask': tokenized['attention_mask'].to(self.device),
            'labels': labels.to(self.device),
        }



if __name__ == "__main__":
    from torch.utils.data import DataLoader
    ### Test
    test_dataset = PDBClusteredDataset(mode="single", split="test", seed=42)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=TokenizeCollator())

    for epoch in range(3):
        test_dataset.set_epoch(epoch)
        first_chain_batch = next(iter(test_loader))
        print(first_chain_batch)
        break
