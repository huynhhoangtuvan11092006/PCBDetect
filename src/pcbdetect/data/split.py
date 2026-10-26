"""Dataset splitting strategies for DeepPCB.

Supports both:
1. Standard Split (reproducing the original 1000 trainval / 500 test benchmark).
2. Strict Group-Disjoint Split (partitioning by physical PCB board layout to eliminate data leakage).
"""

from dataclasses import dataclass
from typing import List, Literal, Tuple
from pcbdetect.data.dataset import DeepPCBSample, load_deeppcb_dataset


@dataclass
class DatasetSplits:
    """Container holding train, val, and test sample splits."""
    train: List[DeepPCBSample]
    val: List[DeepPCBSample]
    test: List[DeepPCBSample]
    mode: str

    @property
    def total_count(self) -> int:
        return len(self.train) + len(self.val) + len(self.test)

    def summary(self) -> dict:
        """Return counts and unique groups per split."""
        return {
            "mode": self.mode,
            "train_count": len(self.train),
            "val_count": len(self.val),
            "test_count": len(self.test),
            "total_count": self.total_count,
            "train_groups": sorted(list({s.group_id for s in self.train})),
            "val_groups": sorted(list({s.group_id for s in self.val})),
            "test_groups": sorted(list({s.group_id for s in self.test})),
        }


def create_standard_split(
    val_ratio: float = 0.2,
    random_seed: int = 42,
) -> DatasetSplits:
    """Create benchmark-compatible split following original DeepPCB setup.

    Original test.txt (500 samples) becomes test set.
    Original trainval.txt (1000 samples) is split into train and val.
    """
    import random

    trainval_samples = load_deeppcb_dataset(split="trainval")
    test_samples = load_deeppcb_dataset(split="test")

    # Deterministic shuffle of trainval
    rng = random.Random(random_seed)
    shuffled = list(trainval_samples)
    rng.shuffle(shuffled)

    val_count = int(len(shuffled) * val_ratio)
    val_samples = shuffled[:val_count]
    train_samples = shuffled[val_count:]

    return DatasetSplits(
        train=train_samples,
        val=val_samples,
        test=test_samples,
        mode="standard",
    )


def create_group_disjoint_split(
    test_groups: Tuple[str, ...] = ("group12000", "group12100", "group90100"),
    val_groups: Tuple[str, ...] = ("group12300", "group77000"),
) -> DatasetSplits:
    """Create strict board-disjoint split where board groups NEVER overlap between splits.

    This ensures models are evaluated on unseen physical PCB topologies, eliminating
    memorization of specific circuit track paths.

    DeepPCB has 11 groups:
      - Test groups: group12000 (14), group12100 (146), group90100 (74) -> ~234 samples (~15.6%)
      - Val groups: group12300 (98), group77000 (107) -> ~205 samples (~13.7%)
      - Train groups: remaining 6 groups -> ~1061 samples (~70.7%)
    """
    all_samples = load_deeppcb_dataset(split="all")

    train_samples: List[DeepPCBSample] = []
    val_samples: List[DeepPCBSample] = []
    test_samples: List[DeepPCBSample] = []

    test_grp_set = set(test_groups)
    val_grp_set = set(val_groups)

    for s in all_samples:
        if s.group_id in test_grp_set:
            test_samples.append(s)
        elif s.group_id in val_grp_set:
            val_samples.append(s)
        else:
            train_samples.append(s)

    return DatasetSplits(
        train=train_samples,
        val=val_samples,
        test=test_samples,
        mode="group_disjoint",
    )


def get_dataset_splits(
    strategy: Literal["standard", "group_disjoint"] = "standard",
    val_ratio: float = 0.2,
    random_seed: int = 42,
) -> DatasetSplits:
    """Unified factory to retrieve dataset splits by strategy."""
    if strategy == "group_disjoint":
        return create_group_disjoint_split()
    return create_standard_split(val_ratio=val_ratio, random_seed=random_seed)
