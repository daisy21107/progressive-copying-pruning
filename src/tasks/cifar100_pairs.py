from typing import Tuple, Dict, Any
import random

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


class CIFAR100PairDataset(Dataset):
    UNDERWATER_CLASSES = {
        "aquarium_fish",
        "flatfish",
        "ray",
        "shark",
        "trout",
        "dolphin",
        "otter",
        "seal",
        "whale",
    }

    FOOD_CLASSES = {
        "apple",
        "mushroom",
        "orange",
        "pear",
        "sweet_pepper",
    }

    def __init__(
        self,
        root: str,
        train: bool,
        transform=None,
        label_rule: str = "parity",
        pairing_mode: str = "static",
        data_seed: int = 12345,
        noise_std: float = 0.0,
        noise_mode: str = "fixed",
        noise_seed: int = 0,
        normalize=None,
        sanity_check: bool = False,
        max_samples: int = 0,
    ):
        self.dataset = datasets.CIFAR100(root=root, train=train, download=True, transform=transform)
        self.label_rule = label_rule
        self.pairing_mode = pairing_mode
        self.noise_std = noise_std
        self.noise_mode = noise_mode
        self.noise_seed = noise_seed
        self.normalize = normalize
        self.sanity_check = sanity_check

        class_to_idx = {name: idx for idx, name in enumerate(self.dataset.classes)}
        self.underwater_indices = sorted([class_to_idx[name] for name in self.UNDERWATER_CLASSES if name in class_to_idx])
        self.food_indices = sorted([class_to_idx[name] for name in self.FOOD_CLASSES if name in class_to_idx])

        if len(self.underwater_indices) == 0 or len(self.food_indices) == 0:
            raise ValueError("Could not resolve CIFAR-100 underwater/food classes")

        self.sample_indices = []
        self.sample_groups = []
        for idx, class_idx in enumerate(self.dataset.targets):
            if class_idx in self.underwater_indices:
                self.sample_indices.append(idx)
                self.sample_groups.append(0)
            elif class_idx in self.food_indices:
                self.sample_indices.append(idx)
                self.sample_groups.append(1)

        if max_samples > 0 and len(self.sample_indices) > max_samples:
            rng = random.Random(data_seed + (0 if train else 1))
            chosen = rng.sample(range(len(self.sample_indices)), k=max_samples)
            self.sample_indices = [self.sample_indices[i] for i in chosen]
            self.sample_groups = [self.sample_groups[i] for i in chosen]

        if pairing_mode == "static":
            rng = random.Random(data_seed + (10 if train else 11))
            self.pairs = [rng.randrange(0, len(self.sample_indices)) for _ in range(len(self.sample_indices))]
        else:
            self.rng = random.Random(data_seed + (20 if train else 21))
            self.pairs = None

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx: int):
        base_idx_1 = self.sample_indices[idx]
        group_1 = self.sample_groups[idx]

        if self.pairing_mode == "static":
            pair_pos = self.pairs[idx]
        else:
            pair_pos = self.rng.randrange(0, len(self.sample_indices))

        base_idx_2 = self.sample_indices[pair_pos]
        group_2 = self.sample_groups[pair_pos]

        img1, _ = self.dataset[base_idx_1]
        img2, _ = self.dataset[base_idx_2]

        if self.noise_std > 0.0:
            img1 = self._apply_noise(img1, idx * 2)
            img2 = self._apply_noise(img2, idx * 2 + 1)

        if self.normalize is not None:
            img1 = self.normalize(img1)
            img2 = self.normalize(img2)

        if self.sanity_check:
            if not torch.isfinite(img1).all() or not torch.isfinite(img2).all():
                raise ValueError("Non-finite values detected in CIFAR pair output")

        if self.label_rule == "parity":
            label = (group_1 + group_2) % 2
            num_classes = 2
        elif self.label_rule == "left_gt_right":
            label = 1 if group_1 > group_2 else 0
            num_classes = 2
        else:
            raise ValueError(f"Unsupported label rule for CIFAR pair task: {self.label_rule}")

        return img1, img2, torch.tensor(label, dtype=torch.long), num_classes

    def _apply_noise(self, img: torch.Tensor, noise_idx: int) -> torch.Tensor:
        if self.noise_mode == "fixed":
            g = torch.Generator()
            g.manual_seed(self.noise_seed + noise_idx)
            noise = torch.randn(img.shape, generator=g, device=img.device, dtype=img.dtype) * self.noise_std
        else:
            noise = torch.randn_like(img) * self.noise_std

        noisy_img = torch.clamp(img + noise, 0.0, 1.0)
        return noisy_img

    def label_distribution(self):
        from collections import Counter

        counts = Counter()
        for i in range(len(self.sample_groups)):
            if self.pairing_mode == "static":
                j = self.pairs[i]
            else:
                j = i
            if self.label_rule == "parity":
                y = (self.sample_groups[i] + self.sample_groups[j]) % 2
            else:
                y = 1 if self.sample_groups[i] > self.sample_groups[j] else 0
            counts[int(y)] += 1
        return dict(counts)

    def show_sample_pairs(self, n=10):
        if self.pairing_mode != "static":
            return None
        return [(i, self.pairs[i]) for i in range(min(n, len(self.pairs)))]


def _pair_collate(batch):
    imgs1 = torch.stack([b[0] for b in batch])
    imgs2 = torch.stack([b[1] for b in batch])
    labels = torch.stack([b[2] for b in batch])
    return imgs1, imgs2, labels


def get_dataloaders(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, int]:
    data_cfg = cfg.get("data", {})
    root = data_cfg.get("root", "./data")
    batch_size = int(data_cfg.get("batch_size", 128))
    num_workers = int(data_cfg.get("num_workers", 0))
    label_rule = data_cfg.get("label_rule", "parity")
    pairing_mode = data_cfg.get("pairing", "static")
    data_seed = int(data_cfg.get("data_seed", 12345))
    noise_std = float(data_cfg.get("noise_std", 0.0))
    noise_mode = data_cfg.get("noise_mode", "fixed")
    noise_seed = int(data_cfg.get("noise_seed", 0))
    sanity_check = bool(data_cfg.get("sanity_check", False))
    max_samples = int(data_cfg.get("max_samples", 0))
    seed = int(cfg.get("seed", 42))

    transform = transforms.ToTensor()
    normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))

    train_dataset = CIFAR100PairDataset(
        root=root,
        train=True,
        transform=transform,
        label_rule=label_rule,
        pairing_mode=pairing_mode,
        data_seed=data_seed,
        noise_std=noise_std,
        noise_mode=noise_mode,
        noise_seed=noise_seed,
        normalize=normalize,
        sanity_check=sanity_check,
        max_samples=max_samples,
    )
    test_dataset = CIFAR100PairDataset(
        root=root,
        train=False,
        transform=transform,
        label_rule=label_rule,
        pairing_mode=pairing_mode,
        data_seed=data_seed,
        noise_std=noise_std,
        noise_mode=noise_mode,
        noise_seed=noise_seed,
        normalize=normalize,
        sanity_check=sanity_check,
        max_samples=max_samples,
    )

    print(f"[Data] Task: cifar100_binary_pairs")
    print(f"[Data] Train pairs: {len(train_dataset)}")
    print(f"[Data] Test pairs: {len(test_dataset)}")
    print(f"[Data] Underwater classes: {train_dataset.underwater_indices}")
    print(f"[Data] Food classes: {train_dataset.food_indices}")
    print(f"[Data] Train label distribution: {train_dataset.label_distribution()}")
    print(f"[Data] Test label distribution: {test_dataset.label_distribution()}")
    if noise_std > 0.0:
        print(f"[Data] Noise enabled: std={noise_std}, mode={noise_mode}, seed={noise_seed}")
    if pairing_mode == "static":
        print(f"[Data] Train sample pairs (first 10): {train_dataset.show_sample_pairs(10)}")
        print(f"[Data] Test sample pairs (first 10): {test_dataset.show_sample_pairs(10)}")

    _, _, _, num_classes = train_dataset[0]

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=_pair_collate,
        generator=generator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_pair_collate,
    )

    return train_loader, test_loader, num_classes
