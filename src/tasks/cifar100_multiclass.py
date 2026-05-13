from typing import Dict, Any, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


CIFAR100_SUPERCLASSES = {
    "aquatic_mammals": [4, 30, 55, 72, 95],
    "fish": [1, 32, 67, 73, 91],
    "flowers": [54, 62, 70, 82, 92],
    "food_containers": [9, 10, 16, 28, 61],
    "fruit_and_vegetables": [0, 51, 53, 57, 83],
    "household_electrical": [22, 39, 40, 86, 87],
    "household_furniture": [5, 20, 25, 84, 94],
    "insects": [6, 7, 14, 18, 24],
    "large_carnivores": [3, 42, 43, 88, 97],
    "vehicles_1": [8, 13, 48, 58, 90],
}


class CIFAR100MulticlassDataset(Dataset):
    def __init__(
        self,
        root: str,
        train: bool,
        transform=None,
        selected_superclasses: Optional[List[str]] = None,
    ):
        self.dataset = datasets.CIFAR100(root=root, train=train, download=True, transform=transform)
        self.selected_superclasses = selected_superclasses or list(CIFAR100_SUPERCLASSES.keys())[:10]
        self.superclass_to_label = {name: idx for idx, name in enumerate(self.selected_superclasses)}

        self.fine_to_label: Dict[int, int] = {}
        for superclass_name in self.selected_superclasses:
            if superclass_name not in CIFAR100_SUPERCLASSES:
                raise ValueError(f"Unknown CIFAR-100 superclass: {superclass_name}")
            label = self.superclass_to_label[superclass_name]
            for fine_label in CIFAR100_SUPERCLASSES[superclass_name]:
                self.fine_to_label[fine_label] = label

        self.indices = [
            index
            for index, fine_label in enumerate(self.dataset.targets)
            if fine_label in self.fine_to_label
        ]
        self.num_classes = len(self.selected_superclasses)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        real_idx = self.indices[idx]
        img, fine_label = self.dataset[real_idx]
        label = self.fine_to_label[int(fine_label)]
        return img, torch.tensor(label, dtype=torch.long)


def get_dataloaders(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, int]:
    data_cfg = cfg.get("data", {})
    root = data_cfg.get("root", "./data")
    batch_size = int(data_cfg.get("batch_size", 128))
    num_workers = int(data_cfg.get("num_workers", 0))
    seed = int(cfg.get("seed", 42))
    selected_superclasses = data_cfg.get("selected_superclasses")

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    train_dataset = CIFAR100MulticlassDataset(
        root=root,
        train=True,
        transform=train_transform,
        selected_superclasses=selected_superclasses,
    )
    test_dataset = CIFAR100MulticlassDataset(
        root=root,
        train=False,
        transform=test_transform,
        selected_superclasses=selected_superclasses,
    )

    print("[Data] Task: cifar100_multiclass")
    print(f"[Data] Selected superclasses: {train_dataset.selected_superclasses}")
    print(f"[Data] Train samples: {len(train_dataset)}")
    print(f"[Data] Test samples: {len(test_dataset)}")

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=generator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, test_loader, train_dataset.num_classes