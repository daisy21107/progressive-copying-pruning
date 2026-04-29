from typing import Tuple, Dict, Any
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import random


class MNISTPairDataset(Dataset):
    def __init__(self, root: str, train: bool, transform=None, seed: int = 42, label_rule: str = "parity",
                 pairing_mode: str = "static", data_seed: int = 12345, noise_std: float = 0.0,
                 noise_mode: str = "fixed", noise_seed: int = 0, normalize=None, sanity_check: bool = False,
                 random_position: bool = False, canvas_size: int = 42, bg_noise_std: float = 0.0):
        self.mnist = datasets.MNIST(root=root, train=train, download=True, transform=transform)
        self.train = train
        self.transform = transform
        self.pairing_mode = pairing_mode
        self.label_rule = label_rule
        self.noise_std = noise_std
        self.noise_mode = noise_mode
        self.noise_seed = noise_seed
        self.normalize = normalize
        self.sanity_check = sanity_check
        self.random_position = random_position
        self.canvas_size = canvas_size
        self.bg_noise_std = bg_noise_std
        self.position_rng = random.Random(data_seed + (100 if train else 101))

        # Static pairing: precompute all partner indices once
        if pairing_mode == "static":
            rng = random.Random(data_seed + (0 if train else 1))
            self.pairs = [rng.randrange(0, len(self.mnist)) for _ in range(len(self.mnist))]
        else:
            # Dynamic pairing: will sample fresh each call
            self.rng = random.Random(seed + (0 if train else 1))
            self.pairs = None

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx: int):
        img1, y1 = self.mnist[idx]
        # Get partner index
        if self.pairing_mode == "static":
            j = self.pairs[idx]
        else:
            j = self.rng.randrange(0, len(self.mnist))

        img2, y2 = self.mnist[j]

        # Apply random positioning if enabled (before noise to keep structure clean)
        if self.random_position:
            img1, img2 = self._apply_random_position(img1, idx * 2), self._apply_random_position(img2, idx * 2 + 1)

        # Apply noise if noise_std > 0 (after ToTensor scaling to [0,1])
        if self.noise_std > 0.0:
            img1, img2 = self._apply_noise(img1, idx * 2), self._apply_noise(img2, idx * 2 + 1)

        # Apply background noise if enabled
        if self.bg_noise_std > 0.0:
            img1, img2 = self._apply_bg_noise(img1, idx * 2), self._apply_bg_noise(img2, idx * 2 + 1)

        # Apply normalization after noise to keep old behavior when noise_std==0
        if self.normalize is not None:
            img1 = self.normalize(img1)
            img2 = self.normalize(img2)

        if self.sanity_check:
            if not torch.isfinite(img1).all() or not torch.isfinite(img2).all():
                raise ValueError("Non-finite values detected in dataset output")

        # Label rules
        if self.label_rule == "parity":
            label = (int(y1) + int(y2)) % 2  # binary
            num_classes = 2
        elif self.label_rule == "sum_mod10":
            label = (int(y1) + int(y2)) % 10  # 10-way
            num_classes = 10
        elif self.label_rule == "left_gt_right":
            label = 1 if int(y1) > int(y2) else 0
            num_classes = 2
        else:
            raise ValueError(f"Unknown label rule: {self.label_rule}")

        return img1, img2, torch.tensor(label, dtype=torch.long), num_classes

    def label_distribution(self):
        """Compute label distribution for sanity check."""
        from collections import Counter
        labels = []
        for idx in range(len(self)):
            _, _, y, _ = self[idx]
            labels.append(int(y))
        return dict(Counter(labels))

    def _apply_noise(self, img: torch.Tensor, noise_idx: int) -> torch.Tensor:
        """Apply Gaussian noise to image tensor.
        
        Args:
            img: Tensor in range [0, 1] (after ToTensor scaling)
            noise_idx: Deterministic index for fixed-mode seeding
        
        Returns:
            Noisy tensor clamped to [0, 1]
        """
        if self.noise_mode == "fixed":
            # Deterministic noise seeded by noise_seed + noise_idx
            g = torch.Generator()
            g.manual_seed(self.noise_seed + noise_idx)
            noise = torch.randn(img.shape, generator=g, device=img.device, dtype=img.dtype) * self.noise_std
        else:
            # Dynamic: fresh noise each call
            noise = torch.randn_like(img) * self.noise_std
        
        noisy_img = img + noise
        # Clamp to valid range [0, 1]
        noisy_img = torch.clamp(noisy_img, 0.0, 1.0)

        if self.sanity_check:
            if not torch.isfinite(noisy_img).all():
                raise ValueError("Non-finite values detected after noise injection")
            if noisy_img.min() < 0.0 or noisy_img.max() > 1.0:
                raise ValueError("Noise injection out of [0,1] range")

        return noisy_img

    def _apply_random_position(self, img: torch.Tensor, position_idx: int) -> torch.Tensor:
        """Place MNIST digit at random position on larger canvas.
        
        Args:
            img: Tensor of shape (1, 28, 28) in range [0, 1]
            position_idx: Deterministic index for position seeding
        
        Returns:
            Tensor of shape (1, canvas_size, canvas_size)
        """
        if img.shape[-1] != 28:
            raise ValueError(f"Expected 28x28 digit, got {img.shape}")
        
        # Create canvas (filled with zeros/background)
        canvas = torch.zeros(1, self.canvas_size, self.canvas_size, dtype=img.dtype)
        
        # Compute random position (deterministic seeding)
        g = torch.Generator()
        g.manual_seed(self.noise_seed + position_idx)  # Reuse noise_seed for determinism
        
        max_offset = self.canvas_size - 28
        if max_offset <= 0:
            # Canvas too small, just return the digit centered
            return img
        
        # Generate random offsets deterministically
        offset_x = int(torch.randint(0, max_offset + 1, (1,), generator=g).item())
        offset_y = int(torch.randint(0, max_offset + 1, (1,), generator=g).item())
        
        # Place digit on canvas
        canvas[:, offset_y:offset_y+28, offset_x:offset_x+28] = img
        
        return canvas

    def _apply_bg_noise(self, img: torch.Tensor, noise_idx: int) -> torch.Tensor:
        """Apply background noise (independent of digit noise).
        
        Args:
            img: Tensor in range [0, 1]
            noise_idx: Deterministic index for seeding
        
        Returns:
            Noisy tensor clamped to [0, 1]
        """
        g = torch.Generator()
        g.manual_seed(self.noise_seed + noise_idx + 10000)  # Offset to avoid collision
        noise = torch.randn_like(img, generator=g, dtype=img.dtype) * self.bg_noise_std
        
        bg_noisy = img + noise
        bg_noisy = torch.clamp(bg_noisy, 0.0, 1.0)
        
        return bg_noisy

    def show_sample_pairs(self, n=10):
        """Show first n (idx, pair_idx) for reproducibility check."""
        if self.pairing_mode == "static":
            return [(i, self.pairs[i]) for i in range(min(n, len(self.pairs)))]
        else:
            return None  # dynamic mode doesn't precompute


def _pair_collate(batch):
    """Collate function at module level for pickling compatibility with multiprocessing."""
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
    random_position = bool(data_cfg.get("random_position", False))
    canvas_size = int(data_cfg.get("canvas_size", 42))
    bg_noise_std = float(data_cfg.get("bg_noise_std", 0.0))
    seed = int(cfg.get("seed", 42))

    transform = transforms.ToTensor()
    normalize = transforms.Normalize((0.1307,), (0.3081,))

    train_dataset = MNISTPairDataset(root=root, train=True, transform=transform, seed=seed, label_rule=label_rule,
                                      pairing_mode=pairing_mode, data_seed=data_seed, noise_std=noise_std,
                                      noise_mode=noise_mode, noise_seed=noise_seed, normalize=normalize,
                                      sanity_check=sanity_check, random_position=random_position,
                                      canvas_size=canvas_size, bg_noise_std=bg_noise_std)
    test_dataset = MNISTPairDataset(root=root, train=False, transform=transform, seed=seed, label_rule=label_rule,
                                     pairing_mode=pairing_mode, data_seed=data_seed, noise_std=noise_std,
                                     noise_mode=noise_mode, noise_seed=noise_seed, normalize=normalize,
                                     sanity_check=sanity_check, random_position=random_position,
                                     canvas_size=canvas_size, bg_noise_std=bg_noise_std)

    # Sanity checks: log label distribution and sample pairs
    train_labels = train_dataset.label_distribution()
    test_labels = test_dataset.label_distribution()
    print(f"[Data] Train label distribution: {train_labels}")
    print(f"[Data] Test label distribution: {test_labels}")
    if random_position:
        print(f"[Data] Random positioning enabled: canvas_size={canvas_size}")
    if noise_std > 0.0:
        print(f"[Data] Noise enabled: std={noise_std}, mode={noise_mode}, seed={noise_seed}")
    if bg_noise_std > 0.0:
        print(f"[Data] Background noise enabled: std={bg_noise_std}")
    if pairing_mode == "static":
        print(f"[Data] Train sample pairs (first 10): {train_dataset.show_sample_pairs(10)}")
        print(f"[Data] Test sample pairs (first 10): {test_dataset.show_sample_pairs(10)}")

    if noise_mode == "dynamic":
        print(f"[WARNING] noise_mode='dynamic': behavior is stochastic by design. "
              f"With num_workers={num_workers}, exact training trajectories may not be reproducible.")

    if sanity_check:
        img1_a, img2_a, _, _ = train_dataset[0]
        img1_b, img2_b, _, _ = train_dataset[0]
        if not torch.isfinite(img1_a).all() or not torch.isfinite(img2_a).all():
            raise ValueError("Sanity check failed: non-finite values in dataset output")
        if noise_std == 0.0:
            print("[Data] Sanity check: noise_std=0.0 OK")
        elif noise_mode == "fixed":
            if not (torch.allclose(img1_a, img1_b) and torch.allclose(img2_a, img2_b)):
                raise ValueError("Sanity check failed: fixed noise is not deterministic")
            print("[Data] Sanity check: fixed noise deterministic OK")
        else:
            if torch.allclose(img1_a, img1_b) and torch.allclose(img2_a, img2_b):
                raise ValueError("Sanity check failed: dynamic noise appears deterministic")
            print("[Data] Sanity check: dynamic noise stochastic OK")

    # Peek to get num_classes deterministically via one sample
    _, _, _, num_classes = train_dataset[0]

    # For reproducible batch ordering: pass seeded generator to DataLoader
    # This ensures shuffle=True produces same batch order across runs (with fixed mode noise)
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, collate_fn=_pair_collate, generator=generator)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, collate_fn=_pair_collate)

    return train_loader, test_loader, num_classes
