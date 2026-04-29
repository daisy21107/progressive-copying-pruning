import os
import random
import json
import csv
from typing import Dict, Any

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(cfg: Dict[str, Any]) -> torch.device:
    use_cuda = cfg.get("device", "auto")
    if use_cuda == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(use_cuda)


def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)


def save_config_snapshot(cfg: Dict[str, Any], out_dir: str):
    with open(os.path.join(out_dir, "config_snapshot.json"), "w") as f:
        json.dump(cfg, f, indent=2)


class CSVLogger:
    def __init__(self, out_dir: str, filename: str = "metrics.csv"):
        self.path = os.path.join(out_dir, filename)
        self.file = open(self.path, mode="w", newline="")
        self.writer = None

    def log(self, metrics: Dict[str, Any]):
        if self.writer is None:
            self.writer = csv.DictWriter(self.file, fieldnames=list(metrics.keys()))
            self.writer.writeheader()
        self.writer.writerow(metrics)
        self.file.flush()

    def close(self):
        if self.file:
            self.file.close()
