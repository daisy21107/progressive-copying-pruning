from typing import Any, Dict

from src.models.cnn_pair import PairClassifier
from src.models.resnet_cifar import CIFARResNet18


def _model_cfg(cfg: Dict[str, Any], role: str) -> Dict[str, Any]:
    if role in cfg and isinstance(cfg.get(role), dict):
        return cfg.get(role, {})
    return cfg.get("model", {})


def build_classifier(cfg: Dict[str, Any], num_classes: int, role: str = "model"):
    data_task = str(cfg.get("data", {}).get("task", "")).lower()
    model_cfg = _model_cfg(cfg, role)
    arch = str(model_cfg.get("arch", model_cfg.get("type", "auto"))).lower()

    if data_task == "cifar100_multiclass" or arch in {"resnet18", "cifar_resnet18", "resnet18_cifar"}:
        in_channels = int(model_cfg.get("in_channels", 3))
        return CIFARResNet18(num_classes=num_classes, in_channels=in_channels)

    width = int(model_cfg.get("width", 32))
    hidden = int(model_cfg.get("hidden", 128))
    shared = bool(model_cfg.get("shared_encoder", True))
    in_channels = int(model_cfg.get("in_channels", 1))
    return PairClassifier(
        num_classes=num_classes,
        width=width,
        hidden=hidden,
        shared_encoder=shared,
        in_channels=in_channels,
    )