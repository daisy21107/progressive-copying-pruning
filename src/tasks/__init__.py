from .dual_mnist import get_dataloaders as get_mnist_dataloaders
from .cifar100_pairs import get_dataloaders as get_cifar100_dataloaders
from .cifar100_multiclass import get_dataloaders as get_cifar100_multiclass_dataloaders


def get_dataloaders(cfg):
	data_cfg = cfg.get("data", {})
	task_name = data_cfg.get("task", "dual_mnist")

	if task_name == "dual_mnist":
		return get_mnist_dataloaders(cfg)
	if task_name == "cifar100_binary_pairs":
		return get_cifar100_dataloaders(cfg)
	if task_name == "cifar100_multiclass":
		return get_cifar100_multiclass_dataloaders(cfg)

	raise ValueError(f"Unknown data.task: {task_name}")


__all__ = ["get_dataloaders"]
