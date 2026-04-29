from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LEGACY_RUNS_DIR = PROJECT_ROOT / "runs"
LEGACY_RESULTS_DIR = PROJECT_ROOT / "results"

CIFAR_PACKAGE_DIR = PROJECT_ROOT / "experiments" / "cifar_final"
CIFAR_RUNS_DIR = CIFAR_PACKAGE_DIR / "runs"
CIFAR_RESULTS_DIR = CIFAR_PACKAGE_DIR / "results"


def get_cifar_runs_dir() -> Path:
    if CIFAR_RUNS_DIR.exists():
        return CIFAR_RUNS_DIR
    return LEGACY_RUNS_DIR


def get_cifar_results_dir(create: bool = True) -> Path:
    target = CIFAR_RESULTS_DIR if CIFAR_PACKAGE_DIR.exists() else LEGACY_RESULTS_DIR
    if create:
        target.mkdir(parents=True, exist_ok=True)
    return target
