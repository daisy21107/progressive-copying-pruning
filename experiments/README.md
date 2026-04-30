# CIFAR-100 Final Experiment Package

## Scope
This folder contains the finalized CIFAR-100 experimental artifacts used for supervisor discussion.

## Task
- Binary classification using CIFAR-100 subsets: underwater vs food
- Noise setting used for final comparison: `noise_std=0.2`

## Runs Included
- Teacher: `cifar_teacher_final`
- Students: `oneshot`, `progressive`, `scratch` with 10 seeds each (`0..9`)
- Total run directories: 31

## Main Results
- Teacher: 78.42%
- Scratch: 74.17% ± 0.82%
- Oneshot: 70.42% ± 1.41%
- Progressive: 69.71% ± 0.76%

## Fidelity Notes (Progressive vs Oneshot)
- Slightly better (lower) KL and MSE metrics
- Slightly worse agreement and final accuracy
- Interpretation: improved distribution/logit matching does not translate to higher task accuracy under current setup

## Folder Map
- `runs/`: teacher and seed runs (metrics/checkpoints)
- `results/`: aggregate JSON + plots
- `configs/`: CIFAR configs used
- `logs/`: CIFAR execution logs
