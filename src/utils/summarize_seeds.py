import os
import csv
import glob
import argparse
from statistics import mean, pstdev


def last_row(path):
    with open(path, "r") as f:
        rows = list(csv.DictReader(f))
        return rows[-1] if rows else None


def summarize(pattern, metrics):
    files = sorted(glob.glob(pattern))
    vals = {m: [] for m in metrics}
    for fp in files:
        row = last_row(fp)
        if not row:
            continue
        for m in metrics:
            if m in row and row[m] != "":
                try:
                    vals[m].append(float(row[m]))
                except ValueError:
                    pass
    summary = {m: (mean(vals[m]) if vals[m] else None, pstdev(vals[m]) if len(vals[m]) > 1 else 0.0) for m in metrics}
    return files, summary


def fmt(name, summary):
    parts = []
    for k, (mu, sd) in summary.items():
        if mu is None:
            parts.append(f"{k}: n/a")
        else:
            parts.append(f"{k}: {mu:.4f} ± {sd:.4f}")
    return f"{name}: " + ", ".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Summarize seed sweep results")
    parser.add_argument("--pattern", type=str, help="Custom glob pattern (e.g., 'runs/scratch_seed*/metrics.csv')")
    parser.add_argument("--regimes", nargs="+", default=["oneshot", "progressive", "scratch"],
                        help="Regimes to summarize (default: oneshot progressive scratch)")
    args = parser.parse_args()
    
    if args.pattern:
        # Custom pattern mode
        metrics_to_track = ["val_acc", "val_kl", "val_agree", "actual_sparsity"]
        files, s = summarize(args.pattern, metrics_to_track)
        print(f"Files matched: {len(files)}")
        if files:
            print(f"  First: {files[0]}")
            print(f"  Last: {files[-1]}")
        print(fmt("Results", s))
        return
    
    # Default mode: summarize all regimes
    print("\n" + "="*80)
    print("SEED SWEEP SUMMARY")
    print("="*80 + "\n")
    
    for regime in args.regimes:
        print(f"\n{regime.upper()}")
        print("-" * 40)
        
        if regime == "oneshot":
            files, s = summarize("runs/oneshot_seed*/metrics.csv", ["val_acc", "val_kl", "val_agree"])
            print(f"Files found: {len(files)}")
            if s:
                print(fmt("Results", s))
        
        elif regime == "progressive":
            files, s = summarize("runs/progressive_seed*/metrics.csv", ["val_acc", "val_kl", "val_agree", "actual_sparsity"])
            print(f"Files found: {len(files)}")
            if s:
                print(fmt("Results", s))
        
        elif regime == "scratch":
            files = sorted(glob.glob("runs/scratch_seed*/metrics.csv"))
            vals = []
            for fp in files:
                row = last_row(fp)
                if row:
                    try:
                        vals.append(float(row["val_acc"]))
                    except Exception:
                        pass
            mu = mean(vals) if vals else None
            sd = pstdev(vals) if len(vals) > 1 else 0.0
            print(f"Files found: {len(files)}")
            if mu is None:
                print("Results: val_acc n/a")
            else:
                print(f"Results: val_acc {mu:.4f} ± {sd:.4f}")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
