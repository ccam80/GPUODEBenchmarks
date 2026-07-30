#!/usr/bin/env python3
"""
Compare numerical results between different integration packages.

This script reads CSV files containing final state arrays from different ODE solver packages
and performs pairwise comparisons using numpy's allclose function. It presents detailed
statistics about the differences between each pair of arrays.

Created for GPUODEBenchmarks numerical comparison
"""

import os
import sys
import numpy as np
from itertools import combinations

# Windows consoles default to a legacy codepage (cp1252) that cannot encode
# the checkmark glyphs printed below; force UTF-8 where supported.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

def load_data(filepath):
    """Load CSV data file."""
    if not os.path.exists(filepath):
        return None
    return np.loadtxt(filepath, delimiter=',')

def compare_arrays(name1, arr1, name2, arr2, rtol=1e-4, atol=1e-6):
    """Compare two arrays using numpy allclose and compute statistics."""
    print(f"\n{'='*80}")
    print(f"Comparing {name1} vs {name2}")
    print(f"{'='*80}")
    
    # Check shapes match
    if arr1.shape != arr2.shape:
        print(f"ERROR: Shape mismatch! {name1}: {arr1.shape}, {name2}: {arr2.shape}")
        return
    
    print(f"Array shape: {arr1.shape}")
    
    # Compute differences
    diff = np.abs(arr1 - arr2)
    relative_diff = np.abs((arr1 - arr2) / (arr1 + 1e-20))  # Add small epsilon to avoid division by zero
    
    # Run numpy allclose
    is_close = np.allclose(arr1, arr2, rtol=rtol, atol=atol)
    print(f"\nnumpy.allclose(rtol={rtol}, atol={atol}): {is_close}")
    
    # Count how many elements pass the allclose test
    elementwise_close = np.abs(arr1 - arr2) <= (atol + rtol * np.abs(arr2))
    num_close = np.sum(elementwise_close)
    total_elements = arr1.size
    percent_close = 100.0 * num_close / total_elements
    print(f"Elements passing allclose test: {num_close}/{total_elements} ({percent_close:.2f}%)")
    
    # Absolute difference statistics
    print(f"\nAbsolute differences:")
    print(f"  Max:  {np.max(diff):.6e}")
    print(f"  Mean: {np.mean(diff):.6e}")
    print(f"  Min:  {np.min(diff):.6e}")
    print(f"  Std:  {np.std(diff):.6e}")
    
    # Relative difference statistics
    print(f"\nRelative differences:")
    print(f"  Max:  {np.max(relative_diff):.6e}")
    print(f"  Mean: {np.mean(relative_diff):.6e}")
    print(f"  Min:  {np.min(relative_diff):.6e}")
    print(f"  Std:  {np.std(relative_diff):.6e}")
    
    # # Per-state statistics (assuming each row is a trajectory and columns are states)
    # if arr1.ndim == 2:
    #     print(f"\nPer-state statistics (over all trajectories):")
    #     for state_idx in range(arr1.shape[1]):
    #         state_diff = diff[:, state_idx]
    #         print(f"  State {state_idx}: max={np.max(state_diff):.6e}, "
    #               f"mean={np.mean(state_diff):.6e}, min={np.min(state_diff):.6e}")
    
    # Find worst mismatches
    if not is_close:
        print(f"\nWorst mismatches (top 5):")
        flat_diff = diff.flatten()
        worst_indices = np.argsort(flat_diff)[-5:][::-1]
        for idx in worst_indices:
            if arr1.ndim == 2:
                row, col = np.unravel_index(idx, arr1.shape)
                print(f"  [{row}, {col}]: {name1}={arr1[row, col]:.6e}, "
                      f"{name2}={arr2[row, col]:.6e}, diff={flat_diff[idx]:.6e}")
            else:
                print(f"  [{idx}]: {name1}={arr1.flat[idx]:.6e}, "
                      f"{name2}={arr2.flat[idx]:.6e}, diff={flat_diff[idx]:.6e}")
    # Return summary statistics so callers can build pairwise tables
    stats = {
        'is_close': bool(is_close),
        'num_close': int(num_close),
        'total_elements': int(total_elements),
        'percent_close': float(percent_close),
        'abs_max': float(np.max(diff)),
        'abs_mean': float(np.mean(diff)),
        'rel_min': float(np.max(relative_diff)),
        'rel_std': float(np.mean(relative_diff)),
    }
    return stats

# Package prefixes we compare. CSVs are named "<package>_<os>_<gpu>.csv" (the key
# is appended by the benchmark writers, see runner_scripts/bench_key.*). Anything
# not in this set (e.g. mpgos_internalsave) is ignored.
KNOWN_PACKAGES = [
    "cubie_adaptive", "cubie_unadaptive",
    "cubie_mlir_adaptive", "cubie_mlir_unadaptive",
    "myokit_cuda",
    "jax", "pytorch", "julia_adaptive", "julia_fixed", "mpgos",
]


def parse_dataset_filename(fname):
    """Parse "<package>_<os>_<gpu>.csv" into (package, os, gpu).

    The key is "<os>_<gpu>" where gpu contains no underscores, so os and gpu are
    the last two underscore-separated fields and the package is everything before.
    Returns None if the file is unkeyed or the package is not recognised.
    """
    if not fname.endswith(".csv"):
        return None
    stem = fname[:-4]
    parts = stem.split("_")
    if len(parts) < 3:
        return None  # unkeyed legacy file
    gpu = parts[-1]
    os_name = parts[-2]
    package = "_".join(parts[:-2])
    if package not in KNOWN_PACKAGES:
        return None
    return package, os_name, gpu


def build_comparison(group_label, datasets):
    """Run pairwise comparisons for one group and write its markdown table.

    `datasets` maps display-name -> numpy array. Writes
    pairwise_comparisons_<group_label>.md and prints per-pair statistics.
    """
    print(f"\n{'#'*80}")
    print(f"# GROUP: {group_label}  ({len(datasets)} datasets)")
    print(f"{'#'*80}")

    if len(datasets) < 2:
        print(f"Only {len(datasets)} dataset(s) in group '{group_label}'; need at least 2. Skipping.")
        return

    names = sorted(datasets.keys())
    pairs = list(combinations(names, 2))
    stats_map = {}
    for name1, name2 in pairs:
        stats = compare_arrays(name1, datasets[name1], name2, datasets[name2])
        stats_map[(name1, name2)] = stats

    md_path = os.path.join('.', f'pairwise_comparisons_{group_label}.md')
    md_lines = []
    md_lines.append(f'# Pairwise comparisons — {group_label}\n')
    md_lines.append('Generated by `compare_numerical_results.py`.\n\n')
    md_lines.append('## Datasets included\n\n')
    md_lines.append(', '.join(names) + '\n\n')
    md_lines.append('## Pairwise difference table\n\n')
    md_lines.append('| |' + '|'.join(names) + '|\n')
    md_lines.append('|' + '---|' * (len(names) + 1) + '\n')
    for row_name in names:
        cells = [row_name]
        for col_name in names:
            if row_name == col_name:
                cells.append('-')
            else:
                key = (row_name, col_name) if (row_name, col_name) in stats_map else (col_name, row_name)
                stats = stats_map.get(key)
                if stats is None:
                    cells.append('N/A')
                else:
                    cells.append(
                        f"Max: {stats['abs_max']:.2e}<br>Mean: {stats['abs_mean']:.2e}<br>%Close: {stats['percent_close']:.1f}%"
                    )
        md_lines.append('|' + '|'.join(cells) + '|\n')

    try:
        with open(md_path, 'w') as f:
            f.writelines(line for line in md_lines)
        print(f"\nWrote pairwise comparison table to: {md_path}")
    except Exception as e:
        print(f"ERROR: Failed to write markdown file: {e}")


def main():
    """Compare all available numerical results, grouped by os and gpu."""
    data_dir = "./data/numerical"

    print("="*80)
    print("GPU ODE Benchmarks - Numerical Results Comparison")
    print("="*80)
    print(f"\nLooking for keyed CSV files in: {data_dir}")

    if not os.path.isdir(data_dir):
        print(f"\nERROR: {data_dir} does not exist. Run the benchmarks with 32768 trajectories first.")
        sys.exit(1)

    # Discover keyed datasets: (package, os, gpu) -> array
    datasets = []  # list of dicts
    for fname in sorted(os.listdir(data_dir)):
        parsed = parse_dataset_filename(fname)
        if parsed is None:
            continue
        package, os_name, gpu = parsed
        arr = load_data(os.path.join(data_dir, fname))
        if arr is None:
            continue
        key = f"{os_name}_{gpu}"
        datasets.append({"package": package, "os": os_name, "gpu": gpu, "key": key, "arr": arr})
        print(f"✓ Loaded {fname} - package={package}, os={os_name}, gpu={gpu}, shape: {arr.shape}")

    if len(datasets) < 2:
        # Exit 3, not 1, so callers can tell this from a real failure.
        print(f"\nNothing to compare: found {len(datasets)} keyed dataset(s), need at least 2.")
        print("Expected files like <package>_<os>_<gpu>.csv (run benchmarks with 32768 trajectories).")
        sys.exit(3)

    # Build the analysis groups: everything combined, one per os, one per gpu.
    oses = sorted({d["os"] for d in datasets})
    gpus = sorted({d["gpu"] for d in datasets})
    groups = [("all", datasets)]
    for os_name in oses:
        groups.append((os_name, [d for d in datasets if d["os"] == os_name]))
    for gpu in gpus:
        groups.append((gpu, [d for d in datasets if d["gpu"] == gpu]))

    print(f"\nFound {len(datasets)} datasets across {len(oses)} os / {len(gpus)} gpu.")
    print(f"Generating {len(groups)} grouped analyses: {', '.join(g[0] for g in groups)}")

    for group_label, group_datasets in groups:
        # Disambiguate names by key only when the group mixes machines.
        multikey = len({d["key"] for d in group_datasets}) > 1
        named = {}
        for d in group_datasets:
            name = f"{d['package']}@{d['key']}" if multikey else d["package"]
            named[name] = d["arr"]
        build_comparison(group_label, named)

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total datasets: {len(datasets)}")
    print(f"Groups written: {', '.join('pairwise_comparisons_' + g[0] + '.md' for g in groups)}")
    print(f"\nTo adjust comparison tolerances, modify rtol and atol in compare_arrays()")


if __name__ == "__main__":
    main()
