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

def main():
    """Main function to compare all available numerical results."""
    data_dir = "./data/numerical"
    
    # Define expected packages
    packages = ["cubie_adaptive", "cubie_unadaptive", "jax", "pytorch", "julia_adaptive", "julia_fixed", "mpgos"]
    
    print("="*80)
    print("GPU ODE Benchmarks - Numerical Results Comparison")
    print("="*80)
    print(f"\nLooking for CSV files in: {data_dir}")
    
    # Load all available data
    data = {}
    for package in packages:
        filepath = os.path.join(data_dir, f"{package}.csv")
        arr = load_data(filepath)
        if arr is not None:
            data[package] = arr
            print(f"✓ Loaded {package}.csv - shape: {arr.shape}")
        else:
            print(f"✗ {package}.csv not found")
    
    if len(data) < 2:
        print(f"\nERROR: Need at least 2 datasets to compare. Found {len(data)}.")
        print("Please run the benchmarks with 32768 trajectories first.")
        sys.exit(1)
    
    print(f"\nFound {len(data)} datasets. Performing pairwise comparisons...")
    
    # Perform pairwise comparisons and collect stats for a summary table
    names = sorted(data.keys())
    pairs = list(combinations(names, 2))
    stats_map = {}

    for name1, name2 in pairs:
        stats = compare_arrays(name1, data[name1], name2, data[name2])
        stats_map[(name1, name2)] = stats
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total datasets compared: {len(data)}")
    print(f"Total pairwise comparisons: {len(pairs)}")
    print(f"\nPackages included: {', '.join(names)}")

    # Build markdown pairwise table and write to file. Each cell shows: max_abs/mean_abs/%close
    md_path = os.path.join('.', 'pairwise_comparisons.md')
    md_lines = []
    md_lines.append('# Pairwise comparisons\n')
    md_lines.append('Generated by `compare_numerical_results.py`.\n\n')
    md_lines.append('## Packages included\n\n')
    md_lines.append(', '.join(names) + '\n\n')
    md_lines.append('## Pairwise difference table\n\n')

    # Header row
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
                    # Use <br> so each labeled value appears on its own line inside the Markdown cell
                    cells.append(
                        f"Max: {stats['abs_max']:.2e}<br>Mean: {stats['abs_mean']:.2e}<br>%Close: {stats['percent_close']:.1f}%"
                    )
        md_lines.append('|' + '|'.join(cells) + '|\n')

    # Write to file
    try:
        with open(md_path, 'w') as f:
            f.writelines(line for line in md_lines)
        print(f"\nWrote pairwise comparison table to: {md_path}")
    except Exception as e:
        print(f"ERROR: Failed to write markdown file: {e}")

    print(f"\nTo adjust comparison tolerances, modify rtol and atol in compare_arrays()")
    print(f"Current defaults: rtol=1e-5, atol=1e-8")

if __name__ == "__main__":
    main()
