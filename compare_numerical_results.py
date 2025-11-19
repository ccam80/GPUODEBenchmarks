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

def compare_arrays(name1, arr1, name2, arr2, rtol=1e-5, atol=1e-8):
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
    
    # Per-state statistics (assuming each row is a trajectory and columns are states)
    if arr1.ndim == 2:
        print(f"\nPer-state statistics (over all trajectories):")
        for state_idx in range(arr1.shape[1]):
            state_diff = diff[:, state_idx]
            print(f"  State {state_idx}: max={np.max(state_diff):.6e}, "
                  f"mean={np.mean(state_diff):.6e}, min={np.min(state_diff):.6e}")
    
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

def main():
    """Main function to compare all available numerical results."""
    data_dir = "./data/numerical"
    
    # Define expected packages
    packages = ["cubie", "jax", "pytorch", "julia", "mpgos"]
    
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
    
    # Perform pairwise comparisons
    pairs = list(combinations(sorted(data.keys()), 2))
    
    for name1, name2 in pairs:
        compare_arrays(name1, data[name1], name2, data[name2])
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total datasets compared: {len(data)}")
    print(f"Total pairwise comparisons: {len(pairs)}")
    print(f"\nPackages included: {', '.join(sorted(data.keys()))}")
    print(f"\nTo adjust comparison tolerances, modify rtol and atol in compare_arrays()")
    print(f"Current defaults: rtol=1e-5, atol=1e-8")

if __name__ == "__main__":
    main()
