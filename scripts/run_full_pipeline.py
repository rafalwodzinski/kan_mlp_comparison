#!/usr/bin/env python3
"""
Master Orchestration Script for 1-Click End-to-End Reproducibility.
Executes the complete experimental lifecycle sequentially:
  1. Main 3x5 Repeated Stratified CV Benchmark (all 11 models & 7 datasets with Optuna HPO)
  2. Dataset Size Scarcity Ablation Study (all 11 models & 6 data fractions)
  3. Analytical Report & Visualization Generation (CSVs, statistical tests, boxplots, radar charts, curves)
"""

import os
import sys
import time
import subprocess
from typing import List, Tuple

def format_duration(seconds: float) -> str:
    """Formats time duration in seconds into human-readable hours, minutes, and seconds."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs:.1f}s"
    elif minutes > 0:
        return f"{minutes}m {secs:.1f}s"
    else:
        return f"{secs:.1f}s"

def print_banner(title: str, subtitle: str = ""):
    """Prints a visually distinct ASCII banner."""
    width = 80
    print("\n" + "=" * width)
    print(f" {title.upper()} ".center(width, "*"))
    if subtitle:
        print(f" {subtitle} ".center(width))
    print("=" * width + "\n", flush=True)

def run_step(step_num: int, total_steps: int, title: str, script_rel_path: str, root_dir: str) -> float:
    """
    Executes a single pipeline step as a subprocess, streaming output in real-time.
    
    Args:
        step_num (int): Current step index (1-based).
        total_steps (int): Total number of pipeline steps.
        title (str): Human-readable name of the step.
        script_rel_path (str): Relative path to the python script from root_dir.
        root_dir (str): Absolute path to project root directory.
        
    Returns:
        float: Elapsed time in seconds.
    """
    script_abs_path = os.path.join(root_dir, script_rel_path)
    if not os.path.exists(script_abs_path):
        print(f"\n[CRITICAL ERROR] Target script not found: {script_abs_path}", flush=True)
        sys.exit(1)
        
    print_banner(
        f"STEP {step_num}/{total_steps}: {title}", 
        f"Target: {script_rel_path}"
    )
    
    start_time = time.time()
    
    try:
        # Execute script using sys.executable to ensure the same Python environment is used.
        # Inheriting stdout/stderr by default streams outputs in real-time to the console.
        process = subprocess.run(
            [sys.executable, script_rel_path],
            cwd=root_dir,
            check=True
        )
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print("\n" + "!" * 80)
        print(f" [PIPELINE FAILURE] Step {step_num} ({title}) aborted with exit code {e.returncode} ".center(80, "!"))
        print(f" Time elapsed before failure: {format_duration(elapsed)} ".center(80))
        print("!" * 80 + "\n", flush=True)
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\n[ABORTED] Pipeline execution interrupted by user (Ctrl+C).", flush=True)
        sys.exit(130)
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Unexpected error while executing {script_rel_path}: {str(e)}", flush=True)
        sys.exit(1)
        
    elapsed = time.time() - start_time
    print(f"\n[STEP {step_num} COMPLETED] Duration: {format_duration(elapsed)}")
    print("-" * 80, flush=True)
    return elapsed

def main():
    """Main pipeline execution workflow."""
    # Resolve project root directory regardless of where the script is invoked from
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    print_banner(
        "KAN vs. MLP MEDICAL BENCHMARK: FULL PIPELINE ORCHESTRATION",
        f"Project Root: {root_dir}"
    )
    
    pipeline_steps: List[Tuple[str, str]] = [
        (
            "Main 3x5 Repeated Stratified CV Benchmark (11 Models, 7 Datasets, Optuna HPO)",
            "scripts/automate_benchmark.py"
        ),
        (
            "Dataset Size Scarcity Ablation Study (11 Models, 6 Data Fractions)",
            "scripts/run_ablation_study.py"
        ),
        (
            "Analytical Report, Statistical Tests & Publication Figure Generation",
            "scripts/generate_report.py"
        )
    ]
    
    total_steps = len(pipeline_steps)
    step_times: List[Tuple[str, float]] = []
    
    pipeline_start_time = time.time()
    
    for idx, (title, script_path) in enumerate(pipeline_steps, start=1):
        elapsed_sec = run_step(idx, total_steps, title, script_path, root_dir)
        step_times.append((title, elapsed_sec))
        
    total_elapsed = time.time() - pipeline_start_time
    
    # Print final summary banner
    print_banner(
        "PIPELINE EXECUTION SUCCESSFULLY COMPLETED",
        f"Total Duration: {format_duration(total_elapsed)}"
    )
    
    print("Execution Breakdown:")
    for idx, (title, duration) in enumerate(step_times, start=1):
        print(f"  [{idx}/{total_steps}] {title:<65} : {format_duration(duration)}")
    print("\nAll benchmark tables, statistical reports, and publication plots have been generated in 'results/'.\n", flush=True)

if __name__ == "__main__":
    main()
