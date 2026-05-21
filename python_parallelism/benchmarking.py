"""Benchmark helpers used by notebooks and scripts."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from collections.abc import Sequence

import numpy as np
import pandas as pd

try:
    from IPython.display import clear_output  # type: ignore
except Exception:  # pragma: no cover
    clear_output = None

def benchmark_function(func, *args, n_runs=5, verbose=True, **kwargs):
    """
    Run a function multiple times and return timing statistics.
    
    Args:
        func: Function to benchmark
        *args: Positional arguments to pass to func
        n_runs: Number of times to run the function (default: 5)
        verbose: Whether to print verbose output (default: True)
        **kwargs: Keyword arguments to pass to func
    
    Returns:
        dict: Contains 'mean', 'std', 'min', 'max', and 'n_runs' of the execution times
        list: List of execution times for each run
    """
    times = []
    for i in range(n_runs):
        if verbose:
            print(f"Run {i+1}/{n_runs}")
        t0 = time.perf_counter()
        func(*args, **kwargs)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        if verbose:
            print(f"Run {i+1}/{n_runs} took {times[-1]:.4f}s")
    
    if verbose and clear_output is not None:
        clear_output()
    
    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
        'n_runs': n_runs
    }, times
    
def display_benchmarks_results(benchmarks):
    """
    Display benchmark results in a readable format.
    
    Args:
        benchmarks: Dictionary of benchmark results, where keys are method names and values are dictionaries with timing statistics.
    """
    for method, stats in benchmarks.items():
        print(f"{method} ({stats['n_runs']} runs):\n"
                f"mean = {stats['mean']:.4f}s\n"
                f"std  = {stats['std']:.4f}s\n"
                f"min  = {stats['min']:.4f}s\n"
                f"max  = {stats['max']:.4f}s")

def save_benchmark(
    key: str,
    stats: dict,
    times: list,
    benchmark_path: Path,
    sample_size: int = 1250,
) -> None:
    """
    Save a benchmark result under the given key in the shared JSON file.
    Existing entries are preserved; only the key is updated.

    Args:
        key:         Descriptive name for this benchmark (e.g. "serial full dict")
        stats:       The stats dict returned by benchmark_function (mean, std, ...)
        times:       The list of per-run execution times returned by benchmark_function
        benchmark_path: Path to the JSON file where results will be saved
        sample_size: Optional override for the sample size stored in JSON.
    """

    benchmark_path = Path(benchmark_path)
    benchmark_path.parent.mkdir(parents=True, exist_ok=True)

    data: dict[str, Any] = {}
    if benchmark_path.exists():
        with open(benchmark_path) as f:
            data = json.load(f)
    data[key] = {
        "sample size": int(sample_size),
        "stats": {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                  for k, v in stats.items()},
        "times": [float(t) for t in times],
    }
    with open(benchmark_path, "w") as f:
        json.dump(data, f, indent=2)

def load_benchmarks_as_dataframe(json_path: Path) -> pd.DataFrame:
    """
    Load saved benchmarks from a JSON file into a Pandas dataframe.

    Args:
        json_path (Path): The path to the JSON file to load

    Returns:
        pd.DataFrame: The loaded benchmarks as a Pandas dataframe, with the "stats" column flattened into separate columns
    """
    # Load the json into a dataframe
    df = pd.read_json(json_path).T

    # Flatten the "stats" column into separate columns in the dataframe
    stats_columns = df["stats"].apply(pd.Series)

    # Concatenate the original dataframe with the new stats columns and drop the original "stats" column
    df = pd.concat(
        [df.drop(columns=["stats"]), stats_columns],
        axis=1
    )
    
    return df


def find_best_batch_size(
    df_benchmark: pd.DataFrame,
    *,
    key_format: str,
    batch_sizes: Sequence[int],
    metric: str = "mean",
    strict: bool = True,
) -> tuple[int, float, pd.Series]:
    """Find the batch size with the lowest metric.

    Args:
        df_benchmark: Dataframe returned by `load_benchmarks_as_dataframe` (index is benchmark keys).
        key_format: Format string such as "batched_process_min_{batch_size}".
        batch_sizes: Candidate batch sizes.
        metric: Column in df_benchmark to optimize (default: "mean").
        strict: If True, raise if any expected key is missing. If False, ignore missing keys.

    Returns:
        (best_batch_size, best_value, series_of_values_by_batch_size)
    """

    values: dict[int, float] = {}
    missing: list[str] = []
    for b in batch_sizes:
        key = key_format.format(batch_size=int(b))
        try:
            values[int(b)] = float(df_benchmark.loc[key, metric])
        except KeyError:
            missing.append(key)

    if strict and missing:
        raise KeyError(
            "Missing benchmark keys in df_benchmark: " + ", ".join(missing)
        )
    if not values:
        raise ValueError("No benchmark values found for the provided batch sizes.")

    series = pd.Series(values).sort_index()
    best_batch_size = int(series.idxmin())
    best_value = float(series.loc[best_batch_size])
    return best_batch_size, best_value, series


def build_batching_overhead_summary(
    df_benchmark: pd.DataFrame,
    *,
    batch_sizes: Sequence[int],
    minimal_key_format: str = "batched_process_min_{batch_size}",
    full_key_format: str = "batched_process_full_{batch_size}",
    minimal_baseline_key: str = "process_min_4w",
    full_baseline_key: str = "process_full_4w",
    metric: str = "mean",
) -> pd.DataFrame:
    """Build a compact table comparing batching for minimal vs full dict."""

    best_min_bs, best_min_time, _ = find_best_batch_size(
        df_benchmark,
        key_format=minimal_key_format,
        batch_sizes=batch_sizes,
        metric=metric,
    )
    best_full_bs, best_full_time, _ = find_best_batch_size(
        df_benchmark,
        key_format=full_key_format,
        batch_sizes=batch_sizes,
        metric=metric,
    )

    no_batch_min_time = float(df_benchmark.loc[minimal_baseline_key, metric])
    no_batch_full_time = float(df_benchmark.loc[full_baseline_key, metric])

    out = pd.DataFrame(
        {
            "baseline_s": [no_batch_min_time, no_batch_full_time],
            "best_batch_size": [best_min_bs, best_full_bs],
            "best_s": [best_min_time, best_full_time],
        },
        index=pd.Index(["minimal", "full"], name="data"),
    )
    out["speedup_vs_baseline"] = out["baseline_s"] / out["best_s"]
    return out
