"""
Utility functions (timing, logging, etc.)
"""

import time

def time_it(func):
    """Decorator to measure execution time of functions."""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f}s")
        return result
    return wrapper


def _experiment_cpu_heavy_task(x: int) -> int:
    """
    Example CPU-heavy function to simulate work.
    Replace with a real workload later.
    """
    total = 0
    for i in range(10_000):
        total += (x * i) % 97
    return total


def _run_with_multiprocessing():
    """Placeholder: run experiment with multiprocessing."""
    print("🚀 Running with multiprocessing (placeholder)")
    time.sleep(0.5)  # simulate work
    return "multiprocessing result"


def _run_with_threading():
    """Placeholder: run experiment with threading."""
    print("🚀 Running with threading (placeholder)")
    time.sleep(0.5)  # simulate work
    return "threading result"


def _run_with_tensorflow():
    """Placeholder: run experiment with TensorFlow parallelism."""
    print("🚀 Running with TensorFlow (placeholder)")
    time.sleep(0.5)  # simulate work
    return "tensorflow result"


def run_parallel_experiment(use_tf: bool = False):
    """
    Run a parallelism experiment.

    Args:
        use_tf (bool): If True, run TensorFlow-based experiment.
                       If False, run Python-based (multiprocessing/threading) experiment.
    """
    if use_tf:
        result = _run_with_tensorflow()
    else:
        # Default: compare multiprocessing & threading
        result_mp = _run_with_multiprocessing()
        result_th = _run_with_threading()
        result = (result_mp, result_th)

    print(f"✅ Experiment completed: {result}")
    return result
