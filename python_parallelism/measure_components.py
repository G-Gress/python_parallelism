"""Small profiling helper to split I/O (load+resize) vs compute time."""

from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union

# Type alias for a path-like object (string or Path).
PathLike = Union[str, Path]

# Aliases for the expected function signatures.
LoadResizeFn = Callable[[PathLike, Tuple[int, int]], Any]
ToGrayFn = Callable[[Any], Any]
SobelEdgesFn = Callable[[Any], Any]


def measure_components(
    paths: Sequence[PathLike] | Iterable[PathLike],
    *,
    load_resize_fn: LoadResizeFn,
    to_gray_fn: ToGrayFn,
    sobel_edges_fn: SobelEdgesFn,
    limit: Optional[int] = 60,
    size: Tuple[int, int] = (224, 224),
    clock: Callable[[], float] = time.perf_counter,
) -> dict:
    """Measure load vs compute time over a sample of images.

    Args:
        paths: Paths to images.
        load_resize_fn: Function that loads an image and resizes it.
            Must accept (path, size=(H, W)) and return an image-like object.
        to_gray_fn: Function that converts the loaded image to grayscale.
        sobel_edges_fn: Function that computes Sobel edges from grayscale.
        limit: Number of images to process. If None, process all.
        size: Target image size (H, W).
        clock: Timing function (defaults to perf_counter).

    Returns:
        Dict with keys: n, load_mean, compute_mean, load_total, compute_total.
        (Matches the schema used in the exploration notebook.)
    """

    # Materialize if needed so we can slice.
    if isinstance(paths, Sequence):
        selected = paths
    else:
        selected = list(paths)

    if limit is not None:
        selected = selected[:limit]

    time_load: list[float] = []
    time_compute: list[float] = []

    for p in selected:
        t0 = clock()
        img = load_resize_fn(p, size)
        t1 = clock()
        gray = to_gray_fn(img)
        sobel_edges_fn(gray)
        t2 = clock()
        time_load.append(t1 - t0)
        time_compute.append(t2 - t1)

    n = len(time_load)
    if n == 0:
        return {
            "n": 0,
            "load_mean": float("nan"),
            "compute_mean": float("nan"),
            "load_total": 0.0,
            "compute_total": 0.0,
        }

    return {
        "n": n,
        "load_mean": statistics.mean(time_load),
        "compute_mean": statistics.mean(time_compute),
        "load_total": float(sum(time_load)),
        "compute_total": float(sum(time_compute)),
    }
