from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import pandas as pd

from python_parallelism.benchmarking import load_benchmarks_as_dataframe

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def plot_processpool_full_vs_min_comparison(
    benchmark_path: Path,
    *,
    title: str = "ProcessPool comparison: full vs min, 4w vs 8w",
    ylabel: str = "Mean Execution Time (s)",
    xlabel: str = "Dictionary Kind",
    figsize: tuple[int, int] = (8, 5),
    show: bool = True,
) -> "Axes":
    """Plot mean times for full/min dict preprocessing with 4 vs 8 workers.

    Expects the benchmark file at `benchmark_path` to contain keys:
    - process_full_4w, process_full_8w, process_min_4w, process_min_8w

    Returns the Matplotlib Axes containing the plot.
    """

    df = load_benchmarks_as_dataframe(benchmark_path)

    required_keys = [
        "process_full_4w",
        "process_full_8w",
        "process_min_4w",
        "process_min_8w",
    ]
    missing = [key for key in required_keys if key not in df.index]
    if missing:
        raise KeyError(
            "Missing required benchmark keys in dataframe index: " + ", ".join(missing)
        )

    plot_df = (
        df.loc[required_keys, ["mean"]]
        .assign(
            dict_kind=["Full", "Full", "Minimal", "Minimal"],
            workers=["4", "8", "4", "8"],
        )
        .pivot(index="dict_kind", columns="workers", values="mean")
        .rename(columns={"4": "4 (physical)", "8": "8 (logical)"})
    )

    _, ax = plt.subplots(figsize=figsize)
    plot_df.plot.bar(
        ax=ax,
        rot=0,
        grid=False,
        ylabel=ylabel,
        xlabel=xlabel,
        title=title,
    )
    
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    if show:
        plt.tight_layout()
        plt.show()

    return ax


def plot_speed_comparison_serial_process_threads(
    benchmark_path: Path,
    *,
    title: str = "Speed Comparison: full vs min, serial vs processes vs threads",
    ylabel: str = "Mean Execution Time (s)",
    xlabel: str = "Dictionary Kind",
    figsize: tuple[int, int] = (8, 5),
    show: bool = True,
) -> "Axes":
    """Plot serial vs processes vs threads mean times for full/min dictionaries.

    Expects benchmark keys:
    - serial_full, serial_min
    - process_full_4w, process_min_4w
    - threads_full, threads_min

    Returns the Matplotlib Axes containing the plot.
    """

    df = load_benchmarks_as_dataframe(benchmark_path)

    required_keys = [
        "serial_full",
        "serial_min",
        "process_full_4w",
        "process_min_4w",
        "threads_full",
        "threads_min",
    ]
    missing = [key for key in required_keys if key not in df.index]
    if missing:
        raise KeyError(
            "Missing required benchmark keys in dataframe index: " + ", ".join(missing)
        )

    plot_df = (
        df.loc[required_keys, ["mean"]]
        .assign(
            dict_kind=[
                "Full",
                "Minimal",
                "Full",
                "Minimal",
                "Full",
                "Minimal",
            ],
            method=[
                "Serial",
                "Serial",
                "ProcessPoolExecutor",
                "ProcessPoolExecutor",
                "ThreadPoolExecutor",
                "ThreadPoolExecutor",
            ],
        )
        .pivot(index="dict_kind", columns="method", values="mean")
    )

    _, ax = plt.subplots(figsize=figsize)
    plot_df.plot.bar(
        ax=ax,
        rot=0,
        grid=False,
        ylabel=ylabel,
        xlabel=xlabel,
        title=title,
    )

    ax.grid(axis="y", linestyle="--", alpha=0.5)

    if show:
        plt.tight_layout()
        plt.show()

    return ax


def build_speedup_table_serial_process_threads(benchmark_path: Path) -> pd.DataFrame:
    """Build a compact comparison table with speedup vs the matching serial baseline.

    Returns a DataFrame with rows:
    - Serial / Processes / Threads for each of (full dict, minimal dict)
    and columns: mean, std, speedup_vs_serial.
    """

    df = load_benchmarks_as_dataframe(benchmark_path)

    required_keys = [
        "serial_full",
        "serial_min",
        "process_full_4w",
        "process_min_4w",
        "threads_full",
        "threads_min",
    ]
    missing = [key for key in required_keys if key not in df.index]
    if missing:
        raise KeyError(
            "Missing required benchmark keys in dataframe index: " + ", ".join(missing)
        )

    serial_full_mean = float(df.loc["serial_full", "mean"])
    serial_min_mean = float(df.loc["serial_min", "mean"])

    rows: list[dict[str, object]] = []

    def add_row(label: str, key: str, serial_mean: float) -> None:
        mean = float(df.loc[key, "mean"])
        std = float(df.loc[key, "std"]) if "std" in df.columns else float("nan")
        speedup = serial_mean / mean if mean != 0 else float("inf")
        rows.append(
            {
                "method": label,
                "mean": mean,
                "std": std,
                "speedup_vs_serial": speedup,
            }
        )

    add_row("Serial (full dict)", "serial_full", serial_full_mean)
    add_row("Processes (full dict)", "process_full_4w", serial_full_mean)
    add_row("Threads (full dict)", "threads_full", serial_full_mean)
    add_row("Serial (minimal dict)", "serial_min", serial_min_mean)
    add_row("Processes (minimal dict)", "process_min_4w", serial_min_mean)
    add_row("Threads (minimal dict)", "threads_min", serial_min_mean)

    out = pd.DataFrame(rows).set_index("method")
    return out


def plot_load_vs_compute_stacked_bar(
    results_df: pd.DataFrame,
    *,
    title: str = "Time taken for loading and computing components at different image sizes",
    xlabel: str = "Image Size",
    ylabel: str = "Time (seconds)",
    legend: Sequence[str] = ("Load Time", "Compute Time"),
    figsize: tuple[int, int] = (10, 6),
    show: bool = True,
) -> "Axes":
    """Plot stacked bars for load/compute totals.

    Expects columns: `load_total`, `compute_total`.
    """

    required_cols = ["load_total", "compute_total"]
    missing_cols = [c for c in required_cols if c not in results_df.columns]
    if missing_cols:
        raise KeyError("Missing required columns: " + ", ".join(missing_cols))

    ax = (
        results_df[["load_total", "compute_total"]]
        .plot(kind="bar", stacked=True, figsize=figsize)
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(list(legend))
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    if show:
        plt.tight_layout()
        plt.show()

    return ax


def plot_total_time_vs_batch_size(
    batch_sizes: Sequence[int],
    total_times: Sequence[float],
    *,
    title: str = "Total Time vs Batch Size",
    xlabel: str = "Batch Size",
    ylabel: str = "Total Time (s)",
    marker: str = "o",
    figsize: tuple[int, int] = (8, 5),
    show: bool = True,
) -> "Axes":
    """Simple line plot: total time vs batch size."""

    batch_sizes_list = list(batch_sizes)
    total_times_list = list(total_times)

    _, ax = plt.subplots(figsize=figsize)
    ax.plot(batch_sizes_list, total_times_list, marker=marker)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(batch_sizes_list)

    if len(total_times_list) > 0:
        ymin = min(total_times_list) * 0.95
        ymax = max(total_times_list) * 1.05
        if ymin != ymax:
            ax.set_ylim(ymin, ymax)

    ax.grid(axis="y", linestyle="--", alpha=0.5)

    if show:
        plt.tight_layout()
        plt.show()

    return ax


def plot_batching_comparison_min_full(
    batch_sizes: Sequence[int],
    *,
    min_times: Sequence[float],
    full_times: Sequence[float],
    no_batch_min_time: float,
    no_batch_full_time: float,
    figsize: tuple[int, int] = (12, 4),
    show: bool = True,
) -> tuple["Figure", tuple["Axes", "Axes"]]:
    """Two-subplot view: batching effect for minimal vs full dictionaries."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    ax1.plot(list(batch_sizes), list(min_times), marker="o", label="Minimal dict", linewidth=2, markersize=8)
    ax1.axhline(
        y=no_batch_min_time,
        color="red",
        linestyle="--",
        label="No batching baseline",
        linewidth=2,
    )
    ax1.set_ylabel("Total Time (s)")
    ax1.set_xlabel("Batch Size")
    ax1.set_title("Minimal Dict")
    ax1.set_xticks(list(batch_sizes))
    ax1.legend()
    ax1.grid(axis="y", linestyle="--", alpha=0.3)

    ax2.plot(
        list(batch_sizes),
        list(full_times),
        marker="s",
        color="green",
        label="Full dict",
        linewidth=2,
        markersize=8,
    )
    ax2.axhline(
        y=no_batch_full_time,
        color="red",
        linestyle="--",
        label="No batching baseline",
        linewidth=2,
    )
    ax2.set_ylabel("Total Time (s)")
    ax2.set_xlabel("Batch Size")
    ax2.set_title("Full Dict")
    ax2.set_xticks(list(batch_sizes))
    ax2.legend()
    ax2.grid(axis="y", linestyle="--", alpha=0.3)

    if show:
        plt.tight_layout()
        plt.show()

    return fig, (ax1, ax2)


def plot_cnn_batch_inference_tradeoffs(
    benchmark_path: Path,
    *,
    batch_sizes: Sequence[int],
    sample_size: int,
    per_image_key: str = "extract_features_one",
    batched_key_format: str = "extract_features_batched_{batch_size}",
    show: bool = True,
    print_summary: bool = True,
) -> pd.DataFrame:
    """Build a small table and plot latency/throughput for CNN inference batch sizes.

    Returns `df_plot` with columns: label, batch_size, mean_s, img_per_s, ms_per_img.
    """

    df_benchmark = load_benchmarks_as_dataframe(benchmark_path)

    required_keys = [per_image_key] + [
        batched_key_format.format(batch_size=b) for b in batch_sizes
    ]
    missing = [k for k in required_keys if k not in df_benchmark.index]
    if missing:
        raise KeyError(
            "Missing benchmark keys in benchmark_results.json: " + ", ".join(missing)
        )

    rows: list[dict[str, object]] = []
    t_one = float(df_benchmark.loc[per_image_key, "mean"])
    rows.append(
        {
            "label": "per-image (bs=1)",
            "batch_size": 1,
            "mean_s": t_one,
            "img_per_s": sample_size / t_one,
            "ms_per_img": (t_one / sample_size) * 1000.0,
        }
    )

    for b in batch_sizes:
        key = batched_key_format.format(batch_size=b)
        t = float(df_benchmark.loc[key, "mean"])
        rows.append(
            {
                "label": f"batched (bs={b})",
                "batch_size": int(b),
                "mean_s": t,
                "img_per_s": sample_size / t,
                "ms_per_img": (t / sample_size) * 1000.0,
            }
        )

    df_plot = pd.DataFrame(rows).sort_values("batch_size")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(df_plot["batch_size"], df_plot["mean_s"], marker="o")
    axes[0].set_title(f"Latency for {sample_size} images")
    axes[0].set_xlabel("Batch size")
    axes[0].set_ylabel("Mean time (s)")
    axes[0].set_xticks(df_plot["batch_size"].tolist())
    axes[0].grid(axis="y", linestyle="--", alpha=0.3)

    axes[1].plot(df_plot["batch_size"], df_plot["img_per_s"], marker="o")
    axes[1].set_title("Throughput")
    axes[1].set_xlabel("Batch size")
    axes[1].set_ylabel("Images / second")
    axes[1].set_xticks(df_plot["batch_size"].tolist())
    axes[1].grid(axis="y", linestyle="--", alpha=0.3)

    if show:
        plt.tight_layout()
        plt.show()

    if print_summary:
        best_row = df_plot[df_plot["batch_size"] != 1].sort_values("mean_s").iloc[0]
        speedup = t_one / float(best_row["mean_s"])
        print(
            f"Best among tested batches: bs={int(best_row['batch_size'])} "
            f"({float(best_row['mean_s']):.4f}s for {sample_size} images)"
        )
        print(f"Speedup vs per-image baseline: {speedup:.2f}x")

    return df_plot


def plot_pca_projection(
    features_2d,
    *,
    title: str,
    color: Sequence[int] | None = None,
    cmap: str = "tab10",
    colorbar: bool = False,
    colorbar_label: str | None = None,
    figsize: tuple[int, int] = (8, 6),
    s: int = 10,
    alpha: float = 0.7,
    xlabel: str = "PC1",
    ylabel: str = "PC2",
    show: bool = True,
) -> "Axes":
    """Scatter plot for a 2D PCA projection.

    `features_2d` is expected to be array-like with shape (n_samples, 2).
    """

    _, ax = plt.subplots(figsize=figsize)
    sc = ax.scatter(
        features_2d[:, 0],
        features_2d[:, 1],
        c=list(color) if color is not None else None,
        cmap=cmap if color is not None else None,
        s=s,
        alpha=alpha,
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if colorbar and color is not None:
        cb = plt.colorbar(sc, ax=ax)
        if colorbar_label:
            cb.set_label(colorbar_label)

    if show:
        plt.tight_layout()
        plt.show()

    return ax
