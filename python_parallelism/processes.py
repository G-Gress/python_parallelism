"""
Processes-based parallelism for applying functions to lists of data in batches.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Callable, Any

def batch_worker(batch: List[Any], func: Callable, *args, **kwargs) -> List[Any]:
    """
    Used as a worker for parallel processing.
    Applies a function to each item of a batch of data.
    Returns a list of results.

    Args:
        batch (List[Any]): The batch of data to process.
        func (Callable): The function to apply to each item in the batch.

    Returns:
        List[Any]: List containing the transformed data.
    """
    return [func(item, *args, **kwargs) for item in batch]

def parallel_map_batched(
    items: List[Any],
    func: Callable,
    is_batched: bool = False,
    batch_size: int = 16,
    max_workers: int = None,
    func_args: tuple = (),
    func_kwargs: dict = {},
    limit: int = None,
    display_progress: bool = False
) -> List[Any]:
    """
    Takes a list of items, splits it into batches
    and applies a function to each batch in parallel using ProcessPoolExecutor.
    If the function takes a single item, it will be applied to each item in the batch using batch_worker.
    If the function takes a list of items, it will be applied directly to each batch.
    Returns a list of results.

    Args:
        items (List[Any]): The list of items to process.
        func (Callable): The function to apply to each item.
        is_batched (bool, optional): Whether the function takes a list of items as its first argument.
                                If None, it will be inferred. Defaults to False.
        batch_size (int, optional): The size of each batch. Defaults to 16.
        max_workers (int, optional): The maximum number of worker processes to use. Defaults to None.
        func_args (tuple, optional): Positional arguments to pass to the function. Defaults to ().
        func_kwargs (dict, optional): Keyword arguments to pass to the function. Defaults to {}.
        limit (int, optional): The maximum number of paths to process. Defaults to None.
        display_progress (bool, optional): Whether to display progress information. Defaults to False.

    Returns:
        List[Any]: The results of applying the function to each batch of paths.
    """

    # Reduces the list of paths to a given limit to avoid processing too many files
    if limit:
        items = items[:limit]

    # Splits the list of paths into batches of a given size
    batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
    results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # If the function takes a list of items, use it directly as the worker
        if is_batched:
            futures = [
                executor.submit(func, batch, *func_args, **func_kwargs)
                for batch in batches
            ]
        # If the function takes a single item, give it to batch_worker
        else:
            futures = [
                executor.submit(batch_worker, batch, func, *func_args, **func_kwargs)
                for batch in batches
            ]

        for i, future in enumerate(as_completed(futures)):
            batch_result = future.result()
            results.extend(batch_result)
            if display_progress:
                print(f"Completed batch {i+1}/{len(batches)}")

    return results
