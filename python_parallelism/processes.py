"""
Processes-based parallelism for applying functions to lists of data in batches.
"""

import inspect
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Callable, Any

def is_batched_function(func: Callable) -> bool:
    """
    Check if the function takes a list or tuple as its first parameter.
    If so, that list or tuple is assumed to be a batch of paths.

    Args:
        func (Callable): The function to inspect.

    Returns:
        bool: True if the function takes a list or tuple as its first parameter, False otherwise.
    """

    # Inspect the function signature to determine if it takes a single path or a list of paths
    signature = inspect.signature(func)
    parameters = list(signature.parameters.values())
    if len(parameters) == 0:
        raise ValueError("The function must have at least one parameter.")

    return (
        len(parameters) > 0 and
        parameters[0].annotation in (list, tuple) or
        parameters[0].name in ('paths', 'batch')
    )

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
    paths: List[Any],
    func: Callable,
    batch_size: int = 16,
    max_workers: int = None,
    func_args: tuple = (),
    func_kwargs: dict = {},
    limit: int = None,
    display_progress: bool = False
) -> List[Any]:
    """
    Takes a list of paths, splits it into batches
    and applies a function to each batch in parallel using ProcessPoolExecutor.
    If the function takes a single path, it will be applied to each item in the batch using batch_worker.
    If the function takes a list of paths, it will be applied directly to each batch.
    Returns a list of results.

    Args:
        paths (List[Any]): The list of paths to process.
        func (Callable): The function to apply to each path.
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
        paths = paths[:limit]

    # Splitting the list of paths into batches of a given size
    batches = [paths[i:i + batch_size] for i in range(0, len(paths), batch_size)]
    results = []

    # Check if the function takes a list of paths or a single path as first argument
    if is_batched_function(func):
        # If the function takes a list of paths, use it directly as the worker
        submit_function = func
    else:
        # If the function takes a single path, use batch_worker as the worker
        submit_function = batch_worker

    # Parallel processing of batches
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(submit_function, batch, *func_args, **func_kwargs)
            for batch in batches
        ]

        for i, future in enumerate(as_completed(futures)):
            batch_result = future.result()
            results.extend(batch_result)
            if display_progress:
                print(f"Completed batch {i+1}/{len(batches)}")

    return results
