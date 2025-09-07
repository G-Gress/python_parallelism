from python_parallelism.processes import parallel_map_batched

def single_item_func(x, scale=1):
    return x * scale

def batch_func(xs, scale=1):
    return [x * scale for x in xs]

def test_single_item_func():
    data = list(range(10))
    out = parallel_map_batched(
        data,
        single_item_func,
        is_batched=False,
        batch_size=3,
        func_kwargs={'scale': 2},
        max_workers=2
    )
    assert sorted(out) == [x * 2 for x in data]

def test_batch_func():
    data = list(range(10))
    out = parallel_map_batched(
        data,
        batch_func,
        is_batched=True,
        batch_size=4,
        func_kwargs={'scale': 3},
        max_workers=2
    )
    assert sorted(out) == [x * 3 for x in data]
