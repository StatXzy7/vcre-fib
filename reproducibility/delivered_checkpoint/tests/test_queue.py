from sfibai_b.queue import formal_task_order


def test_queue_contains_exactly_the_single_seed_a_through_e_matrix() -> None:
    tasks = formal_task_order()
    assert tasks == [(arm, 2026) for arm in "ABCDE"]
