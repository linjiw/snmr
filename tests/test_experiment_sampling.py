from collections import Counter

import pytest

from snmr.experiment import balanced_combination_schedule


def test_balanced_combination_schedule_covers_pairs_and_exposures_equally():
    robots = ("a", "b", "c", "d", "e")

    schedule = balanced_combination_schedule(robots, 2, seed=7)

    assert len(schedule) == 10
    assert len(set(schedule)) == 10
    assert Counter(robot for group in schedule for robot in group) == {
        robot: 4 for robot in robots
    }
    assert schedule == balanced_combination_schedule(robots, 2, seed=7)
    assert schedule != balanced_combination_schedule(robots, 2, seed=8)


@pytest.mark.parametrize(
    ("items", "items_per_step"),
    [
        ((), 1),
        (("a", "a"), 1),
        (("a",), 0),
        (("a",), 2),
    ],
)
def test_balanced_combination_schedule_rejects_invalid_inputs(items, items_per_step):
    with pytest.raises(ValueError):
        balanced_combination_schedule(items, items_per_step, seed=0)
