from dataclasses import dataclass
from typing import List, Tuple

from ortools.sat.python.cp_model import IntVar, CpModel, LinearExpr

TIME_PERIOD = 5
MAX_MINUTES = 24 * 60
MAX_TIME = MAX_MINUTES // TIME_PERIOD


def get_index_col(indexed=False):
    return 'group' if indexed else 'label'


def scale_to_discrete_time_step(d: dict) -> dict:
    return {key: hours_to_discrete_time_step(value) for key, value in d.items()}


def scale_to_time_step(d: dict) -> dict:
    return {key: hours_to_time_step(value) for key, value in d.items()}


def hours_to_discrete_time_step(hours: float, time_period=TIME_PERIOD) -> int:
    return int((hours * 60) // time_period)


def hours_to_time_step(hours: float, time_period=TIME_PERIOD) -> float:
    return (hours * 60) / time_period


def stepwise(model: CpModel, x: IntVar, breaks_xy: List[Tuple[float, float]],
             pre_y=None, post_y=None) -> LinearExpr:
    breaks_xy = sorted(breaks_xy, key=lambda b: b[0])
    constraints = []

    x_prev, y_prev = breaks_xy[0]

    pre_y = y_prev if pre_y is None else pre_y
    post_y = y_prev if post_y is None else post_y

    # 1. Handle case before first point

    # Add indicator for value before first point
    less_than_prev = model.NewBoolVar(f'{x.Name()} < {x_prev}')
    model.Add(x < x_prev).OnlyEnforceIf(less_than_prev)
    model.Add(x >= x_prev).OnlyEnforceIf(less_than_prev.Not())

    # Generate constraint
    constraints.append(less_than_prev * pre_y)

    # 2. Handle case between previous point and next point
    for x_next, y_next in breaks_xy:
        # Combine (!less_than_prev and less_than_next) to make in_interval indicator
        less_than_next = model.NewBoolVar(f'{x.Name()} < {x_next}')
        in_interval = model.NewBoolVar(f'{x_prev} <= {x.Name()} < {x_next}')

        model.Add(x < x_next).OnlyEnforceIf(less_than_next)
        model.Add(x >= x_next).OnlyEnforceIf(less_than_next.Not())
        model.AddBoolAnd([less_than_prev.Not(), less_than_next]).OnlyEnforceIf(in_interval)
        model.AddBoolOr([less_than_prev, less_than_next.Not()]).OnlyEnforceIf(in_interval.Not())

        # Generate constraint
        constraints.append(in_interval * y_prev)

        x_prev, y_prev = x_next, y_next

    # 3. Handle case after last point

    # Add indicator for value after last point
    larger_than_last = model.NewBoolVar(f'{x_prev} <= {x.Name()}')
    model.Add(x >= x_prev).OnlyEnforceIf(larger_than_last)
    model.Add(x < x_prev).OnlyEnforceIf(larger_than_last.Not())

    # Generate constraint
    constraints.append(larger_than_last * post_y)

    return sum(constraints)


def piecewise(model: CpModel, x: IntVar, points_xy: List[Tuple[float, float]],
              pre_slope=0.0, post_slope=0.0) -> LinearExpr:
    points_xy = sorted(points_xy, key=lambda b: b[0])
    constraints = []

    x_prev, y_prev = points_xy[0]

    # 1. Handle case before first point

    # Add indicator for value before first point
    less_than_prev = model.NewBoolVar(f'{x.Name()} < {x_prev}')
    model.Add(x < x_prev).OnlyEnforceIf(less_than_prev)
    model.Add(x >= x_prev).OnlyEnforceIf(less_than_prev.Not())

    # Define difference from first point
    difference = model.NewIntVar(-MAX_TIME, MAX_TIME, f'{x.Name()} - {x_prev}')
    model.Add(difference == x - x_prev).OnlyEnforceIf(less_than_prev)
    model.Add(difference == 0).OnlyEnforceIf(less_than_prev.Not())

    # Generate constraint
    constraints.append(less_than_prev * y_prev + pre_slope * difference)

    # 2. Handle case between previous point and next point
    for x_next, y_next in points_xy[1:]:
        # Combine (!less_than_prev and less_than_next) to make in_interval indicator
        less_than_next = model.NewBoolVar(f'{x.Name()} < {x_next}')
        in_interval = model.NewBoolVar(f'{x_prev} <= {x.Name()} < {x_next}')

        model.Add(x < x_next).OnlyEnforceIf(less_than_next)
        model.Add(x >= x_next).OnlyEnforceIf(less_than_next.Not())
        model.AddBoolAnd([less_than_prev.Not(), less_than_next]).OnlyEnforceIf(in_interval)
        model.AddBoolOr([less_than_prev, less_than_next.Not()]).OnlyEnforceIf(in_interval.Not())

        # Define difference from start point
        interval_diff = model.NewIntVar(-MAX_TIME, MAX_TIME, f'{x.Name()} - {x_prev}')
        model.Add(interval_diff == x - x_prev).OnlyEnforceIf(in_interval)
        model.Add(interval_diff == 0).OnlyEnforceIf(in_interval.Not())

        # Compute slope and generate constraint
        slope = (y_next - y_prev) / (x_next - x_prev)
        constraint = in_interval * y_prev + slope * interval_diff
        constraints.append(constraint)

        x_prev, y_prev = x_next, y_next

    # 3. Handle case after last point

    # Add indicator for value after last point
    larger_than_last = model.NewBoolVar(f'{x_prev} <= {x.Name()}')
    model.Add(x >= x_prev).OnlyEnforceIf(larger_than_last)
    model.Add(x < x_prev).OnlyEnforceIf(larger_than_last.Not())

    # Define difference from last point
    difference = model.NewIntVar(-MAX_TIME, MAX_TIME, f'{x.Name()} - {x_prev}')
    model.Add(difference == x - x_prev).OnlyEnforceIf(larger_than_last)
    model.Add(difference == 0).OnlyEnforceIf(larger_than_last.Not())

    # Generate constraint
    constraints.append(larger_than_last * y_prev + post_slope * difference)

    return sum(constraints)




