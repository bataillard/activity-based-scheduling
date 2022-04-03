from dataclasses import dataclass
from typing import List, Tuple

from ortools.sat.python.cp_model import IntVar, CpModel

TIME_PERIOD = 5
MAX_MINUTES = 24 * 60
MAX_TIME = MAX_MINUTES // TIME_PERIOD


def scale_to_discrete_time_step(d: dict) -> dict:
    return {key: hours_to_discrete_time_step(value) for key, value in d.items()}


def scale_to_time_step(d: dict) -> dict:
    return {key: hours_to_time_step(value) for key, value in d.items()}


def hours_to_discrete_time_step(hours: float, time_period=TIME_PERIOD) -> int:
    return int((hours * 60) // time_period)


def hours_to_time_step(hours: float, time_period=TIME_PERIOD) -> float:
    return (hours * 60) / time_period


@dataclass
class Step:
    in_interval: IntVar
    value: float

    @property
    def constraint(self):
        return self.in_interval * self.value


def stepwise(model: CpModel, x: IntVar, pre_y: float, breaks_xy: List[Tuple[float, float]]) -> List[Step]:
    breaks_xy = sorted(breaks_xy, key=lambda b: b[0])
    steps = []

    x_prev, y_prev = breaks_xy[0]
    less_than_prev = model.NewBoolVar(f'{x.Name()} < {x_prev}')
    model.Add(x < x_prev).OnlyEnforceIf(less_than_prev)
    model.Add(x >= x_prev).OnlyEnforceIf(less_than_prev.Not())
    steps.append(Step(less_than_prev, pre_y))

    for x_next, y_next in breaks_xy:
        less_than_next = model.NewBoolVar(f'{x.Name()} < {x_next}')
        model.Add(x < x_next).OnlyEnforceIf(less_than_next)
        model.Add(x >= x_next).OnlyEnforceIf(less_than_next.Not())

        in_interval = model.NewBoolVar(f'{x_prev} <= {x.Name()} < {x_next}')

        model.AddBoolAnd(less_than_prev.Not(), less_than_next).OnlyEnforceIf(in_interval)
        model.AddBoolOr(less_than_prev, less_than_next.Not()).OnlyEnforceIf(in_interval.Not())

        steps.append(Step(in_interval, y_prev))
        x_prev, y_prev = x_next, y_next

    larger_than_last = model.NewBoolVar(f'{x_prev} <= {x.Name()}')
    model.Add(x >= x_prev).OnlyEnforceIf(larger_than_last)
    model.Add(x < x_prev).OnlyEnforceIf(larger_than_last.Not())
    steps.append(Step(larger_than_last, y_prev))

    return steps
