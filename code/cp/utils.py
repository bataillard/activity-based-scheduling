from typing import List, Tuple, Callable

from model import TIME_PERIOD


def discretize_dict(d: dict) -> dict:
    return {key: hours_to_time_step(value) for key, value in d.items()}


def hours_to_time_step(hours: float, time_period=TIME_PERIOD) -> int:
    return (hours * 60) // time_period


def piecewise(pre_slope: float, breaks_xy: List[Tuple[float, float]], post_slope: float) -> Callable[[float], float]:
    breaks_xy = sorted(breaks_xy, key=lambda x: x[0])

    def piecewise_func(x: float) -> float:
        x_prev, y_prev = breaks_xy[0]
        if x <= x_prev:
            return pre_slope * (x - x_prev) + y_prev

        for x_next, y_next in breaks_xy:
            if x_prev == x_next and x <= x_next:
                return y_next
            elif x <= x_next:
                slope = (y_next - y_prev) / (x_next - x_prev)
                return slope * (x - x_prev) + y_prev

            x_prev, y_prev = x_next, y_next

        return post_slope * (x - x_prev) + y_prev

    return piecewise_func
