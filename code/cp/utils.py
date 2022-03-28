from model import TIME_PERIOD


def discretize_dict(d: dict) -> dict:
    return {key: hours_to_time_step(value) for key, value in d.items()}


def hours_to_time_step(hours: float, time_period=TIME_PERIOD) -> int:
    return (hours * 60) // time_period
