import math
import pickle
from dataclasses import dataclass
from typing import Callable, Tuple, Union

import joblib
import pandas as pd
from ortools.sat.python import cp_model
from ortools.sat.python.cp_model import CpSolver, CpModel

import cp.model_basic as basic
import cp.model_indexed as indexed
import cp.model_interval as interval
from cp.schedules import plot_schedule

DataSource = Callable[[], Tuple[pd.DataFrame, dict]]
OptimizerFunction = Callable[[pd.DataFrame, dict], Tuple[int, CpSolver, CpModel, pd.DataFrame]]


@dataclass
class Model:
    optimize_schedule: OptimizerFunction
    name_prefix: str


RES_PATH = "../res/"
OUTPUT_PATH = "../out/"
SCHEDULE_OUTPUT_PATH = OUTPUT_PATH + "schedules/"
IMAGE_OUTPUT_PATH = OUTPUT_PATH + "img/"

BASIC_MODEL = Model(basic.optimize_schedule, name_prefix='basic')
INDEXED_MODEL = Model(indexed.optimize_schedule, name_prefix='indexed')
INTERVAL_MODE = Model(interval.optimize_schedule, name_prefix='interval')


def main():
    n_iter = 11
    basic_times = run_cp(load_claire, BASIC_MODEL, n_iter, verbose=10, print_schedules=True, export_to_csv=True)
    indexed_times = run_cp(load_claire, INDEXED_MODEL, n_iter, verbose=10, print_schedules=True, export_to_csv=True)
    interval_times = run_cp(load_claire, INTERVAL_MODE, n_iter, verbose=10, print_schedules=True, export_to_csv=True)

    print("Basic model:", basic_times.describe())
    print("Indexed model:", indexed_times.describe())
    print("Interval model:", interval_times.describe())


def run_cp(data_source: DataSource, optimizer: Model, n_iter: int,
           verbose: Union[bool, int] = False, print_schedules=False, export_to_csv=False):
    activities, travel_times = data_source()

    if verbose:
        print("*" * 30)
        print(f'* Running model: {optimizer.name_prefix}')
        print("*" * 30)

    wall_times = []
    for i in range(n_iter):
        status, solver, model, schedule = optimizer.optimize_schedule(activities, travel_times)
        wall_times.append(solver.WallTime())

        if status != cp_model.OPTIMAL and status != cp_model.FEASIBLE:
            raise Exception(f'Model is {solver.StatusName(status)}')

        console_interval = verbose if isinstance(verbose, int) else 10
        if verbose and i % console_interval == 0:
            print(f"= Schedule {optimizer.name_prefix} {i}/{n_iter} ================")
            print(schedule)
            print("======================================\n")

        n_zeroes = math.ceil(math.log10(n_iter))

        if print_schedules:
            print_path = IMAGE_OUTPUT_PATH + optimizer.name_prefix + str(i).rjust(n_zeroes, '0') + '.png'
            plot_schedule(schedule, path=print_path)

        if export_to_csv:
            csv_path = SCHEDULE_OUTPUT_PATH + optimizer.name_prefix + str(i).rjust(n_zeroes, '0') + '.csv'
            schedule.to_csv(csv_path)

    return pd.Series(data=wall_times, index=range(n_iter), name=optimizer.name_prefix)


def load_example() -> (pd.DataFrame, dict):
    activities_df = pd.read_csv(RES_PATH + f'example_activities.csv')
    tt_driving = pickle.load(open(RES_PATH + f'example_travel_times.pickle', "rb"))
    travel_times_by_mode = {'driving': tt_driving}

    return activities_df, travel_times_by_mode


def load_claire() -> (pd.DataFrame, dict):
    activities_df = pd.read_csv(RES_PATH + "claire_activities.csv")
    _, travel_times_by_mode, _ = joblib.load(RES_PATH + 'claire_preprocessed.joblib', 'r')

    return activities_df, travel_times_by_mode


if __name__ == '__main__':
    main()
