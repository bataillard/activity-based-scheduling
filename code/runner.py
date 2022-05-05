import math
import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Tuple, Union

import joblib
import pandas as pd
from ortools.sat.python import cp_model
from ortools.sat.python.cp_model import CpSolver, CpModel

import cp.model_basic as basic
import cp.model_indexed as indexed
import cp.model_interval as interval
import milp.model as milp
from cp.schedules import plot_schedule
from milp.data_utils import plot_schedule as milp_plot_schedule

DataSource = Callable[[], Tuple[pd.DataFrame, dict]]
OptimizerFunction = Callable[[pd.DataFrame, dict], Tuple[int, CpSolver, CpModel, pd.DataFrame]]


@dataclass
class Model:
    optimize_schedule: OptimizerFunction
    name_prefix: str


RES_PATH = Path("../res/")
OUTPUT_PATH = Path("../out/")
SCHEDULE_OUTPUT_PATH = OUTPUT_PATH / "schedules"
IMAGE_OUTPUT_PATH = OUTPUT_PATH / "img"

BASIC_MODEL = Model(basic.optimize_schedule, name_prefix='basic')
INDEXED_MODEL = Model(indexed.optimize_schedule, name_prefix='indexed')
INTERVAL_MODE = Model(interval.optimize_schedule, name_prefix='interval')


def main(data_source: DataSource, n_iter=100):
    basic_times = run_cp(data_source, BASIC_MODEL, n_iter, verbose=10, print_schedules=True, export_to_csv=True)
    indexed_times = run_cp(data_source, INDEXED_MODEL, n_iter, verbose=10, print_schedules=True, export_to_csv=True)
    interval_times = run_cp(data_source, INTERVAL_MODE, n_iter, verbose=10, print_schedules=True, export_to_csv=True)
    milp_times = run_milp(data_source, n_iter, verbose=10, print_schedules=True, export_to_csv=True)

    times = [s.describe() for s in [basic_times, indexed_times, interval_times, milp_times]]
    times = pd.concat(times, axis=1).T

    times.to_csv(OUTPUT_PATH / 'claire_results.csv')
    print(times)


def run_cp(data_source: DataSource, optimizer: Model, n_iter: int,
           verbose: Union[bool, int] = False, print_schedules=False, export_to_csv=False):
    activities, travel_times = data_source()

    # Create output folders and empty them if necessary
    print_path = IMAGE_OUTPUT_PATH / optimizer.name_prefix
    csv_path = SCHEDULE_OUTPUT_PATH / optimizer.name_prefix

    shutil.rmtree(print_path, ignore_errors=True)
    shutil.rmtree(csv_path, ignore_errors=True)

    print_path.mkdir(parents=True, exist_ok=True)
    csv_path.mkdir(parents=True, exist_ok=True)

    # Iterate and run model
    if verbose:
        print("*" * 30)
        print(f'* Running model: {optimizer.name_prefix}')
        print("*" * 30)

    wall_times = []
    for i in range(n_iter):

        # Run solver
        status, solver, model, schedule = optimizer.optimize_schedule(activities, travel_times)
        wall_times.append(solver.WallTime())

        if status != cp_model.OPTIMAL and status != cp_model.FEASIBLE:
            raise Exception(f'Model is {solver.StatusName(status)}')

        # Print to console if verbose and iteration correct multiple
        console_interval = verbose if isinstance(verbose, int) else 10
        if verbose and i % console_interval == 0:
            print(f"= Schedule {optimizer.name_prefix} {i}/{n_iter} ================")
            print(schedule)
            print("======================================\n")

        # Export results of this iteration
        n_zeroes = math.ceil(math.log10(n_iter))
        filename = optimizer.name_prefix + '_' + str(i).rjust(n_zeroes, '0')

        if print_schedules:
            path = print_path / (filename + '.png')
            plot_schedule(schedule, path=path)

        if export_to_csv:
            path = csv_path / (filename + '.csv')
            schedule.to_csv(path)

    return pd.Series(data=wall_times, index=range(n_iter), name=optimizer.name_prefix)


def run_milp(data_source: DataSource, n_iter: int, verbose: Union[bool, int] = False,
             print_schedules=False, export_to_csv=False):
    activities, travel_times = data_source()

    # Create output folders and empty them if necessary
    print_path = IMAGE_OUTPUT_PATH / 'milp'
    csv_path = SCHEDULE_OUTPUT_PATH / 'milp'

    shutil.rmtree(print_path, ignore_errors=True)
    shutil.rmtree(csv_path, ignore_errors=True)

    print_path.mkdir(parents=True, exist_ok=True)
    csv_path.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("*" * 30)
        print('* Running model: milp')
        print("*" * 30)

    wall_times = []
    for i in range(n_iter):
        schedule, wall_time = milp.optimize_schedule(activities, travel_times)
        wall_times.append(wall_time)

        console_interval = verbose if isinstance(verbose, int) else 10
        if verbose and i % console_interval == 0:
            print(f"= Schedule milp {i}/{n_iter} ================")
            print(schedule)
            print("======================================\n")

        # Export results of this iteration
        n_zeroes = math.ceil(math.log10(n_iter))
        filename = 'milp_' + str(i).rjust(n_zeroes, '0')

        if print_schedules:
            path = print_path / (filename + '.png')
            milp_plot_schedule(schedule, path=path)

        if export_to_csv:
            path = csv_path / (filename + '.csv')
            schedule.to_csv(path)

    return pd.Series(data=wall_times, index=range(n_iter), name='milp')


def load_example() -> (pd.DataFrame, dict):
    activities_df = pd.read_csv(RES_PATH / f'example_activities.csv')
    tt_driving = pickle.load(open(RES_PATH / f'example_travel_times.pickle', "rb"))
    travel_times_by_mode = {'driving': tt_driving}

    return activities_df, travel_times_by_mode


def load_claire() -> (pd.DataFrame, dict):
    activities_df = pd.read_csv(RES_PATH / "claire_activities.csv")
    _, travel_times_by_mode, _ = joblib.load(RES_PATH / 'claire_preprocessed.joblib', 'r')

    return activities_df, travel_times_by_mode


if __name__ == '__main__':
    main(data_source=load_example)
    main(data_source=load_claire)
