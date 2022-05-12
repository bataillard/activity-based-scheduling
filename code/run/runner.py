import math
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
from run.generation import load_example, load_claire, load_random

OUTPUT_PATH = Path("../out/")

DataLoadFunction = Callable[[], Tuple[pd.DataFrame, dict]]
OptimizerFunction = Callable[[pd.DataFrame, dict], Tuple[int, CpSolver, CpModel, pd.DataFrame]]


@dataclass
class DataSource:
    load_data: DataLoadFunction
    name_prefix: str

    @property
    def output_path(self):
        return OUTPUT_PATH / self.name_prefix


@dataclass
class Model:
    optimize_schedule: OptimizerFunction
    name_prefix: str


EXAMPLE_DATA = DataSource(load_example, 'example')
CLAIRE_DATA = DataSource(load_claire, 'claire')

BASIC_MODEL = Model(basic.optimize_schedule, name_prefix='basic')
INDEXED_MODEL = Model(indexed.optimize_schedule, name_prefix='indexed')
INTERVAL_MODE = Model(interval.optimize_schedule, name_prefix='interval')


def main():
    # Compare basic examples first
    # compare(data_source=EXAMPLE_DATA)
    # compare(data_source=CLAIRE_DATA)

    # Compare runtimes on random data with increasing number of activities
    seed = 42
    min_activities, max_activities = 2, 6

    for n_activities in range(min_activities, max_activities):
        data_name = f'random_{n_activities}'
        activities, travel_times = load_random(n_activities, seed)

        # Create output directory
        output_path = OUTPUT_PATH / data_name
        shutil.rmtree(output_path, ignore_errors=True)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save generated activities and travel times for future reference
        activities.to_csv(output_path / 'data_activities.csv')
        joblib.dump(travel_times, output_path / 'data_travel_times.joblib')

        compare(data_source=DataSource(lambda: (activities, travel_times), data_name), runtime_summary=False)


def compare(data_source: DataSource, n_iter=100, runtime_summary=True):
    print("*" * 100)
    print(f"* Starting comparison for data '{data_source.name_prefix}'".upper())
    print("*" * 100)

    results_path = data_source.output_path / ('results_' + data_source.name_prefix + '.csv')

    basic_times = run_cp(data_source, BASIC_MODEL, n_iter, verbose=10)
    indexed_times = run_cp(data_source, INDEXED_MODEL, n_iter, verbose=10, )
    interval_times = run_cp(data_source, INTERVAL_MODE, n_iter, verbose=10, )
    milp_times = run_milp(data_source, n_iter, verbose=10, )

    times = [s for s in [basic_times, indexed_times, interval_times, milp_times]]
    if runtime_summary:
        times = [s.describe() for s in times]

    times = pd.concat(times, axis=1).T
    times.to_csv(results_path)

    print(f"Comparison for '{data_source.name_prefix}' finished. Results saved to '{results_path}'")


def run_cp(data_source: DataSource, optimizer: Model, n_iter: int, verbose: Union[bool, int] = False,
           print_schedules=True, export_to_csv=True):
    activities, travel_times = data_source.load_data()

    # Create output folders and empty them if necessary
    print_path = data_source.output_path / 'img' / optimizer.name_prefix
    csv_path = data_source.output_path / 'schedules' / optimizer.name_prefix

    shutil.rmtree(print_path, ignore_errors=True)
    shutil.rmtree(csv_path, ignore_errors=True)

    print_path.mkdir(parents=True, exist_ok=True)
    csv_path.mkdir(parents=True, exist_ok=True)

    # Iterate and run model
    if verbose:
        print("*" * 30)
        print(f'* Running model: {optimizer.name_prefix}')
        print("*" * 30 + '\n')

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
             print_schedules=True, export_to_csv=True):
    activities, travel_times = data_source.load_data()

    # Create output folders and empty them if necessary
    print_path = data_source.output_path / 'img' / 'milp'
    csv_path = data_source.output_path / 'schedules' / 'milp'

    shutil.rmtree(print_path, ignore_errors=True)
    shutil.rmtree(csv_path, ignore_errors=True)

    print_path.mkdir(parents=True, exist_ok=True)
    csv_path.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("*" * 30)
        print('* Running model: milp')
        print("*" * 30 + '\n')

    wall_times = []
    for i in range(n_iter):
        schedule, wall_time = milp.optimize_schedule(activities, travel_times)
        wall_times.append(wall_time)

        console_interval = verbose if isinstance(verbose, int) else 10
        if verbose and i % console_interval == 0:
            print(f"= Schedule milp {i}/{n_iter} ================")
            print(schedule)
            print(f"======================================\n")

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


if __name__ == '__main__':
    main()
