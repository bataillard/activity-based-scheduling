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
from cp.parameters import compute_piecewise_errors, compute_stepwise_errors, compute_no_activity_errors
from cp.schedules import plot_schedule
from generation import load_example, load_claire, load_random
from milp.data_utils import plot_schedule as milp_plot_schedule

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

pw = lambda model: lambda *args: model.optimize_schedule(*args, error_function=compute_piecewise_errors)
sw = lambda model: lambda *args: model.optimize_schedule(*args, error_function=compute_stepwise_errors)
ne = lambda model: lambda *args: model.optimize_schedule(*args, error_function=compute_no_activity_errors)

BASIC_MODEL_PW = Model(pw(basic), name_prefix='basic_piecewise')
INDEXED_MODEL_PW = Model(pw(indexed), name_prefix='indexed_piecewise')
INTERVAL_MODE_PW = Model(pw(interval), name_prefix='interval_piecewise')

BASIC_MODEL_SW = Model(sw(basic), name_prefix='basic_stepwise')
INDEXED_MODEL_SW = Model(sw(basic), name_prefix='indexed_stepwise')
INTERVAL_MODE_SW = Model(sw(basic), name_prefix='interval_stepwise')

BASIC_MODEL_NE = Model(ne(basic), name_prefix='basic_none')
INDEXED_MODEL_NE = Model(ne(basic), name_prefix='indexed_none')
INTERVAL_MODE_NE = Model(ne(basic), name_prefix='interval_none')


def main():
    # Compare basic examples first
    compare(data_source=EXAMPLE_DATA)
    compare(data_source=CLAIRE_DATA)

    # Compare runtimes on random data with increasing number of activities
    seed = 42
    min_activities, max_activities = 2, 5

    for n_activities in range(min_activities, max_activities + 1):
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

    runtimes_path = data_source.output_path / ('results_runtimes_' + data_source.name_prefix + '.csv')
    objectives_path = data_source.output_path / ('results_objectives_' + data_source.name_prefix + '.csv')

    # Piecewise models
    basic_pw_results = run_cp(data_source, BASIC_MODEL_PW, n_iter, verbose=10)
    indexed_pw_results = run_cp(data_source, INDEXED_MODEL_PW, n_iter, verbose=10, )
    interval_pw_results = run_cp(data_source, INTERVAL_MODE_PW, n_iter, verbose=10, )
    milp_pw_results = run_milp(data_source, n_iter, verbose=10, error_function_type='piecewise')

    # Stepwise models
    basic_sw_results = run_cp(data_source, BASIC_MODEL_SW, n_iter, verbose=10)
    indexed_sw_results = run_cp(data_source, INDEXED_MODEL_SW, n_iter, verbose=10, )
    interval_sw_results = run_cp(data_source, INTERVAL_MODE_SW, n_iter, verbose=10, )
    milp_sw_results = run_milp(data_source, n_iter, verbose=10, error_function_type='stepwise')

    # No activity error models
    basic_ne_results = run_cp(data_source, BASIC_MODEL_NE, n_iter, verbose=10)
    indexed_ne_results = run_cp(data_source, INDEXED_MODEL_NE, n_iter, verbose=10, )
    interval_ne_results = run_cp(data_source, INTERVAL_MODE_NE, n_iter, verbose=10, )
    milp_ne_results = run_milp(data_source, n_iter, verbose=10, error_function_type='none')

    results = [basic_pw_results, indexed_pw_results, interval_pw_results, milp_pw_results,
               basic_sw_results, indexed_sw_results, interval_sw_results, milp_sw_results,
               basic_ne_results, indexed_ne_results, interval_ne_results, milp_ne_results]

    times = [res[0] for res in results]
    objectives = [res[1] for res in results]

    if runtime_summary:
        times = [s.describe() for s in times]
        objectives = [s.describe() for s in objectives]

    times = pd.concat(times, axis=1).T
    times.to_csv(runtimes_path)

    objectives = pd.concat(objectives, axis=1).T
    objectives.to_csv(objectives_path)

    print(f"Comparison for '{data_source.name_prefix}' finished. Results saved to '{runtimes_path}'")


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
    solver_values = []
    for i in range(n_iter):

        # Run solver
        status, solver, model, schedule = optimizer.optimize_schedule(activities, travel_times)
        wall_times.append(solver.WallTime())
        solver_values.append(solver.ObjectiveValue())

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

    return [pd.Series(data=wall_times, index=range(n_iter), name=optimizer.name_prefix),
            pd.Series(data=solver_values, index=range(n_iter), name=optimizer.name_prefix)]


def run_milp(data_source: DataSource, n_iter: int, verbose: Union[bool, int] = False,
             print_schedules=True, export_to_csv=True, error_function_type='piecewise'):
    activities, travel_times = data_source.load_data()

    model_name = f'milp_{error_function_type}'

    # Create output folders and empty them if necessary
    print_path = data_source.output_path / 'img' / model_name
    csv_path = data_source.output_path / 'schedules' / model_name

    shutil.rmtree(print_path, ignore_errors=True)
    shutil.rmtree(csv_path, ignore_errors=True)

    print_path.mkdir(parents=True, exist_ok=True)
    csv_path.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("*" * 30)
        print(f'* Running model: {model_name}')
        print("*" * 30 + '\n')

    wall_times = []
    solver_values = []
    for i in range(n_iter):
        schedule, wall_time, solver_value = milp.optimize_schedule(activities, travel_times,
                                                                   error_function_type=error_function_type)
        wall_times.append(wall_time)
        solver_values.append(solver_value)

        console_interval = verbose if isinstance(verbose, int) else 10
        if verbose and i % console_interval == 0:
            print(f"= Schedule {model_name} {i}/{n_iter} ================")
            print(schedule)
            print(f"======================================\n")

        # Export results of this iteration
        n_zeroes = math.ceil(math.log10(n_iter))
        filename = model_name + "_" + str(i).rjust(n_zeroes, '0')

        if print_schedules:
            path = print_path / (filename + '.png')
            milp_plot_schedule(schedule, path=path)

        if export_to_csv:
            path = csv_path / (filename + '.csv')
            schedule.to_csv(path)

    return [pd.Series(data=wall_times, index=range(n_iter), name=model_name),
            pd.Series(data=solver_values, index=range(n_iter), name=model_name)]


if __name__ == '__main__':
    main()
