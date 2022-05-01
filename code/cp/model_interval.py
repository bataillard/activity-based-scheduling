import pickle, joblib

import numpy as np
import pandas as pd
from ortools.sat.python import cp_model

from parameters import extract_penalties, extract_times, extract_error_terms, extract_flexibilities, \
    extract_indexed_activities, prepare_indexed_data, compute_travel_time_index
from schedules import plot_schedule, model_indexed_to_schedule
from utils import MAX_TIME, stepwise

TIME_OVER_MAX_PENALTY = 10000
MIN_DURATION = 1
EXAMPLE_PATH = "../milp/example/"
RES_PATH = "../../res/"


def main(example=False):
    if example:
        h = 145440
        activities_df = pd.read_csv(EXAMPLE_PATH + f'{h}.csv')
        tt_driving = pickle.load(open(EXAMPLE_PATH + f'{h}_driving.pickle', "rb"))
        travel_times_by_mode = {'driving': tt_driving}
    else:
        activities_df = pd.read_csv(RES_PATH + "claire_activities.csv")
        _, travel_times_by_mode, _ = joblib.load(RES_PATH + 'claire_preprocessed.joblib', 'r')

    wall_times = []
    n_iter = 100

    for i in range(n_iter):
        status, solver, model, schedule = optimize_schedule(activities_df, travel_times_by_mode)
        if i % 10 == 0:
            print(f"= Schedule {i}/{n_iter} ================")
            print(schedule)
            print()

        wall_times.append(solver.WallTime())

    print(f'Solved in {sum(wall_times) / len(wall_times)}s on average')


def optimize_schedule(df: pd.DataFrame, travel_times_dict: dict, parameters=None, deterministic=False):
    # ==========================================
    # = Model Parameters and Setup             =
    # ==========================================

    df, activity_locations, travel_times, modes, locations = prepare_indexed_data(df, travel_times_dict)

    error_w, error_x, error_d, error_z, ev_error = extract_error_terms(deterministic, parameters)
    feasible_start, feasible_end, des_start, des_duration = extract_times(df, parameters, indexed=True)
    activities, act_id = extract_indexed_activities(df)

    model = cp_model.CpModel()

    # ==========================================
    # = Decision variables                     =
    # ==========================================

    w = {a: model.NewBoolVar(f'w_{a}') for a in activities}

    # Sequencing indicator
    z = {(a, b): model.NewBoolVar(f'z_{a},{b}') for a in activities for b in activities}

    # Start time
    x = {a: model.NewIntVar(0, MAX_TIME, f'x_{a}') for a in activities}

    # Duration
    d = {a: model.NewIntVar(0, MAX_TIME, f'dur_{a}') for a in activities}

    # Travel mode
    m = {a: model.NewIntVar(0, len(modes) - 1, f'mode_{a}') for a in activities}

    # Location
    l = {a: model.NewIntVarFromDomain(cp_model.Domain.FromValues(activity_locations[a]), f'location_{a}')
         for a in activities}

    # Travel time
    t = {(a, b): model.NewIntVar(0, MAX_TIME, f'travel_time_{a}->{b}') for a in activities for b in activities}

    # Intervals
    intervals = {a: create_interval_var(a, x, d, t, w, activities, model) for a in activities}

    # ==========================================
    # = Constraints                            =
    # ==========================================

    # 11. Durations and travel times sum to time budget
    day_duration = sum(d[a] + sum(t[(a, b)] for b in activities)
                       for a in activities)
    model.Add(MAX_TIME == day_duration)

    # 12. Dusk and dawn are mandatory
    model.Add(w['dawn'] == 1)
    model.Add(w['dusk'] == 1)

    for a in activities:
        # 13. Activity lasts longer than minimum duration
        model.Add(MIN_DURATION <= d[a]).OnlyEnforceIf(w[a])

        # 14. Activity lasts less than whole day
        model.Add(d[a] <= MAX_TIME).OnlyEnforceIf(w[a])

        # 14b. Activity has duration 0 if it does not occur
        model.Add(d[a] == 0).OnlyEnforceIf(w[a].Not())

        # 15. Activities can only follow each other once
        for b in activities:
            model.AddBoolOr((z[(a, b)].Not(), z[(b, a)].Not()))

        # 16. Dawn and dusk have no predecessor or successor
        model.Add(z[(a, 'dawn')] == 0)
        model.Add(z[('dusk', a)] == 0)

        # 17. Activity has 1 predecessor if selected, 0 otherwise
        if a != 'dawn':
            model.Add(w[a] == sum(z[(b, a)] for b in activities))

        # 18. Activity has 1 successor if selected, 0 otherwise
        if a != 'dusk':
            model.Add(w[a] == sum(z[(a, b)] for b in activities))

        # 19. Activities that follow each other much have matching respective end and start times
        for b in activities:
            if a != b:
                model.Add(x[a] + d[a] + t[(a, b)] == x[b]) \
                     .OnlyEnforceIf(z[(a, b)])
        model.AddNoOverlap(intervals.values())

        # 22. Activity starts after opening
        model.Add(x[a] >= feasible_start[a])

        # 23. Activity finishes before closing
        model.Add(x[a] + d[a] <= feasible_end[a])

        for b in activities:
            tt_index = model.NewIntVar(0, len(travel_times), f'tt_index_{a}')

            # 25. Travel time to non-sequential activities is zero
            model.Add(tt_index == len(travel_times) - 1).OnlyEnforceIf(z[(a, b)].Not())

            # 24. Travel time between activities depends on mode and locations
            model.Add(tt_index == compute_travel_time_index(locations, m[a], l[a], l[b])).OnlyEnforceIf(z[(a, b)])
            model.AddElement(tt_index, travel_times, t[(a, b)])

        for b in activities:
            model.Add(t[(a, b)] == 0).OnlyEnforceIf(z[(a, b)].Not())

    # ==========================================
    # = Objective function                     =
    # ==========================================

    activity_penalties = create_activity_penalties(df, model, activities, w, x, d, t, parameters)

    error_w_steps = {a: stepwise(model, w[a], 0, [(k, error_w[k]) for k in [0, 1]]) for a in activities}
    error_x_steps = {a: stepwise(model, x[a], 0, [(a, error_x[b]) for a, b in zip(np.arange(0, 24, 6), np.arange(4))])
                     for a in activities}
    error_d_steps = {a: stepwise(model, d[a], 0, [(a, error_d[b]) for a, b in zip([0, 1, 3, 8, 12, 16], np.arange(6))])
                     for a in activities}
    error_z_steps = {(a, b): stepwise(model, z[(a, b)], 0, [(k, error_z[k]) for k in [0, 1]])
                     for b in activities for a in activities}

    error_utility = {
        a: (sum(step.constraint for step in error_w_steps[a])
            + sum(step.constraint for step in error_x_steps[a])
            + sum(step.constraint for step in error_d_steps[a])
            + sum(sum(step.constraint for step in error_z_steps[(a, b)]) for b in activities))
        for a in activities
    }

    model.Maximize(sum(activity_penalties[a] + error_utility[a] for a in activities) + ev_error)
    # model.Maximize(sum(activity_penalties[a] for a in activities) + ev_error)

    # ==========================================
    # = Solving the problem                    =
    # ==========================================

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    # ==========================================
    # = Printing the solutions                 =
    # ==========================================

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        schedule = model_indexed_to_schedule(model, solver, activities, w, x, d, l, m, locations, modes, act_id)
        plot_schedule(schedule)
    else:
        print("Model is", solver.StatusName(status))
        schedule = pd.DataFrame()

    return status, solver, model, schedule


def create_interval_var(a, x, d, t, w, activities, model: cp_model.CpModel):
    duration_and_travel = model.NewIntVar(0, MAX_TIME, f'duration_and_travel_{a}')
    model.Add(duration_and_travel == d[a] + sum(t[(a, b)] for b in activities))

    activity_end = model.NewIntVar(0, MAX_TIME, f'end_time_{a}')
    model.Add(activity_end == x[a] + duration_and_travel)

    return model.NewOptionalIntervalVar(
        start=x[a],
        size=duration_and_travel,
        end=activity_end,
        is_present=w[a],
        name=f'interval_{a}'
    )


def create_activity_penalties(df, model, activities, w, x, d, t, parameters):
    p_st_e, p_st_l, p_dur_s, p_dur_l, p_t = extract_penalties(parameters)
    flex_early, flex_late, flex_short, flex_long = extract_flexibilities(df, indexed=True)
    _, _, des_start, des_duration = extract_times(df, parameters, indexed=True)

    start_time_early = {a: model.NewIntVar(-MAX_TIME, MAX_TIME, f'desired_start_{a} - x_{a}') for a in activities}
    start_time_late = {a: model.NewIntVar(-MAX_TIME, MAX_TIME, f'x_{a} - desired_time_{a}') for a in activities}
    duration_short = {a: model.NewIntVar(-MAX_TIME, MAX_TIME, f'des_duration_{a} - d_{a}') for a in activities}
    duration_long = {a: model.NewIntVar(-MAX_TIME, MAX_TIME, f'd_{a} - des_duration_{a}') for a in activities}

    for a in activities:
        positive_start_early = model.NewBoolVar(f'positive_start_early_{a}')
        positive_start_late = model.NewBoolVar(f'positive_start_late_{a}')
        positive_duration_short = model.NewBoolVar(f'positive_duration_short_{a}')
        positive_duration_long = model.NewBoolVar(f'positive_duration_long_{a}')

        model.Add(des_start[a] - x[a] >= 0).OnlyEnforceIf(positive_start_early)
        model.Add(des_start[a] - x[a] < 0).OnlyEnforceIf(positive_start_early.Not())
        model.Add(x[a] - des_start[a] >= 0).OnlyEnforceIf(positive_start_late)
        model.Add(x[a] - des_start[a] < 0).OnlyEnforceIf(positive_start_late.Not())
        model.Add(des_duration[a] - d[a] >= 0).OnlyEnforceIf(positive_duration_short)
        model.Add(des_duration[a] - d[a] < 0).OnlyEnforceIf(positive_duration_short.Not())
        model.Add(d[a] - des_duration[a] >= 0).OnlyEnforceIf(positive_duration_long)
        model.Add(d[a] - des_duration[a] < 0).OnlyEnforceIf(positive_duration_long.Not())

        time_constraints = [
            model.Add(start_time_early[a] == des_start[a] - x[a]).OnlyEnforceIf(positive_start_early),
            model.Add(start_time_early[a] == 0).OnlyEnforceIf(positive_start_early.Not()),
            model.Add(start_time_late[a] == x[a] - des_start[a]).OnlyEnforceIf(positive_start_late),
            model.Add(start_time_late[a] == 0).OnlyEnforceIf(positive_start_late.Not()),
            model.Add(duration_short[a] == des_duration[a] - d[a]).OnlyEnforceIf(positive_duration_short),
            model.Add(duration_short[a] == 0).OnlyEnforceIf(positive_duration_short.Not()),
            model.Add(duration_long[a] == d[a] - des_duration[a]).OnlyEnforceIf(positive_duration_long),
            model.Add(duration_long[a] == 0).OnlyEnforceIf(positive_duration_long.Not())
        ]

        for tc in time_constraints:
            tc.OnlyEnforceIf(w[a])

        model.Add(start_time_early[a] == 0).OnlyEnforceIf(w[a].Not())
        model.Add(start_time_late[a] == 0).OnlyEnforceIf(w[a].Not())
        model.Add(duration_short[a] == 0).OnlyEnforceIf(w[a].Not())
        model.Add(duration_long[a] == 0).OnlyEnforceIf(w[a].Not())

    return {
        a:
            p_st_e[flex_early[a]] * start_time_early[a] +
            p_st_l[flex_late[a]] * start_time_late[a] +
            p_dur_s[flex_short[a]] * duration_short[a] +
            p_dur_l[flex_long[a]] * duration_long[a] +
            p_t * sum(t[(a, b)] for b in activities)
        for a in activities
    }


if __name__ == '__main__':
    main()
