import pickle

import pandas as pd
from ortools.sat.python import cp_model

from parameters import extract_penalties, extract_times, extract_error_terms, extract_flexibilities, \
    extract_activities, prepare_data
from schedules import model_to_schedule, plot_schedule
from utils import MAX_TIME


def optimize_schedule(df: pd.DataFrame, travel_times: dict, parameters=None, deterministic=False):
    # ==========================================
    # = Model Parameters and Setup             =
    # ==========================================

    df, travel_times = prepare_data(df, travel_times)

    error_w, error_x, error_d, error_z, ev_error = extract_error_terms(deterministic, parameters)
    feasible_start, feasible_end, des_start, des_duration = extract_times(df, parameters)
    activities, location, group, mode, act_id = extract_activities(df)

    model = cp_model.CpModel()

    # ==========================================
    # = Decision variables                     =
    # ==========================================

    # Activity occurs indicator
    w = {a: model.NewBoolVar(f'w_{a}') for a in activities}

    # Sequencing indicator
    z = {(a, b): model.NewBoolVar(f'z_{a},{b}') for a in activities for b in activities}

    # Start time
    x = {a: model.NewIntVar(0, MAX_TIME, f'x_{a}') for a in activities}

    # Duration
    d = {a: model.NewIntVar(0, MAX_TIME, f'tau_{a}') for a in activities}

    # ==========================================
    # = Constraints                            =
    # ==========================================

    # 11. Durations and travel times sum to time budget
    day_duration = sum(d[a] + z[(a, b)] * travel_times[mode[a]][location[a]][location[b]]
                       for a in activities for b in activities)
    model.Add(MAX_TIME >= day_duration)

    # 12. Dusk and dawn are mandatory
    model.Add(w['dawn'] == 1)
    model.Add(w['dusk'] == 1)

    for a in activities:
        # 13. Activity lasts longer than minimum duration
        model.Add(0 <= d[a]).OnlyEnforceIf(w[a])

        # 14. Activity lasts less than whole day
        model.Add(d[a] <= MAX_TIME).OnlyEnforceIf(w[a])

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
                model.Add(x[a] + d[a] + travel_times[mode[a]][location[a]][location[b]] == x[b]) \
                    .OnlyEnforceIf(z[(a, b)])

        # 21. Only a single duplicate activity is selected
        model.Add(sum(w[b] for b in activities if group[b] == group[a]) <= 1)

        # 22. Activity starts after opening
        model.Add(x[a] >= feasible_start[a])

        # 23. Activity finishes before closing
        model.Add(x[a] + d[a] <= feasible_end[a])

    # ==========================================
    # = Objective function                     =
    # ==========================================

    activity_penalties = create_activity_penalties(df, model, activities, w, x, d, z, location, mode, parameters,
                                                   travel_times)

    error_utility = [
        # TODO piecewise doesn't work when applied to elements
        # error_w(w[a]) +
        # error_x(x[a]) +
        # error_d(d[a]) +
        # sum(error_z(z[(a, b)]) for b in activities)
        # for a in activities
    ]

    # Heavily penalize days lasting longer than MAX_TIME so that model selects the lowest one
    time_over_max = model.NewIntVar(-MAX_TIME, MAX_TIME, 'time_over_max')
    model.Add(time_over_max == (day_duration - MAX_TIME))

    # model.Maximize(ev_error + sum(w[a] * activity_utilities[a] + error_utility[a] for a in activities))
    model.Maximize(ev_error + sum(activity_penalties[a] for a in activities) - 10000 * time_over_max)

    # ==========================================
    # = Solving the problem                    =
    # ==========================================

    solver = cp_model.CpSolver()

    solver.parameters.enumerate_all_solutions = True
    solution_printer = cp_model.VarArrayAndObjectiveSolutionPrinter(w.values())
    status = solver.Solve(model, solution_printer)

    # ==========================================
    # = Printing the solutions                 =
    # ==========================================

    schedule = model_to_schedule(model, solver, activities, w, x, d, location, act_id)
    plot_schedule(schedule)

    return status, solver, model


def create_activity_penalties(df, model, activities, w, x, d, z, location, mode, parameters, travel_times):
    p_st_e, p_st_l, p_dur_s, p_dur_l, p_t = extract_penalties(parameters)
    flex_early, flex_late, flex_short, flex_long = extract_flexibilities(df)
    _, _, des_start, des_duration = extract_times(df, parameters)

    start_time_early = {a: model.NewIntVar(-MAX_TIME, MAX_TIME, f'desired_start_{a} - x_{a}') for a in activities}
    start_time_late = {a: model.NewIntVar(-MAX_TIME, MAX_TIME, f'x_{a} - desired_time_{a}') for a in activities}
    duration_short = {a: model.NewIntVar(-MAX_TIME, MAX_TIME, f'des_duration_{a} - d_{a}]') for a in activities}
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
        model.Add(des_duration[a] - d[a] < 0).OnlyEnforceIf(positive_duration_short)
        model.Add(d[a] - des_duration[a] >= 0).OnlyEnforceIf(positive_duration_long)
        model.Add(d[a] - des_duration[a] < 0).OnlyEnforceIf(positive_duration_long)

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
            p_t * sum(z[(a, b)] * travel_times[mode[a]][location[a]][location[b]] for b in activities)
        for a in activities
    }


if __name__ == '__main__':
    EXAMPLE_PATH = "../milp/example/"
    h = 145440
    activities_df = pd.read_csv(EXAMPLE_PATH + f'{h}.csv')

    tt_driving = pickle.load(open(EXAMPLE_PATH + f'{h}_driving.pickle', "rb"))
    travel_times_by_mode = {'driving': tt_driving}

    status, solver, model = optimize_schedule(activities_df, travel_times_by_mode)

    print(solver.StatusName(status), model.Validate())
