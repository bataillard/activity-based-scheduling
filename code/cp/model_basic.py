import pandas as pd
from ortools.sat.python import cp_model

from cp.parameters import extract_penalties, extract_times, extract_error_terms, extract_flexibilities, \
    extract_activities, prepare_data, compute_piecewise_errors
from cp.schedules import model_to_schedule
from cp.utils import MAX_TIME

MIN_DURATION = 1


def optimize_schedule(df: pd.DataFrame, travel_times: dict, parameters=None, deterministic=False,
                      error_function=compute_piecewise_errors):
    # ==========================================
    # = Model Parameters and Setup             =
    # ==========================================

    df, travel_times = prepare_data(df, travel_times)

    error_w, error_x, error_d, error_z, ev_error = extract_error_terms(deterministic, parameters)
    feasible_start, feasible_end, des_start, des_duration = extract_times(df, parameters)
    activities, location, group, act_mode, act_id, is_home = extract_activities(df)

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
    d = {a: model.NewIntVar(0, MAX_TIME, f'dur_{a}') for a in activities}

    # Uses private mode (car, bike) indicator
    mode_car = {a: model.NewBoolVar(f'car_available_{a}') for a in activities}

    # ==========================================
    # = Constraints                            =
    # ==========================================

    # 11. Durations and travel times sum to time budget
    day_duration = sum(d[a] + sum(z[(a, b)] * travel_times[act_mode[a]][location[a]][location[b]] for b in activities)
                       for a in activities)
    model.Add(MAX_TIME == day_duration)

    # 12. Dusk and dawn are mandatory
    model.Add(sum(w[a] for a in activities if group[a] == 'dawn') >= 1)
    # multiple transport modes
    model.Add(sum(w[a] for a in activities if group[a] == 'dusk') >= 1)

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
        for dawn in filter(lambda a: group[a] == 'dawn', activities):
            model.Add(z[(a, dawn)] == 0)

        for dusk in filter(lambda a: group[a] == 'dusk', activities):
            model.Add(z[(dusk, a)] == 0)

        # 17. Activity has 1 predecessor if selected, 0 otherwise
        if group[a] != 'dawn':
            model.Add(w[a] == sum(z[(b, a)] for b in activities))

        # 18. Activity has 1 successor if selected, 0 otherwise
        if group[a] != 'dusk':
            model.Add(w[a] == sum(z[(a, b)] for b in activities))

        # 19. Activities that follow each other much have matching respective end and start times
        for b in activities:
            if a != b:
                model.Add(x[a] + d[a] + travel_times[act_mode[a]][location[a]][location[b]] == x[b]) \
                    .OnlyEnforceIf(z[(a, b)])

        # 21. Only a single duplicate activity is selected
        model.Add(sum(w[b] for b in activities if group[b] == group[a]) <= 1)

        # 22. Activity starts after opening
        model.Add(x[a] >= feasible_start[a])

        # 23. Activity finishes before closing
        model.Add(x[a] + d[a] <= feasible_end[a])

        # 30. Car is always available at home
        if is_home[a]:
            car_available = 1 if act_mode[a] == 'driving' else 0
            model.Add(mode_car[a] == car_available)

        # 31. Activity using car can only be used if car available
        if act_mode[a] == 'driving':
            model.Add(w[a] == 0).OnlyEnforceIf(mode_car[a].Not())

        # 32. Activity following one that uses the car must also use car
        for b in activities:
            if not is_home[b]:
                model.Add(mode_car[b] == mode_car[a]).OnlyEnforceIf(z[(a, b)])

    # ==========================================
    # = Objective function                     =
    # ==========================================

    activity_penalties = create_activity_penalties(df, model, activities, w, x, d, z, location, act_mode, parameters,
                                                   travel_times)
    error_utility = error_function(model, activities, w, x, d, z, error_w, error_x, error_d, error_z)

    model.Maximize(sum(activity_penalties[a] + error_utility[a] for a in activities) + ev_error)

    # ==========================================
    # = Solving the problem                    =
    # ==========================================

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    # ==========================================
    # = Printing the solutions                 =
    # ==========================================

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        schedule = model_to_schedule(model, solver, activities, w, x, d, location, act_id, act_mode)
    else:
        schedule = pd.DataFrame()

    return status, solver, model, schedule


def create_activity_penalties(df, model, activities, w, x, d, z, location, mode, parameters, travel_times):
    p_st_e, p_st_l, p_dur_s, p_dur_l, p_t = extract_penalties(parameters)
    flex_early, flex_late, flex_short, flex_long = extract_flexibilities(df)
    _, _, des_start, des_duration = extract_times(df, parameters)

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
            p_t * sum(z[(a, b)] * travel_times[mode[a]][location[a]][location[b]] for b in activities)
        for a in activities
    }
