import pandas as pd
import numpy as np
from ortools.sat.python import cp_model
from dataclasses import dataclass
from parameters import extract_penalties, extract_times, extract_error_terms, extract_flexibilities, extract_activities, \
    prepare_data

MAX_MINUTES = 24 * 60
TIME_PERIOD = 5
MAX_TIME = MAX_MINUTES / TIME_PERIOD


def optimize_schedule(df: pd.DataFrame, travel_times: dict, parameters=None, deterministic=False):
    # ==========================================
    # = Model Parameters and Setup             =
    # ==========================================

    df, travel_times = prepare_data(df, travel_times)

    p_st_e, p_st_l, p_dur_s, p_dur_l, p_t = extract_penalties(parameters)
    error_w, error_x, error_d, error_z = extract_error_terms(deterministic, parameters)
    feasible_start, feasible_end, des_start, des_duration = extract_times(df, parameters)
    flex_early, flex_late, flex_short, flex_long = extract_flexibilities(df)
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
    t = {a: model.NewIntVar(0, MAX_TIME, f'tau_{a}') for a in activities}

    # ==========================================
    # = Constraints                            =
    # ==========================================

    # 11. Durations and travel times sum to time budget
    model.Add(MAX_TIME == sum(t[a] + z[(a, b)] * travel_times[mode[a]][a][b]
                              for a in activities for b in activities))

    # 12. Dusk and dawn are mandatory
    model.Add(w['dawn'] == 1)
    model.Add(w['dusk'] == 1)

    for a in activities:
        # 13. Activity lasts longer than minimum duration
        # TODO model.AddImplication(w[a], )

        # 14. Activity lasts less than whole day
        model.AddImplication(w[a], t[a] <= MAX_TIME)

        # 15. Activities can only follow each other once
        for b in activities:
            model.AddBoolAnd((z[(a, b)].Not(), z[(b, a)].Not()))

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
                model.AddImplication(z[(a, b)], x[a] + t[a] + travel_times[mode[a]][a][b] == x[b])

        # 21. Only a single duplicate activity is selected
        model.Add(sum(w[b] for b in activities if group[b] == group[a]) <= 1)

        # 22. Activity starts after opening
        model.Add(x[a] >= feasible_start[a])

        # 23. Activity finishes before closing
        model.Add(x[a] + t[a] <= feasible_end[a])