import time

import numpy as np
from docplex.mp.model import Model

from milp.data_utils import cplex_to_df, create_dicts


def optimize_schedule(df=None, travel_times=None, parameters=None, deterministic=False,
                      error_function_type='piecewise'):
    '''
    Optimize schedule using CPLEX solver, given timing preferences and travel time matrix.
    Can produce a graphical output if specified (by argument plot_every)
    travel_times = used to be 2d nest Orig X Dest, changed to 3d nest Mode X Orig X Dest --> need to add mode in dictionary
    '''

    period = 24
    modes = ["driving", "bicycling", "transit", "walking"]
    if parameters is None:
        p_st_e = {'F': 0, 'M': -0.61, 'R': -2.4}  # penalties for early arrival
        p_st_l = {'F': 0, 'M': -2.4, 'R': -9.6}  # penalties for late arrival
        p_dur_s = {'F': -0.61, 'M': -2.4, 'R': -9.6}  # penalties for short duration
        p_dur_l = {'F': -0.61, 'M': -2.4, 'R': -9.6}  # penalties for long duration
        p_t = -1  # penalty for travel time

        error_w = np.random.normal(size=2)
        error_x = np.random.normal(size=4)  # discretization start time: 4h time blocks
        error_d = np.random.normal(size=6)
        error_z = np.random.normal(size=2)

        preferences = None
    else:
        p_st_e = {'F': parameters['p_st_e_f'], 'M': parameters['p_st_e_m'], 'R': parameters['p_st_e_r']}
        p_st_l = {'F': parameters['p_st_l_f'], 'M': parameters['p_st_l_m'], 'R': parameters['p_st_l_r']}
        p_dur_s = {'F': parameters['p_dur_s_f'], 'M': parameters['p_dur_s_m'], 'R': parameters['p_dur_s_r']}
        p_dur_l = {'F': parameters['p_dur_l_f'], 'M': parameters['p_dur_l_m'], 'R': parameters['p_dur_l_r']}

        p_t = parameters['p_t']

        error_w = parameters['error_w']
        error_x = parameters['error_x']
        error_d = parameters['error_d']
        error_z = parameters['error_z']

        pref_st = {1: parameters['d_st_h'], 2: parameters['d_st_w'], 3: parameters['d_st_edu'], 4: parameters['d_st_s'],
                   5: parameters['d_st_er'], 6: parameters['d_st_b'], 8: parameters['d_st_l'], 9: parameters['d_st_es']}

        pref_dur = {1: parameters['d_dur_h'], 2: parameters['d_dur_w'], 3: parameters['d_dur_edu'],
                    4: parameters['d_dur_s'],
                    5: parameters['d_dur_er'], 6: parameters['d_dur_b'], 8: parameters['d_dur_l'],
                    9: parameters['d_dur_es']}

        preferences = [pref_st, pref_dur]

    if deterministic:
        EV_error = 0
    else:
        EV_error = np.random.gumbel()

    # dictionaries containing data
    keys, location, feasible_start, feasible_end, des_start, des_duration, flex_early, flex_late, flex_short, flex_long, group, mode, act_id = create_dicts(
        df, preferences)

    # print(keys, des_start, des_duration, flex_early, flex_late, flex_short, flex_long, mode)

    m = Model()
    m.parameters.optimalitytarget = 3  # global optimum for non-convex models

    # decision variables
    x = m.continuous_var_dict(keys, lb=0, name='x')  # start time
    z = m.binary_var_matrix(keys, keys, name='z')  # activity sequence indicator
    d = m.continuous_var_dict(keys, lb=0, name='d')  # duration
    w = m.binary_var_dict(keys, name='w')  # indicator of  activity choice
    tt = m.continuous_var_dict(keys, lb=0, name='tt')  # travel time
    # md = m.binary_var_matrix(keys, modes, name = 'md') #mode of transportation (availability)
    md_car = m.binary_var_dict(keys, name='md')  # mode of transportation (availability)

    # z_md = m.binary_var_cube(keys, keys, modes, name = 'z_md') #dummy variable to linearize product of z and md

    # piecewise error variables
    # error_w = m.piecewise(0, [(k,error_participation[k]) for k in [0,1]], 0)
    # error_z = m.piecewise(0, [(k,error_succession[k]) for k in [0,1]], 0)
    # error_x = m.piecewise(0, [(a, error_start[b]) for a,b in zip(np.arange(0, 24, 6), np.arange(4))], 0)
    # error_d = m.piecewise(0, [(a, error_duration[b]) for a,b in zip([0, 1, 3, 8, 12, 16], np.arange(6))], error_duration[-1])

    def to_function_points(points, function_type='piecewise'):
        if function_type == 'none':
            return [(0, 0)]
        elif function_type == 'stepwise':
            steps = [[(k0, e0), (k1, e0)] for (k0, e0), (k1, _) in zip(points, points[1:])]
            flat_points = [point for step in steps for point in step]

            return flat_points
        elif function_type == 'piecewise':
            return points
        else:
            raise AttributeError(f'Invalid function type {function_type}')

    w_points = to_function_points([(k, error_w[k]) for k in [0, 1]], error_function_type)
    z_points = to_function_points([(k, error_z[k]) for k in [0, 1]], error_function_type)
    x_points = to_function_points([(a, error_x[b]) for a, b in zip(np.arange(0, 24, 6), np.arange(4))],
                                  error_function_type)
    d_points = to_function_points([(a, error_x[b]) for a, b in zip(np.arange(0, 24, 6), np.arange(4))],
                                  error_function_type)

    error_w = m.piecewise(0, w_points, 0)
    error_z = m.piecewise(0, z_points, 0)
    error_x = m.piecewise(0, x_points, 0)
    error_d = m.piecewise(0, d_points, 0)

    # constraints

    for a in keys:
        ct_sequence = m.add_constraints(z[a, b] + z[b, a] <= 1 for b in keys if b != a)
        ct_sequence_dawn = m.add_constraints(z[a, dawn] == 0 for dawn in keys if group[dawn] == 'dawn')
        ct_sequence_dusk = m.add_constraints(z[dusk, a] == 0 for dusk in keys if group[dusk] == 'dusk')
        ct_sameact = m.add_constraint(z[a, a] == 0)
        ct_times_inf = m.add_constraints(x[a] + d[a] + tt[a] - x[b] >= (z[a, b] - 1) * period for b in keys)
        ct_times_sup = m.add_constraints(x[a] + d[a] + tt[a] - x[b] <= (1 - z[a, b]) * period for b in keys)
        ct_traveltime = m.add_constraint(
            tt[a] == m.sum(z[a, b] * travel_times[mode[a]][location[a]][location[b]] for b in keys))

        if group[a] in ["home", "dawn", "dusk"]:
            ct_car_home = m.add_constraint(md_car[a] == 1)

        if mode[a] == "driving":
            ct_car_avail = m.add_constraint(w[a] <= md_car[a])

        ct_car_consist_neg = m.add_constraints(md_car[a] >= md_car[b] + z[a, b] - 1 for b in keys)
        ct_car_consist_pos = m.add_constraints(md_car[b] >= md_car[a] + z[a, b] - 1 for b in keys)

        ct_nullduration = m.add_constraint(w[a] <= d[a])
        ct_noactivity = m.add_constraint(d[a] <= w[a] * period)
        ct_tw_start = m.add_constraint(x[a] >= feasible_start[a])
        ct_tw_end = m.add_constraint(x[a] + d[a] <= feasible_end[a])

        # if not mtmc: #no duplicates in MTMC !
        ct_duplicates = m.add_constraint(m.sum(w[b] for b in keys if group[b] == group[a]) <= 1)

        if group[a] != 'dawn':
            ct_predecessor = m.add_constraint(m.sum(z[b, a] for b in keys if b != a) == w[a])
        if group[a] != 'dusk':
            ct_successor = m.add_constraint(m.sum(z[a, b] for b in keys if b != a) == w[a])

    ct_period = m.add_constraint(m.sum(d[a] + tt[a] for a in keys) == period)
    ct_startdawn = m.add_constraints(x[dawn] == 0 for dawn in keys if group[dawn] == 'dawn')
    ct_enddusk = m.add_constraints(x[dusk] + d[dusk] == period for dusk in keys if group[dusk] == 'dusk')

    # objective function
    m.maximize(m.sum(w[a] * ((p_st_e[flex_early[a]]) * m.max(des_start[a] - x[a], 0)
                             + (p_st_l[flex_late[a]]) * m.max(x[a] - des_start[a], 0)
                             + (p_dur_s[flex_short[a]]) * m.max(des_duration[a] - d[a], 0)
                             + (p_dur_l[flex_long[a]]) * m.max(d[a] - des_duration[a], 0)
                             + (p_t) * tt[a])
                     + error_w(w[a])
                     + error_x(x[a])
                     + error_d(d[a])
                     + m.sum(error_z(z[a, b]) for b in keys) for a in keys) + EV_error)
    # + error_w*w[a]
    # + error_x*x[a]
    # + error_d*d[a]
    # + m.sum(error_z*z[a,b] for b in keys) for a in keys)+ EV_error)

    start_time = time.time()
    solution = m.solve()
    end_time = time.time()
    figure = None
    solution_df = None
    mode_figure = None

    try:
        solution_value = solution.get_objective_value()
    except:
        solution_value = None
        print('Could not find a solution - see details')
        print(m.solve_details)
        print('------------------')
        raise Exception('Could not find solution')

    solution_df = cplex_to_df(w, x, d, tt, md_car, mode, keys, act_id, location)  # transform into pandas dataframe

    return solution_df, end_time - start_time
