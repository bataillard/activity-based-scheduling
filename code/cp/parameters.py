import copy
from ast import literal_eval
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd

from cp.utils import hours_to_discrete_time_step, scale_to_discrete_time_step, scale_to_time_step, \
    hours_to_time_step, MAX_TIME, get_index_col


def prepare_data(df: pd.DataFrame, travel_times: dict):
    # Prepare activities dataframe
    modes = ["driving", "bicycling", "transit", "walking"]

    df = df.infer_objects()

    # Convert locations back to tuple as they get imported as strings
    if isinstance(df['location'].iloc[0], str):
        df['location'] = df.location.apply(literal_eval)

    # Ignore mode choice for this example
    if 'mode' not in df.columns:
        df['mode'] = modes[0]

    # Create groups and set first and last to dawn and dusk,
    # as dawn and dusk are allowed to be duplicated
    if 'group' not in df.columns:
        df['group'] = df.act_label.copy()
        df.loc[0, 'group'] = 'dawn'
        df.loc[df.index[-1], 'group'] = 'dusk'

    # Prepare Travel Times dictionary
    travel_times = copy.deepcopy(travel_times)
    for mode, origins in travel_times.items():
        for origin, destinations in origins.items():
            travel_times[mode][origin] = scale_to_discrete_time_step(destinations)

    return df, travel_times


def prepare_indexed_data(df: pd.DataFrame, travel_times: dict) \
        -> (pd.DataFrame, List[Tuple[float, float]], List[int], List[int], List[int]):
    modes = ["driving", "bicycling", "transit", "walking"]

    df = df.infer_objects()

    # Convert locations back to tuple as they get imported as strings
    if isinstance(df['location'].iloc[0], str):
        df['location'] = df.location.apply(literal_eval)

    # Create groups and set first and last to dawn and dusk,
    # as dawn and dusk are allowed to be duplicated
    if 'group' not in df.columns:
        df['group'] = df.act_label.copy()
        df.loc[0, 'group'] = 'dawn'
        df.loc[df.index[-1], 'group'] = 'dusk'

    # Find all locations and index them
    locations = df['location'].unique()
    location_indicies = {loc: idx for idx, loc in enumerate(locations)}

    # Build a list of possible locations for each group of duplicated activities
    activity_locations = {}
    for activity, act_locs in df.groupby('group').location:
        activity_locations[activity] = [location_indicies[loc] for loc in act_locs.unique()]

    # Prepare Travel Times dictionary
    travel_times = copy.deepcopy(travel_times)
    for mode, origins in travel_times.items():
        for origin, destinations in origins.items():
            travel_times[mode][origin] = scale_to_discrete_time_step(destinations)

    # Flatten travel times array (first element is )
    tt_list = [0] * (len(modes) * len(locations) ** 2 + 1)

    for m, mode in enumerate(modes):
        for o, origin in enumerate(locations):
            for d, destination in enumerate(locations):
                index = compute_travel_time_index(locations, m, o, d)
                tt_list[index] = travel_times.get(mode, {}) \
                    .get(origin, {}) \
                    .get(destination, MAX_TIME)

    return df, activity_locations, tt_list, modes, locations


def extract_penalties(parameters: Optional[pd.DataFrame] = None):
    if not parameters:
        p_st_e = {'F': 0, 'M': -0.61, 'R': -2.4}  # penalties for early arrival
        p_st_l = {'F': 0, 'M': -2.4, 'R': -9.6}  # penalties for late arrival
        p_dur_s = {'F': -0.61, 'M': -2.4, 'R': -9.6}  # penalties for short duration
        p_dur_l = {'F': -0.61, 'M': -2.4, 'R': -9.6}  # penalties for long duration
        p_t = hours_to_discrete_time_step(-1)  # penalty for travel time
    else:
        p_st_e = {'F': parameters['p_st_e_f'], 'M': parameters['p_st_e_m'], 'R': parameters['p_st_e_r']}
        p_st_l = {'F': parameters['p_st_l_f'], 'M': parameters['p_st_l_m'], 'R': parameters['p_st_l_r']}
        p_dur_s = {'F': parameters['p_dur_s_f'], 'M': parameters['p_dur_s_m'], 'R': parameters['p_dur_s_r']}
        p_dur_l = {'F': parameters['p_dur_l_f'], 'M': parameters['p_dur_l_m'], 'R': parameters['p_dur_l_r']}

        p_t = hours_to_time_step(parameters['p_t'])

    return scale_to_time_step(p_st_e), scale_to_time_step(p_st_l), scale_to_time_step(p_dur_s), \
           scale_to_time_step(p_dur_l), p_t


def extract_error_terms(deterministic: bool, parameters: Optional[pd.DataFrame] = None):
    if not parameters:
        error_w = np.random.normal(size=2)
        error_x = np.random.normal(size=4)  # discretization start time: 4h time blocks
        error_d = np.random.normal(size=6)
        error_z = np.random.normal(size=2)
    else:
        error_w = parameters['error_w']
        error_x = parameters['error_x']
        error_d = parameters['error_d']
        error_z = parameters['error_z']

    ev_error = 0 if deterministic else np.random.gumbel()

    return hours_to_time_step(error_w), hours_to_time_step(error_x), hours_to_time_step(error_d), hours_to_time_step(
        error_z), hours_to_time_step(ev_error)


def extract_times(activities: pd.DataFrame, parameters: Optional[pd.DataFrame] = None, indexed=False):
    idx_col = get_index_col(indexed)

    if not parameters:
        des_start = activities.set_index(idx_col)['start_time'].to_dict()
        des_duration = activities.set_index(idx_col)['duration'].to_dict()
    else:
        pref_st = {1: parameters['d_st_h'], 2: parameters['d_st_w'], 3: parameters['d_st_edu'], 4: parameters['d_st_s'],
                   5: parameters['d_st_er'], 6: parameters['d_st_b'], 8: parameters['d_st_l'], 9: parameters['d_st_es']}

        pref_dur = {1: parameters['d_dur_h'], 2: parameters['d_dur_w'], 3: parameters['d_dur_edu'],
                    4: parameters['d_dur_s'],
                    5: parameters['d_dur_er'], 6: parameters['d_dur_b'], 8: parameters['d_dur_l'],
                    9: parameters['d_dur_es']}

        des_start = {}
        des_duration = {}

        for i, row in activities.iterrows():
            des_start[row[idx_col]] = pref_st[row.act_id]
            des_duration[row[idx_col]] = pref_dur[row.act_id]

    feasible_start = scale_to_discrete_time_step(activities.set_index(idx_col)['feasible_start'].to_dict())
    feasible_end = scale_to_discrete_time_step(activities.set_index(idx_col)['feasible_end'].to_dict())

    return feasible_start, feasible_end, scale_to_discrete_time_step(des_start), scale_to_discrete_time_step(
        des_duration)


def extract_flexibilities(activities: pd.DataFrame, indexed=False):
    idx_col = get_index_col(indexed)

    flex_early = activities.set_index(idx_col)['flex_early'].to_dict()
    flex_late = activities.set_index(idx_col)['flex_late'].to_dict()
    flex_short = activities.set_index(idx_col)['flex_short'].to_dict()
    flex_long = activities.set_index(idx_col)['flex_long'].to_dict()

    return flex_early, flex_late, flex_short, flex_long


def extract_activities(df: pd.DataFrame):
    idx_col = get_index_col(indexed=False)

    activities = df[idx_col].unique().tolist()
    location = df.set_index(idx_col)['location'].to_dict()
    group = df.set_index(idx_col)['group'].to_dict() if 'group' in df.columns else None
    mode = df.set_index(idx_col)['mode'].to_dict() if 'mode' in df.columns else None
    act_id = df.set_index(idx_col)['act_id'].to_dict()
    is_home = (df.set_index(idx_col)['act_id'] == 'home').to_dict()

    return activities, location, group, mode, act_id, is_home


def extract_indexed_activities(df: pd.DataFrame):
    idx_col = get_index_col(indexed=True)

    activities = df[get_index_col(idx_col)].unique().tolist()
    act_ids = df.groupby(idx_col)['act_id'].first()
    is_home = (act_ids == 'home')

    return activities, act_ids.to_dict(), is_home.to_dict()


def compute_travel_time_index(locations, m, la, lb):
    return (len(locations) ** 2) * m + len(locations) * la + lb
