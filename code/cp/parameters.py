from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from utils import hours_to_time_step, discretize_dict
from ast import literal_eval


@dataclass
class Location:
    id: int
    lat: float
    lon: float


@dataclass
class Mode:
    name: str


@dataclass
class Timespan:
    start: int
    end: int
    duration: int = field(init=False)

    def __init__(self, start, end):
        if not start <= end:
            raise AttributeError(f"Start at {start} must occur before end at {end}")

        self.start = start
        self.end = end
        self.duration = self.end - self.start


@dataclass
class Activity:
    activity_id: int
    group: str
    label: str
    location: Location
    mode: Mode
    mandatory: bool
    preferred_start: Timespan
    minimum_duration: int
    desired_duration: Timespan
    feasible_time_range: Timespan


def build_activities(df: pd.DataFrame):
    # TODO clean up other code
    # Build locations
    locations = df.groupby(['loc_id', 'location']).size().reset_index().drop(columns=0)
    locations = locations.location.str.strip('()').str.split(',')
    locations = locations.apply(lambda x: Location(x.loc_id, x.location[0], x.location[1]), axis=1) \
        .set_index(locations.loc_id)

    # Build travel modes
    modes = [Mode('driving')]
    df['mode'] = modes[0].name

    #

    for _, activity in df.iterrows():
        build_activitiy(activity, locations)


def build_activitiy(row: pd.Series, locations: pd.Series):
    activity = Activity(
        activity_id=row['act_id'],
        label=row['label'],
        location=locations[row['loc_id']],
        mode=Mode(row['mode']),
        # TODO complete this
    )


# ========================================================
# Temporary code
# ========================================================


def prepare_data(df: pd.DataFrame, travel_times: dict):
    # Prepare activities dataframe
    modes = ["driving", "bicycling", "transit", "walking"]

    df = df.infer_objects()

    # Convert locations back to tuple as they get imported as strings
    df['location'] = df.location.apply(literal_eval)

    # Ignore mode choice for this example
    df['mode'] = modes[0]

    # Create groups and set first and last to dawn and dusk,
    # as dawn and dusk are allowed to be duplicated
    df['group'] = df.act_label.copy()
    df.loc[0, 'group'] = 'dawn'
    df.loc[df.index[-1], 'group'] = 'dusk'

    # Prepare Travel Times dictionary
    for mode, origins in travel_times.items():
        for origin, destinations in origins.items():
            travel_times[mode][origin] = discretize_dict(destinations)

    return df, travel_times


def extract_penalties(parameters: Optional[pd.DataFrame] = None):
    if not parameters:
        p_st_e = {'F': 0, 'M': -0.61, 'R': -2.4}  # penalties for early arrival
        p_st_l = {'F': 0, 'M': -2.4, 'R': -9.6}  # penalties for late arrival
        p_dur_s = {'F': -0.61, 'M': -2.4, 'R': -9.6}  # penalties for short duration
        p_dur_l = {'F': -0.61, 'M': -2.4, 'R': -9.6}  # penalties for long duration
        p_t = hours_to_time_step(-1)  # penalty for travel time
    else:
        p_st_e = {'F': parameters['p_st_e_f'], 'M': parameters['p_st_e_m'], 'R': parameters['p_st_e_r']}
        p_st_l = {'F': parameters['p_st_l_f'], 'M': parameters['p_st_l_m'], 'R': parameters['p_st_l_r']}
        p_dur_s = {'F': parameters['p_dur_s_f'], 'M': parameters['p_dur_s_m'], 'R': parameters['p_dur_s_r']}
        p_dur_l = {'F': parameters['p_dur_l_f'], 'M': parameters['p_dur_l_m'], 'R': parameters['p_dur_l_r']}

        p_t = hours_to_time_step(parameters['p_t']) # TODO make sure this should be discretized

    return p_st_e, p_st_l, p_dur_s, p_dur_l, p_t


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

    EV_error = 0 if deterministic else np.random.gumbel()

    return error_w, error_x, error_d, error_z, EV_error


def extract_times(activities: pd.DataFrame, parameters: Optional[pd.DataFrame] = None):
    if not parameters:
        des_start = activities.set_index('label')['start_time'].to_dict()
        des_duration = activities.set_index('label')['duration'].to_dict()
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
            des_start[row.label] = pref_st[row.act_id]
            des_duration[row.label] = pref_dur[row.act_id]

    feasible_start = discretize_dict(activities.set_index('label')['feasible_start'].to_dict())
    feasible_end = discretize_dict(activities.set_index('label')['feasible_end'].to_dict())

    return feasible_start, feasible_end, discretize_dict(des_start), discretize_dict(des_duration)


def extract_flexibilities(activities: pd.DataFrame):
    flex_early = activities.set_index('label')['flex_early'].to_dict()
    flex_late = activities.set_index('label')['flex_late'].to_dict()
    flex_short = activities.set_index('label')['flex_short'].to_dict()
    flex_long = activities.set_index('label')['flex_long'].to_dict()

    return flex_early, flex_late, flex_short, flex_long


def extract_activities(df: pd.DataFrame):
    activities = df.label.values.tolist()
    location = df.set_index('label')['location'].to_dict()
    group = df.set_index('label')['group'].to_dict() if 'group' in df.columns else None
    mode = df.set_index('label')['mode'].to_dict() if 'mode' in df.columns else None
    act_id = df.set_index('label')['act_id'].to_dict()

    return activities, location, group, mode, act_id
