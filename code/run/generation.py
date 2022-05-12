import pickle
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd

RES_PATH = Path("../res")


def load_example() -> (pd.DataFrame, dict):
    activities_df = pd.read_csv(RES_PATH / f'example_activities.csv')
    tt_driving = pickle.load(open(RES_PATH / f'example_travel_times.pickle', "rb"))
    travel_times_by_mode = {'driving': tt_driving}

    return activities_df, travel_times_by_mode


def load_claire() -> (pd.DataFrame, dict):
    activities_df = pd.read_csv(RES_PATH / "claire_activities.csv")
    _, travel_times_by_mode, _ = joblib.load(RES_PATH / 'claire_preprocessed.joblib', 'r')

    return activities_df, travel_times_by_mode


def load_random(n_activities=4, seed=None) -> (pd.DataFrame, dict):
    num_other_activities = n_activities - 2

    rng = np.random.default_rng(seed=seed)
    dusk, dawn, home_location, next_act_id, rng = generate_dusk_dawn(rng)

    activities = []
    for i in range(num_other_activities):
        act, next_act_id, rng = generate_activity(next_act_id, rng, home_location)
        activities.append(act)

    duplicated_activities = duplicate_activities([dawn] + activities + [dusk])
    activities_df = pd.DataFrame.from_records(duplicated_activities)

    return activities_df, {}


def generate_dusk_dawn(rng: np.random.Generator) -> (dict, dict, Tuple[float, float], int, np.random.Generator):
    home_location = (46.6131, 6.50688)

    dawn = {'act_id': 0, 'act_label': 'dawn', 'start_time': 0, 'end_time': 9.5, 'duration': 9.5}
    dusk = {'act_id': 1, 'act_label': 'dusk', 'start_time': 19, 'end_time': 24, 'duration': 5}
    common = {
        'feasible_start': 0,
        'feasible_end': 24,
        'locations': [home_location],
        'loc_ids': [0],
        'flex_early': 'F',
        'flex_late': 'M',
        'flex_short': 'F',
        'flex_long': 'F',
    }

    dawn, dusk = dict(dawn, **common), dict(dusk, **common)     # Add common part to dusk and dawn

    return dawn, dusk, home_location, 2, rng


def generate_activity(act_id: int, rng: np.random.Generator, home_location: Tuple[float, float]) \
        -> (dict, int, np.random.Generator):
    activities = ['work', 'education', 'business_trip', 'errands', 'escort', 'home', 'shopping', 'leisure']
    flexibilities = ['F', 'M', 'R']
    locations = {0: home_location, 1: (46.5032, 6.4116), 2: (46.6149, 6.50523)}
    max_locations_per_activity = 2

    min_feasibility = 6  # Assume activities are feasible for at least 6 hrs
    feasible_start = rng.integers(0, 24 - min_feasibility)
    feasible_end = rng.integers(feasible_start + min_feasibility, 25)

    des_start = rng.uniform(feasible_start, feasible_end)
    des_end = rng.uniform(des_start, feasible_end)
    des_duration = rng.uniform(min(0.1, des_end - des_start), des_end - des_start)

    num_locations = rng.integers(1, max_locations_per_activity + 1)
    selected_loc_ids = list(rng.choice(list(locations.keys()), size=num_locations, replace=False))
    selected_locations = [locations[loc_id] for loc_id in selected_loc_ids]

    activity = {
        'act_id': act_id,
        'act_label': rng.choice(activities),
        'start_time': des_start,
        'end_time': des_end,
        'duration': des_duration,
        'feasible_start': feasible_start,
        'feasible_end': feasible_end,
        'locations': selected_locations,
        'loc_ids': selected_loc_ids,
        'flex_early': rng.choice(flexibilities),
        'flex_late': rng.choice(flexibilities),
        'flex_short': rng.choice(flexibilities),
        'flex_long': rng.choice(flexibilities),
    }

    return activity, act_id + 1, rng


def duplicate_activities(activities):
    duplicates = []

    for activity in activities:
        activity = activity.copy()
        locations = activity.pop('locations')
        loc_ids = activity.pop('loc_ids')

        for location, loc_id in zip(locations, loc_ids):
            for mode in ['car', 'bicycling', 'transit', 'walking']:
                duplicate = activity.copy()

                act_label = duplicate['act_label']
                label = f'{act_label} @ {loc_id} ({mode})'
                duplicate = dict(duplicate, label=label, location=location, loc_id=loc_id, mode=mode, group=act_label)

                duplicates.append(duplicate)

    return duplicates


