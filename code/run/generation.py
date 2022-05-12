import pickle
from pathlib import Path

import joblib
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


def load_random(n_actitivities=4, seed=42):
    activities = ['work', 'education', 'business_trip', 'errands', 'escort', 'home', 'shopping', 'leisure']
    flexibilities = ['']
    modes = ['car', 'bicycling', 'transit', 'walking']