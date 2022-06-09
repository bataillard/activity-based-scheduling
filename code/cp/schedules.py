import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import Normalize, ListedColormap
from ortools.sat.python.cp_model import CpModel, CpSolver

from cp.utils import MAX_TIME, TIME_PERIOD, get_index_col


def model_to_schedule(model: CpModel, solver: CpSolver, activities, w, x, d,
                      location, act_id, mode=None) -> pd.DataFrame:
    schedule = pd.DataFrame(columns=['act_id', 'label', 'start_time', 'end_time', 'duration', 'location', 'mode'])

    if not mode:
        mode = {a: 'driving' for a in activities}

    for idx, a in enumerate(activities):
        if solver.BooleanValue(w[a]):
            schedule.loc[idx, 'act_id'] = act_id[a]
            schedule.loc[idx, 'label'] = a
            schedule.loc[idx, 'start_time'] = solver.Value(x[a])
            schedule.loc[idx, 'duration'] = solver.Value(d[a])
            schedule.loc[idx, 'location'] = location[a]
            schedule.loc[idx, 'mode'] = mode[a]

            schedule.end_time = schedule.start_time + schedule.duration

    return schedule


def model_indexed_to_schedule(model: CpModel, solver: CpSolver, activities, w, x, d, l, m,
                              locations, modes, act_id) -> pd.DataFrame:
    location = {a: locations[solver.Value(l[a])] for a in activities}
    mode = {a: modes[solver.Value(m[a])] for a in activities}

    return model_to_schedule(model, solver, activities, w, x, d, location, act_id, mode)


def plot_schedule(schedule: pd.DataFrame, path='cp_schedule.png'):
    cmap = ListedColormap(sns.color_palette("colorblind").as_hex())
    norm = Normalize(vmin=1, vmax=11)

    idx_col = get_index_col(indexed=False)

    fig = plt.figure(figsize=[20, 3])
    y1 = [0, 0]
    y2 = [1, 1]
    plt.fill_between([0, MAX_TIME + 1], y1, y2, color='silver')

    for idx, row in schedule.iterrows():
        x = [row['start_time'], row['end_time']]
        plt.fill_between(x, y1, y2, color=cmap(norm(row['act_id'])))
        txt_x = np.mean(x)
        txt_y = 1.2
        if 'home' not in row[idx_col]:
            plt.text(txt_x, txt_y, '{}'.format(row[idx_col]), horizontalalignment='center', verticalalignment='center',
                     fontsize=12)  # , fontweight = 'bold')

    plt.xticks(np.arange(0, MAX_TIME + 1, 60 / TIME_PERIOD))
    plt.yticks([])
    plt.xlim([0, MAX_TIME])
    plt.ylim([-1, 2])
    plt.xlabel('Time [h]')
    plt.rcParams['axes.facecolor'] = 'white'

    plt.savefig(path)
    plt.close(fig)
