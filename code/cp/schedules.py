import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib.colors import Normalize, ListedColormap
from ortools.sat.python.cp_model import CpModel, CpSolver

from utils import MAX_TIME, TIME_PERIOD


def model_to_schedule(model: CpModel, solver: CpSolver, activities, w, x, d, location, act_id) -> pd.DataFrame:
    schedule = pd.DataFrame(columns=['act_id', 'label', 'start_time', 'end_time', 'duration', 'location'])

    for idx, a in enumerate(activities):
        if solver.BooleanValue(w[a]):
            schedule.loc[idx, 'act_id'] = act_id[a]
            schedule.loc[idx, 'label'] = a
            schedule.loc[idx, 'start_time'] = solver.Value(x[a])
            schedule.loc[idx, 'duration'] = solver.Value(d[a])
            schedule.loc[idx, 'location'] = location[a]

            schedule.end_time = schedule.start_time + schedule.duration

    return schedule


def plot_schedule(schedule: pd.DataFrame):
    cmap = ListedColormap(sns.color_palette("colorblind").as_hex())
    norm = Normalize(vmin=1, vmax=11)

    fig = plt.figure(figsize=[20, 3])
    y1 = [0, 0]
    y2 = [1, 1]
    plt.fill_between([0, MAX_TIME+1], y1, y2, color='silver')

    for idx, row in schedule.iterrows():
        x = [row['start_time'], row['end_time']]
        plt.fill_between(x, y1, y2, color=cmap(norm(row['act_id'])))
        txt_x = np.mean(x)
        txt_y = 1.2
        if 'home' not in row['label']:
            plt.text(txt_x, txt_y, '{}'.format(row['label']), horizontalalignment='center', verticalalignment='center',
                     fontsize=12)  # , fontweight = 'bold')

    plt.xticks(np.arange(0, MAX_TIME+1, 60 / TIME_PERIOD))
    plt.yticks([])
    plt.xlim([0, MAX_TIME])
    plt.ylim([-1, 2])
    plt.xlabel('Time [h]')

    plt.savefig('cp_schedule.png')
    plt.close(fig)
