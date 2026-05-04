# sbatch --array=0-129 --time=240 --mem-per-cpu=200000 --wrap="python a_training_1.py"

import pandas as pd
from seismostats import Catalog
import numpy as np
import csv
import os
import itertools as it
import time
import sys

import warnings
from seismostats.analysis import (
    BPositiveBValueEstimator,
    estimate_mc_maxc,
    ClassicBValueEstimator)

from functions.transformation_functions import transform_and_rotate
from functions.space_time_separated_map import mac_spacetime

# ===== job_index ===========================
# job_index = int(os.getenv("SLURM_ARRAY_TASK_ID"))
job_index = int(sys.argv[1])
print("running index:", job_index, "type", type(job_index))
t = time.time()

# ===== Changeable Params ===========================
results_dir = "results/training_20260504"

n_time_list = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
n_space_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

param_grid = it.product(
    n_time_list,
    n_space_list
)

param_combinations = list(param_grid)
print(f"{len(param_combinations)} parameter combinations found.")

n_time, n_space = param_combinations[job_index]

# ========== Params ===========================
delta_m = 0.01
dmc = 0.2
fmd_bin = 0.1
correction_factor = 0.2

# ======== get Data =========================
location = 'data/training/Amatrice_CAT5_train.csv'

cat_raw = pd.read_csv(location)
cat_train = Catalog(cat_raw)
cat_train.delta_m = delta_m

# ========== Helpers ==========================
# define overall mc estimator


def estimate_mc(magnitudes):
    mc, _ = estimate_mc_maxc(magnitudes, fmd_bin=fmd_bin,
                             correction_factor=correction_factor)
    return mc


# ======== Partition of Data ======================


# estimate overall mc, filter
cat_train.estimate_mc_maxc(
    fmd_bin=fmd_bin, correction_factor=correction_factor)
cat_train = cat_train[cat_train['magnitude'] > cat_train.mc - delta_m/2]
# coords
coords = [cat_train.x.values, cat_train.y.values, cat_train.z.values]

# eval coords
eval_times = [cat_train.time.max()]
eval_coords = [[coords[0][0]], [coords[1][0]], [coords[2][0]]]

# limits
limits = [
    [cat_train.x.min(), cat_train.x.max()],
    [cat_train.y.min(), cat_train.y.max()],
    [cat_train.z.min(), cat_train.z.max()]]

# length scale
volume = (limits[0][1] - limits[0][0]) * (limits[1][1] - limits[1][0]) * (
    limits[2][1] - limits[2][0])
cell_volume = volume / n_space
s_scale = (cell_volume/np.pi * 3/4)**(1/3)
t_scale = (cat_train.time.max() - cat_train.time.min()) / n_time
t_scale = t_scale / pd.Timedelta(days=1)

# ========= Training ===============================
warnings.filterwarnings('ignore')
if n_time * n_space >= 15 and len(cat_train) / (n_time * n_space) > 4:
    (b_average, b_std,
     mac, mu_mac, std_mac,
     mac_time, mu_mac_time, std_mac_time,
     mac_map) = mac_spacetime(
        coords=coords,
        mags=cat_train.magnitude,
        delta_m=cat_train.delta_m,
        times=cat_train.time,
        limits=limits,
        n_space=n_space,
        n_time=n_time,
        space_realizations=40,
        time_realizations=20,
        eval_coords=eval_coords,
        eval_times=eval_times,
        min_num=50,
        method=BPositiveBValueEstimator,
        mc=cat_train.mc,
        mc_method=estimate_mc,
        transform=True,
        voronoi_method='random',
        time_cut_method='constant_time',
        min_count=20,
        time_bar=False,
        dmc=dmc)
else:
    print('no estimation.')
    b_average = np.nan
    b_std = np.nan
    mac = np.nan
    mu_mac = np.nan
    std_mac = np.nan
    mac_time = np.nan
    mu_mac_time = np.nan
    std_mac_time = np.nan

# save as csv
filename = f"train_n_time{n_time}_n_space{n_space}.csv"
path = os.path.join(results_dir, filename)
with open(path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "s_scale",
        "t_scale",
        "mac",
        "mu_mac",
        "std_mac",
        "mac_time",
        "mu_mac_time",
        "std_mac_time"
    ])
    writer.writerow([
        s_scale,
        t_scale,
        mac,
        mu_mac,
        std_mac,
        mac_time,
        mu_mac_time,
        std_mac_time
    ])

print(f"Results saved to {path} in {time.time() - t} seconds")
