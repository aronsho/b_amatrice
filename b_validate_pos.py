# sbatch --array=0-119  --time=960 --mem-per-cpu=150000 --wrap="python c_validate.py"

import pandas as pd
from seismostats import Catalog
import numpy as np
import csv
import os
import itertools as it
import time
from scipy import stats

import warnings
from seismostats.analysis import (
    BPositiveBValueEstimator,
    ClassicBValueEstimator,
    estimate_mc_maxc
)

from functions.transformation_functions import transform_and_rotate
from functions.space_time_separated_map import mac_spacetime
from functions.general_functions import likelihood_exp
from functions.main_functions import loglik_test, positive_test

# ===== job_index ===========================
job_index = int(os.getenv("SLURM_ARRAY_TASK_ID"))
print("running index:", job_index, "type", type(job_index))
t = time.time()

# ===== Changeable Params ===========================
results_dir = "results/validation_pos_20260504"

n_time_list = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
n_space_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

param_grid = it.product(
    n_time_list,
    n_space_list
)

param_combinations = list(param_grid)
print(f"{len(param_combinations)} parameter combinations found.")

n_time, n_space = param_combinations[job_index]

# ========== Training Params ===========================
mc_train = 0.3

# ========== Overall Params ===========================
delta_m = 0.01
dmc = 0.2
correction_factor = 0.2
fmd_bin = 0.1

step = 1000  # discretization of evaluation times in order to save computation
mc_chosen_classic = 2.7
mc_chosen_positive = 1.0

# ======== get Data =========================
# == Train==
location = 'data/training/Amatrice_CAT5_train.csv'
cat_raw = pd.read_csv(location)
cat_traintest = Catalog(cat_raw)
cat_traintest.delta_m = delta_m

# == use only 60% for training, 40% for validation
training_share = 0.6
cat_train = cat_traintest.iloc[:int(len(cat_traintest)*training_share)].copy()
cat_test = cat_traintest.iloc[int(len(cat_traintest)*training_share):].copy()

# ========== Helpers ==========================


def estimate_mc(magnitudes):
    mc, _ = estimate_mc_maxc(magnitudes, fmd_bin=fmd_bin,
                             correction_factor=correction_factor)
    return mc


# ======== Preparation of Data ======================
# ======TRAIN related========
# estimate overall b-values (training)
estimator = ClassicBValueEstimator()
_ = estimator.calculate(
    cat_train.magnitude, mc=mc_train, delta_m=cat_train.delta_m)
b_all_classic = estimator.b_value

estimator = BPositiveBValueEstimator()
_ = estimator.calculate(
    cat_train.magnitude, mc=mc_train, delta_m=cat_train.delta_m, dmc=dmc)
b_all_positive = estimator.b_value


# ======TEST related========
# remove mags below overall mc for cat_test
cat_test.mc = mc_chosen_positive
cat_test = cat_test[cat_test['magnitude'] > cat_test.mc - delta_m/2]
coords_test = [cat_test.x.values, cat_test.y.values, cat_test.z.values]

# eval coords
eval_times = cat_test.time.values
eval_coords = coords_test

# ======TRAINTEST related========
# limits
limits = [
    [cat_traintest.x.min(), cat_traintest.x.max()],
    [cat_traintest.y.min(), cat_traintest.y.max()],
    [cat_traintest.z.min(), cat_traintest.z.max()]]

# estimate differences for cat_traintest
cat_traintest = cat_traintest[cat_traintest['magnitude']
                              > mc_train - delta_m/2]
cat_traintest = cat_traintest.sort_values(by='time').reset_index(drop=True)
cat_traintest['magnitude'] = cat_traintest['magnitude'].diff()
cat_traintest = cat_traintest.iloc[1:]
cat_traintest.mc = dmc
cat_traintest = cat_traintest[cat_traintest['magnitude']
                              > cat_traintest.mc - delta_m/2]
coords_traintest = [
    cat_traintest.x.values, cat_traintest.y.values, cat_traintest.z.values]


# Scale n_time  correctly (n_space is same since volume is similar)
training_period = cat_train.time.max() - cat_train.time.min()
trainval_period = cat_traintest.time.max() - cat_traintest.time.min()
n_time_scaled = int(trainval_period / training_period * n_time)

# length scale
volume = (limits[0][1] - limits[0][0]) * (limits[1][1] - limits[1][0]) * (
    limits[2][1] - limits[2][0])
cell_volume = volume / n_space
s_scale = (cell_volume/np.pi * 3/4)**(1/3)
t_scale = (cat_traintest.time.max() - cat_traintest.time.min()) / n_time_scaled
t_scale = t_scale / pd.Timedelta(days=1)

# ========= Training ===============================
warnings.filterwarnings('ignore')
if n_time * n_space >= 15 and len(cat_traintest) / (n_time * n_space) > 8:
    warnings.filterwarnings('ignore')
    (b_average_matrix, b_std_matrix,
     mac, mu_mac, std_mac,
     mac_time, mu_mac_time, std_mac_time,
     mac_map) = mac_spacetime(
        coords=coords_traintest,
        mags=cat_traintest.magnitude,
        delta_m=cat_traintest.delta_m,
        times=cat_traintest.time,
        limits=limits,
        n_space=n_space,
        n_time=n_time_scaled,
        space_realizations=40,
        time_realizations=20,
        eval_coords=eval_coords,
        eval_times=eval_times[::step],
        min_num=50,
        method=ClassicBValueEstimator,
        mc=cat_traintest.mc,
        mc_method=estimate_mc,
        transform=True,
        voronoi_method='random',
        time_cut_method='constant_time',
        min_count=20,
        time_bar=False)

    # estimate b_average for all eval points
    b_average = np.ones(len(eval_times)) * np.nan
    new_time = np.repeat(eval_times[::step], step + 1)[:len(eval_times)]
    for ii in range(np.shape(b_average_matrix)[0]):
        time_loop = eval_times[::step][ii]
        b_average[new_time == time_loop] = b_average_matrix[ii, :][
            new_time == time_loop]

    # esimate classic IG
    IG_classic = cat_test.copy()
    IG_classic = loglik_test(
        IG_classic, b_average, b_all_classic, mc_chosen=mc_chosen_classic)

    # esimate positive IG
    IG_positive = cat_test.copy()
    IG_positive = positive_test(
        IG_positive, b_average, b_all_positive,
        mc_chosen=mc_chosen_positive,  dmc=dmc)

    # paired t-test for all IGs
    idx_nan = IG_classic['loglike'].isna()
    sample_classic = IG_classic['loglike'][~idx_nan]
    _, p_classic = stats.ttest_1samp(
        sample_classic, popmean=0, alternative='greater')

    idx_nan = IG_positive['loglike'].isna()
    sample_positive = IG_positive['loglike'][~idx_nan]
    _, p_positive = stats.ttest_1samp(
        sample_positive, popmean=0, alternative='greater')

    # estimate sum
    IG_classic = IG_classic['loglike'].sum()
    IG_positive = IG_positive['loglike'].sum()

    # estimate IG per earthquake
    IG_classic_norm = IG_classic / len(sample_classic)
    IG_positive_norm = IG_positive / len(sample_positive)

else:
    print('no estimation.')
    IG_classic = np.nan
    IG_positive = np.nan
    IG_classic_norm = np.nan
    IG_positive_norm = np.nan
    p_classic = np.nan
    p_positive = np.nan


# save as csv
filename = f"valid_n_time{n_time}_n_space{n_space}.csv"
path = os.path.join(results_dir, filename)
with open(path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "s_scale",
        "t_scale",
        "n_time_scaled",
        "n_space",
        "IG_classic",
        "IG_positive",
        "p_classic",
        "p_positive",
        "IG_classic_norm",
        "IG_positive_norm",
    ])
    writer.writerow([
        s_scale,
        t_scale,
        n_time_scaled,
        n_space,
        IG_classic,
        IG_positive,
        p_classic,
        p_positive,
        IG_classic_norm,
        IG_positive_norm,
    ])

print(f"Results saved to {path} in {time.time() - t} seconds")
