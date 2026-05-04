import numpy as np
import inspect
import warnings
from tqdm import tqdm
from typing import Callable
import pandas as pd

from seismostats.analysis.bvalue import (
    ClassicBValueEstimator, BValueEstimator)
from seismostats.analysis.avalue import AValueEstimator
from functions.general_functions import (
    update_welford, finalize_welford)
from functions.space_functions import (
    mirror_voronoi,
    find_nearest_vor_node,
    find_points_in_tile,
    volumes_vor,
)
from functions.one_dimensional import cut_constant_value
from seismostats.analysis.b_significant import (
    values_from_partitioning,
    est_morans_i,
    cut_constant_idx,
    transform_n)


def estimate_mc_local(tile_mags, mc_min, mc_method):
    '''
    Helper function to estimate local mc values.
    '''
    if mc_method is None:
        return None
    mc_local = np.full(len(tile_mags), np.nan, dtype=float)
    for i, tiles in enumerate(tile_mags):
        if len(tiles) > 0:
            mc_local[i] = max(mc_min, mc_method(tiles))
    return mc_local


def mac_spacetime(
        coords: np.ndarray,
        mags: np.ndarray,
        delta_m: float,
        times: np.ndarray,
        limits: np.ndarray,
        n_space: int,
        n_time: int,
        space_realizations: int,
        time_realizations: int,
        eval_coords: np.ndarray | None = None,
        eval_times: np.ndarray | None = None,
        min_num: int = 10,
        method: BValueEstimator | AValueEstimator = ClassicBValueEstimator,
        mc: float | None = None,
        mc_method: Callable | None = None,
        transform: bool = True,
        scaling_factor: float = 1.0,
        voronoi_method: str = 'random',
        time_cut_method: str = 'constant_idx',
        min_count: int = 1,
        time_bar: bool = False,
        **kwargs,
):
    """
    This function estimates the mean autocorrelation (Moran's I) for the
    D-dimensional case (tested for 2 and 3 dimensions). Additionally, it
    provides the mean a- and b-values for each grid-point. The partitioning
    method is based on voronoi tessellation.

    Source:
        - Mirwald et al. (2024), How to b-significant when analyzing b-value
            variations

    Args:
        coords:         Coordinates of the earthquakes. It should have the
            structure [x1, ... , xD], where xi are vectors of the same length
            (N = number of events)
        mags:           Magnitudes of the earthquakes (length N)
        delta_m:        Magnitude bin width
        times:          Times of the earthquakes (length N)
        limits:         Limits of the area of interest. It should be a list
            with the minimum and maximum values of each variable.
            [[x1min, x1max], ..., [xDmin, xDmax]] (D = number of dimensions).
            The limits should be such that all the coordinates are within
            the limits.
        n_space:        Number of voronoi nodes
        n_m_time:       Number of events in each time window
        n_realizations: Number of realizations of random voronoi tessellation
            for the estimation of the mean values
        n_skip_time:    Number of time steps to skip (if n_skip_time=1, all
            possible partitions in time are considered)
        eval_coords:    Coordinates of the grid-points where the mean a- and
            b-values are estimated. It should have the structure
            [x1, ..., xD], where xi are vectors length M (M = number of
            grid-points). If None, the earthquake coordinates are used.
        eval_time:      Times at which the mean a- and b-values are estimated.
            It should have the structure [t1, ..., tK], where ti are the
            times at which the mean values are estimated. (K = number of time
            steps). If None, earthquake times are used.
        min_num:        Minimum number of events to estimate a- and b-values in
            each tile
        method:         Method to estimate b-values. Options are "positive" and
            "classic"
        mc:             Completeness magnitude. If mc_method is also provided,
            max(mc, mc_method) is used.
        mc_method:      Method to estimate the completeness magnitude. needs
            to be a function that takes the magnitudes as input and returns
            the  mc
        transform:      If True, the b-values are transformed according to the
            number of events used to estimate them (such that their
            distribution is IID under the null hypothesis of unchanging
            b-value). IF a-values are estimated, transform is set to False.
        scaling_factor: Scaling factor for the a-value estimation
        voronoi_method: Method to partition the area. Options are 'random'
            (random area) and 'location' (based on density of events). Default
            is 'random'.
        time_cut_method: Method to cut the time series. Options are
            1. 'constant_idx' (constant number of events in each time window)
            2. 'constant_time' (constant time window)
            3. 'eval_times' (eval_times are used to cut the time series)
                The times are cut such that that each value is the right edge
                of the time window. Note that n_time is ignored in this case.
            The default is 'constant_idx'.
        min_count:     Minimum number of realizations required to estimate
            the mean and standard deviation.
        time_bar:      If True, a progress bar is shown during the computation.
        **kwargs:       additional keyword arguments for b-value(a-value)
            method

    Returns:
        x_average:      Mean a- or b-values at each grid-point (np.array with
            shape M x K: [x_(t1), .., x_(tK)], where x_i are vectors of size M)
        x_std:          Standard deviation of a- or b-values at each grid-point
            (size M x K, same structure as x_average)
        mac:            Mean spatial autocorrelation (Moran's I)
        mu_mac:         Expected mean spatial autocorrelation under the null
            hypothesis
        std_mac:       Expected standard deviation of spatial autocorrelation
            under the null hypothesis
        mac_time:      Mean temporal autocorrelation (Moran's I)
        mu_mac_time:   Expected mean temporal autocorrelation under the null
            hypothesis
        std_mac_time:       Expected standard deviation of temporal
            autocorrelation under the null hypothesis
        mac_map:       Local temporal mean autocorrelation map (at each
            evaluation grid-point, size M). The mac is transformed such that
            under the null hypothesis of no temporal variation, it has mean 0
            and standard deviation 1.
    """

    # 1. preparation
    # convert all to np.dnarrays
    mags = np.asarray(mags)
    times = np.asarray(times)
    times = pd.to_datetime(times).values.astype("datetime64[ns]")
    coords = np.asarray(coords)
    limits = np.asarray(limits)
    eval_coords = np.asarray(
        eval_coords) if eval_coords is not None else coords
    if time_cut_method == 'eval_times':
        if eval_times is None:
            raise ValueError(
                "If time_cut_method is 'eval_times', eval_times must be "
                "provided")

    eval_times = np.asarray(
        eval_times) if eval_times is not None else times
    eval_times = pd.to_datetime(eval_times).values.astype("datetime64[ns]")

    # dimensions (2 and 3D possible)
    dim, n_events = coords.shape
    n_eval_space = len(eval_coords[0, :])
    n_eval_time = len(eval_times)
    x_is_a = issubclass(method, AValueEstimator)
    x_is_b = issubclass(method, BValueEstimator)

    # some data checks
    if len(mags) != n_events:
        raise ValueError("The number of magnitudes and coordinates do not "
                         "match")
    if len(times) != n_events:
        raise ValueError("The number of times and coordinates do not match")
    if len(limits) != dim:
        raise ValueError("The number of limits and dimensions do not match")
    if len(eval_coords[:, 0]) != dim:
        raise ValueError("The number of evaluation coordinates and dimensions "
                         "do not match")

    for ii in range(dim):
        if np.min(eval_coords[ii, :]) < limits[ii][0] or np.max(
                eval_coords[ii, :]) > limits[ii][1]:
            raise ValueError(
                "The evaluation coordinates are outside the limits")
        if np.min(coords[ii, :]) < limits[ii][0] or np.max(
                coords[ii, :]) > limits[ii][1]:
            raise ValueError(
                "The earthquake coordinates are outside the limits")

    # estimate overall mc
    if mc is not None:
        if mc_method is not None:
            warnings.warn(
                "Both mc and mc_method are provided."
                "In this case, max(mc, mc_method) is used for local estimates")
            mc_min = mc
        elif mc_method is None:
            mc_local = mc
    elif mc is None and mc_method is None:
        raise ValueError(
            "Either mc or mc_method should be provided")

    # estimate overall b-value for transformation if needed
    if x_is_a:
        transform = False
    elif x_is_b and transform:
        estimator = method()
        sig = inspect.signature(estimator.calculate)
        if 'times' in sig.parameters:
            b_all = estimator.calculate(
                mags, mc=mc, delta_m=delta_m, times=times, **kwargs)
        else:
            b_all = estimator.calculate(mags, mc=mc, delta_m=delta_m, **kwargs)

    # order everything in time
    idx = np.argsort(times)
    times = times[idx]
    coords = coords[:, idx]
    mags = mags[idx]

    # initiate the aggregates
    aggregate = [tuple(np.zeros(n_eval_space) for _ in range(3))
                 for _ in range(n_eval_time)]
    aggregate_std = [tuple(np.zeros(n_eval_space) for _ in range(3))
                     for _ in range(n_eval_time)]
    agg_ac, agg_n_p, agg_n = [0, 0, 0], [0, 0, 0], [0, 0, 0]
    agg_ac_temporal, agg_n_p_temporal, agg_n_temporal = [
        0, 0, 0], [0, 0, 0], [0, 0, 0]
    agg_mac_map = [np.zeros(n_eval_space),
                   np.zeros(n_eval_space),
                   np.zeros(n_eval_space)]

    # estimate positions of time cuts and evaluation times
    all_time_steps = []
    if time_cut_method == 'constant_idx':
        n_m_time = len(mags) // n_time
        time_realizations = min(time_realizations, n_m_time)
        n_skip_time = max(1, int(n_m_time / time_realizations))
        time_steps = np.arange(0, n_m_time, n_skip_time)
        for ii in range(1, n_time+1):
            shifted = time_steps - 1 + ii * n_m_time
            all_time_steps.append(shifted)
        all_time_steps = np.concatenate(all_time_steps)
        all_time_steps = all_time_steps[all_time_steps < n_events].astype(int)
    elif time_cut_method == 'constant_time':
        length_time = (np.max(times) - np.min(times)) / n_time
        delta_skip_time = length_time / time_realizations
        time_steps = np.arange(np.min(times), np.min(
            times) + length_time, delta_skip_time)
        for ii in range(1, n_time+1):
            shifted = time_steps + ii * length_time
            all_time_steps.append(shifted)
        all_time_steps = np.concatenate(all_time_steps)
        all_time_steps = all_time_steps[all_time_steps < np.max(times)]
        all_time_steps = np.searchsorted(
            times, all_time_steps, side='right') - 1
    elif time_cut_method == 'eval_times':
        if np.all(eval_times[:-1] <= eval_times[1:]):
            all_time_steps = np.searchsorted(
                times, eval_times, side='right') - 1
            time_steps = [0]
        else:
            raise ValueError("eval_times must be sorted")
    all_time_steps = all_time_steps[all_time_steps != -1]

    idx_tmp = np.searchsorted(
        times[all_time_steps], eval_times, side='right')
    idx_tmp -= 1

    idx_eval = all_time_steps[idx_tmp]
    idx_eval[idx_tmp == -1] = 0  # these are not evaulated

    # progress bar
    if time_bar:
        progress_bar = tqdm(total=space_realizations *
                            len(time_steps), desc="Progress")

    # preestimate nearest neighbor matrix
    def create_w(n_timeslice, n_space):
        T = n_timeslice * n_space
        w_temporal = np.zeros((T, T), dtype=float)
        rows = np.arange(T - n_space)
        cols = rows + n_space
        w_temporal[rows, cols] = 1.0
        return w_temporal
    if time_cut_method == 'constant_idx':
        n_timeslice = int(np.ceil(n_events / n_m_time))
    elif time_cut_method == 'constant_time':
        n_timeslice = n_time
    elif time_cut_method == 'eval_times':
        n_timeslice = len(all_time_steps) + 1
    w_temporal_list = []
    w_temporal_list.append(create_w(n_timeslice, n_space))
    w_temporal_list.append(create_w(n_timeslice + 1, n_space))
    w_temporal_list.append(create_w(n_timeslice + 2, n_space))

    # 2. loop over realizations
    for ii in range(space_realizations):
        if voronoi_method == 'random':
            # create random voronoi nodes
            coords_vor = np.random.rand(dim, n_space)
            for jj in range(dim):
                coords_vor[jj, :] = limits[jj][0] + (
                    limits[jj][1] - limits[jj][0]) * coords_vor[jj, :]
        if voronoi_method == 'location':
            # choose random coordinates of EQs as voronoi nodes
            idx = np.random.choice(n_events, n_space)
            coords_vor = coords[:, idx]
        if x_is_a:
            vor = mirror_voronoi(coords_vor, limits)

        # find magnitudes and times corresponding to the voronoi nodes
        nearest_events = find_nearest_vor_node(coords_vor, coords)
        tile_magnitudes = find_points_in_tile(
            coords_vor, coords, mags, nearest=nearest_events)
        tile_times = find_points_in_tile(
            coords_vor, coords, times, nearest=nearest_events)
        if mc_method is not None:
            mc_local = estimate_mc_local(
                tile_magnitudes, mc_min, mc_method)

        # estimate b-values (for transformation only)
        if x_is_b:
            x_space, std_vec, n_m = values_from_partitioning(
                tile_magnitudes,
                tile_times,
                mc_local,
                delta_m,
                method=method,
                **kwargs)
            x_space[n_m < min_num] = np.nan

        # find the nearest voronoi node for each grid-point
        nearest = find_nearest_vor_node(coords_vor, eval_coords)

        # precompute volume
        if x_is_a:
            volume_space = volumes_vor(vor, n_space)
            for mm in range(dim):
                volume_space /= (limits[mm][1] - limits[mm][0])

        # go through time with constant window
        for _, time_step in enumerate(time_steps):
            # cut time in to pieces of constant length
            if time_cut_method == 'constant_idx':
                idx_left, time_magnitudes = cut_constant_idx(
                    mags, n=n_m_time, offset=time_step)
                time_times = np.array_split(times, idx_left)
            elif time_cut_method == 'constant_time':
                idx_left, time_times = cut_constant_value(
                    times, delta_value=length_time,
                    offset=time_step - min(times))
                time_magnitudes = np.array_split(mags, idx_left)
            elif time_cut_method == 'eval_times':
                idx_left = all_time_steps + 1
                time_magnitudes = np.array_split(mags, idx_left)
                time_times = np.array_split(times, idx_left)
            time_nearest = np.array_split((nearest_events), idx_left)
            idx_right = np.concatenate([idx_left - 1, [len(mags)-1]])
            idx_left = np.concatenate([[0], idx_left])

            # estimate the full nearest neighbour matrix, with only
            # connections in time
            w_temporal = w_temporal_list[int(len(idx_left) - n_timeslice)]

            # initialize arrays for estimating autocorrelation
            x_vec = np.zeros(len(idx_left) * n_space)
            n_m = np.zeros(len(idx_left) * n_space)

            # go through all time-sections
            for jj, mags_cut in enumerate(time_magnitudes):
                # find maggnitudes and times corresponding to the voronoi nodes
                tile_magnitudes_cut = find_points_in_tile(
                    coords_vor,  np.nan, mags_cut, nearest=time_nearest[jj])
                tile_times_cut = find_points_in_tile(
                    coords_vor, np.nan, time_times[jj],
                    nearest=time_nearest[jj])

                # estimate local mc if needed
                if mc_method is not None:
                    mc_local = estimate_mc_local(
                        tile_magnitudes_cut, mc_min, mc_method)

                # estimate a- or b-values
                if x_is_a:
                    # scale a by time-window and spatial volume
                    scale_tmp = (
                        np.max(time_times[jj]) - np.min(time_times[jj]))/(
                        np.max(times) - np.min(times))
                    scale_tmp *= volume_space
                    scale_tmp *= scaling_factor
                    x_loop, std_vec, n_m_loop = values_from_partitioning(
                        tile_magnitudes_cut,
                        tile_times_cut,
                        mc_local,
                        delta_m,
                        method=method,
                        list_scaling=scale_tmp,
                        **kwargs)
                    # std for a-value is not yet implemented
                    std_vec[n_m_loop > 0] = 0
                if x_is_b:
                    x_loop, std_vec, n_m_loop = values_from_partitioning(
                        tile_magnitudes_cut,
                        tile_times_cut,
                        mc_local,
                        delta_m,
                        method=method,
                        **kwargs)
                x_loop[n_m_loop < min_num] = np.nan
                x_vec[jj*n_space:(jj+1)*n_space] = x_loop
                n_m[jj*n_space:(jj+1)*n_space] = n_m_loop

                # average the b-values or a-values (welford algorithm)
                idx_loop = np.argwhere(idx_eval == idx_right[jj]).flatten()
                x_tmp = x_loop[nearest]
                x_std_tmp = std_vec[nearest]
                for idx in idx_loop:
                    aggregate[idx] = update_welford(aggregate[idx], x_tmp)
                    aggregate_std[idx] = update_welford(
                        aggregate_std[idx], x_std_tmp)

            # estimate Morans I (spatial autocorrelation)

            # result 1: Is there any temoral variation on top of the spatial
            # one? -> mac_time (overall significance), mac_map (local)
            x_time_mean = np.zeros(len(idx_left) * n_space)
            ac_map, n_map, n_p_map = np.zeros(
                n_space), np.zeros(n_space), np.zeros(n_space)
            mu_map, std_map = np.zeros(n_space), np.zeros(n_space)
            x_trans_local = x_vec.copy()
            for jj in range(n_space):
                idx = np.arange(jj, len(idx_left) * n_space, n_space)
                x_tmp = x_trans_local[idx]
                if transform:  # transform with local b-value
                    x_tmp = transform_n(
                        x_tmp, x_space[jj], n_m[idx], max(n_m[idx]))
                    x_trans_local[idx] = x_tmp
                # local temporal autocorrelation map at least 10 time steps
                if n_time >= 10:
                    ac_map[jj], n_map[jj], n_p_map[jj] = est_morans_i(x_tmp)
                    # mean for global temporal autocorrelation
                x_time_mean[idx] = np.nanmean(x_tmp)
            # transform such that mean=0s    and std=1 under null hypothesis
            with np.errstate(divide="ignore"):
                mu_map = -1/n_map
            std_map = np.sqrt(1/n_p_map)
            ac_map = (ac_map - mu_map) / std_map
            # add values to map
            agg_mac_map = update_welford(agg_mac_map, ac_map[nearest])

            # global temporal autocorrelation
            ac_temporal, n_temporal, n_p_temporal = (
                est_morans_i(x_trans_local, w_temporal, mean_v=x_time_mean))
            agg_ac_temporal = update_welford(agg_ac_temporal, ac_temporal)
            agg_n_p_temporal = update_welford(agg_n_p_temporal, n_p_temporal)
            agg_n_temporal = update_welford(agg_n_temporal, n_temporal)

            # result 2: Can we better forecast with spatial and temporal
            # variation? -> mac
            if transform:  # transform with overall b-value
                x_vec = transform_n(x_vec, b_all, n_m, max(n_m))
            ac, n, n_p = est_morans_i(x_vec, w_temporal)
            agg_ac = update_welford(agg_ac, ac)
            agg_n_p = update_welford(agg_n_p, n_p)
            agg_n = update_welford(agg_n, n)

            if time_bar:
                progress_bar.update(1)

    # 3. estimate the averages & estimate expected standard deviation of MAC
    x_average, x_std = [], []
    for mm in range(n_eval_time):
        x_avrg_loop, var_loop = finalize_welford(
            aggregate[mm], min_count=min_count)
        std_loop, _ = finalize_welford(aggregate_std[mm], min_count=min_count)
        x_std_loop = np.fmax(std_loop, np.sqrt(var_loop))
        x_average.append(x_avrg_loop)
        x_std.append(x_std_loop)

    mac = finalize_welford(agg_ac)[0]
    mean_n_p = finalize_welford(agg_n_p)[0]
    mean_n = finalize_welford(agg_n)[0]
    with np.errstate(divide="ignore"):
        mu_mac = -1/mean_n
    std_mac = np.sqrt(1/mean_n_p)

    mac_time = finalize_welford(agg_ac_temporal)[0]
    mean_n_p_time = finalize_welford(agg_n_p_temporal)[0]
    mean_n_time = finalize_welford(agg_n_temporal)[0]
    mu_mac_time = -1/mean_n_time
    std_mac_time = np.sqrt(1/mean_n_p_time)

    mac_map = finalize_welford(agg_mac_map, min_count=min_count)[0]

    return (
        np.asarray(x_average), np.asarray(x_std),
        mac, mu_mac, std_mac,
        mac_time, mu_mac_time, std_mac_time,
        mac_map
    )
