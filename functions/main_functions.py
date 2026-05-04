# main functions used for Amatrice analysis
from seismostats.plots import plot_fmd
from seismostats.analysis import BPositiveBValueEstimator, ClassicBValueEstimator
from functions.general_functions import likelihood_exp
import numpy as np


def loglik_test(cat, b_vals, b_all, mc_chosen):
    estimator = ClassicBValueEstimator()
    estimator.calculate(cat.magnitude, delta_m=cat.delta_m, mc=mc_chosen)
    cat = cat.copy()

    idx_abovemc = cat.index.values[estimator.idx]
    b_vals_abovemc = b_vals[estimator.idx]
    # 2. estimate likelihood and reference likelihood
    loglike = np.log10(
        likelihood_exp(
            magnitude=estimator.magnitudes,
            mc=mc_chosen, delta_m=cat.delta_m,
            b_value=b_vals_abovemc))
    loglike_ref = np.log10(
        likelihood_exp(
            magnitude=estimator.magnitudes,
            mc=mc_chosen, delta_m=cat.delta_m,
            b_value=b_all))
    cat['loglike'] = np.nan
    cat.loc[idx_abovemc, 'loglike'] = loglike - loglike_ref
    return cat


def positive_test(cat, b_vals, b_all, mc_chosen, dmc):
    # 1. only retain mags that are larger than previous
    estimator = BPositiveBValueEstimator()
    estimator.calculate(cat.magnitude, delta_m=cat.delta_m,
                        mc=mc_chosen, times=cat.time, dmc=dmc)
    cat = cat.copy()

    idx_positive = cat.index.values[estimator.idx]
    b_vals_positive = b_vals[estimator.idx]
    # 2. estimate likelihood and reference likelihood
    loglike = np.log10(
        likelihood_exp(
            magnitude=estimator.magnitudes,
            mc=dmc, delta_m=cat.delta_m,
            b_value=b_vals_positive))
    loglike_ref = np.log10(
        likelihood_exp(magnitude=estimator.magnitudes,
                       mc=dmc, delta_m=cat.delta_m,
                       b_value=b_all))
    cat['loglike'] = np.nan
    cat.loc[idx_positive, 'loglike'] = loglike - loglike_ref
    return cat
