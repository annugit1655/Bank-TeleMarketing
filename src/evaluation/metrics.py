"""
Calibration Metrics for Model Evaluation

Comprehensive set of metrics for assessing calibration quality.
"""

import numpy as np
from scipy import stats
from typing import Tuple, Optional
import warnings

def expected_calibration_error(y_true: np.asarray,
                               y_prob: np.asarray,
                               n_bins: int=10,
                               strategy: str = 'uniform') -> float:

    """
    Calculate Expected Calibration Error (ECE).

    ECE measures the average difference between predicted probabilities 
    and observed frequencies across bins.

    Parameters:
    ------------
    y_true: np.asarray
            True Label (0 or 1)
    y_prob: np.asarray
            Predicted Probabilities
    n_bins: int, default = 10
            Number of bins for grouping predictions
    strategy : str, default = 'uniform'
                Binning Strategy ('uniform' or 'quantile')
                'uniform' : equal-width interval but not equal no. of observations
                'quantile' : equal no. of observations bt not equal-width interval
    
    Returns:
            float
                Expected Calibration Error
    """

    ece = 0

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    total_samples = len(y_true)

    if strategy == 'uniform':
        bin_boundaries = np.linspace(0, 1, n_bins + 1) 
    elif strategy == 'quantile':
        bin_boundaries = np.quantile(y_prob, np.linspace(0, 1, n_bins + 1))
    else:
        raise ValueError("strategy must be 'uniform' or 'quantile'")
    
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        if bin_upper == 1.0:
            in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
        else:
            in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)

    # prop_in_bin = in_bin.sum()/len(in_bin) = in_bin.mean()
    prop_in_bin = in_bin.mean()

    if prop_in_bin > 0:
        accuracy_in_bin = y_true[in_bin].mean()
        avg_confidence_in_bin = y_prob[in_bin].mean()
        ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin

    return ece


def maximum_calibration_error(y_true: np.asarray,
                               y_prob: np.asarray,
                               n_bins: int=10,
                               strategy: str = 'uniform') -> float:

    """
    Calculate Maximum Calibration Error (MCE).

    MCE measures the worst-case calibration error across all bins.

    Parameters:
    ------------
    y_true: np.asarray
            True Label (0 or 1)
    y_prob: np.asarray
            Predicted Probabilities
    n_bins: int, default = 10
            Number of bins for grouping predictions
    strategy : str, default = 'uniform'
                Binning Strategy ('uniform' or 'quantile')
                'uniform' : equal-width interval but not equal no. of observations
                'quantile' : equal no. of observations bt not equal-width interval
    
    Returns:
            float
                Maximum calibration error
    """

    mce = 0

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    

    if strategy == 'uniform':
        bin_boundaries = np.linspace(0, 1, n_bins + 1) 
    elif strategy == 'quantile':
        bin_boundaries = np.quantile(y_prob, np.linspace(0, 1, n_bins + 1))
    else:
        raise ValueError("strategy must be 'uniform' or 'quantile'")
    
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        if bin_upper == 1.0:
            in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
        else:
            in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)

    if in_bin.sum() > 0:
        accuracy_in_bin = y_true[in_bin].mean()
        avg_confidence_in_bin = y_prob[in_bin].mean()
        mce = max(mce, np.abs(accuracy_in_bin, avg_confidence_in_bin))

    return mce


def reliability_score(y_true: np.ndarray,
                     y_prob: np.ndarray,
                     n_bins: int = 10) -> float:
    """
    Calculate reliability score (1 - ECE).

    Higher values indicate better calibration.

    Parameters:
    -----------
    y_true : np.ndarray
        True binary labels (0 or 1)
    y_prob : np.ndarray
        Predicted probabilities
    n_bins : int, default=10
        Number of bins for ECE calculation

    Returns:
    --------
    float
        Reliability score
    """
    ece = expected_calibration_error(y_true, y_prob, n_bins)
    return 1 - ece


def hosmer_lemeshow_test(y_true: np.ndarray,
                        y_prob: np.ndarray,
                        n_bins: int = 10) -> Tuple[float, float]:
    """
    Perform Hosmer-Lemeshow goodness-of-fit test for calibration.

    Tests whether observed and expected frequencies agree.

    Parameters:
    -----------
    y_true : np.ndarray
        True binary labels (0 or 1)
    y_prob : np.ndarray
        Predicted probabilities
    n_bins : int, default=10
        Number of bins for grouping predictions

    Returns:
    --------
    tuple
        (chi2_statistic, p_value)
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    # Create bins based on predicted probabilities
    bin_boundaries = np.quantile(y_prob, np.linspace(0, 1, n_bins + 1))
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    observed_events = []
    expected_events = []
    observed_non_events = []
    expected_non_events = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        if bin_upper == 1.0:
            in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
        else:
            in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)

        if np.sum(in_bin) > 0:
            obs_events = np.sum(y_true[in_bin])
            exp_events = np.sum(y_prob[in_bin])
            obs_non_events = np.sum(in_bin) - obs_events
            exp_non_events = np.sum(in_bin) - exp_events

            observed_events.append(obs_events)
            expected_events.append(exp_events)
            observed_non_events.append(obs_non_events)
            expected_non_events.append(exp_non_events)

    if len(observed_events) < 2:
        warnings.warn("Insufficient bins for Hosmer-Lemeshow test")
        return np.nan, np.nan

    observed = np.array([observed_events, observed_non_events]).T
    expected = np.array([expected_events, expected_non_events]).T

    # Calculate chi-square statistic
    with np.errstate(divide='ignore', invalid='ignore'):
        chi2_stat = np.sum((observed - expected) ** 2 / (expected + 1e-10))

    df = len(observed) - 2
    p_value = 1 - stats.chi2.cdf(chi2_stat, df) if df > 0 else np.nan

    return chi2_stat, p_value
