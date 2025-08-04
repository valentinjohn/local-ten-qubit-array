# %% Path management

import os
from pathlib import Path

from config import DATA_DIR

try:
    script_dir = Path(__file__).resolve().parent
except:
    script_dir = Path(os.getcwd()) / 'single_qubit_rb'
fig_path = script_dir / "images"
data_path = DATA_DIR

# %% imports

import numpy as np
import math
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.optimize import approx_fprime

# %% definition

def RB_decay(N, amp, offset, p):
    return amp*p**N + offset

def RB_decay_fixed_p(N, amp, offset, p_fixed):
    return lambda N_, amp_, offset_: amp_ * p_fixed ** N_ + offset_

def RB_decay_fixed_AB(N, amp_fixed, offset_fixed, p):
    return lambda N_, p_: amp_fixed * p_ ** N_ + offset_fixed

def get_gate_fidelity_from_decay(p, p_err):
    error = 1 - p
    F_gate = 1 - (error / (2 * 2))
    F_err = p_err / (2 * 2)
    return F_gate, F_err

def get_value_with_error(value, error):
    if error == 0:
        return f"{value:.6g}(0)"
    # Determine the order of magnitude of the error
    error_order = int(math.floor(math.log10(abs(error))))
    if error_order > -3:
        # If error is big, round to 3 decimal places
        error_order = -3
    # Round the error to 1 significant figure
    rounded_error = round(error, -error_order)
    # Determine how many decimal places to round the value
    decimal_places = max(0, -error_order)
    # Round the value to the same decimal place as the error
    rounded_value = round(value, decimal_places)
    # Get the error digit (first significant digit only)
    error_digit = int(round(rounded_error / (10 ** error_order)))
    # Format the value and the error in parentheses
    format_string = f"{{:.{decimal_places}f}}({{}})"
    return format_string.format(rounded_value, error_digit)


# %% Using least square method

def get_fidelity_from_least_squares(N_Cliffords, Z_data, amp=1, offset=0.5, f_gate_start=0.99, f_gate_end=0.999):
    z_avrg = np.average(Z_data, axis=0)
    z_std = np.std(Z_data, axis=0)
    z_sigma = z_std / np.sqrt(Z_data.shape[0])  # standard error of the mean

    # Define grid of p values
    gate_fidelity_values = np.linspace(f_gate_start, f_gate_end, 201)
    p_values = 1 - 4 * (1 - gate_fidelity_values)  # convert gate fidelity to p
    chisq_values = []

    for p_fixed in p_values:
        # Define model with p fixed
        model = RB_decay_fixed_p(N_Cliffords, None, None, p_fixed)

        # Fit only amp and offset
        try:
            popt_tmp, _ = curve_fit(model, N_Cliffords, z_avrg,
                                    sigma=z_sigma,
                                    p0=[amp, offset],
                                    maxfev=1_000_000,
                                    absolute_sigma=True)

            # Compute residuals and RSS
            residuals = z_avrg - model(N_Cliffords, *popt_tmp)
            chisq = np.sum((residuals / z_sigma) ** 2)  # chi-squared
            chisq_values.append(chisq)
        except RuntimeError:
            chisq_values.append(np.nan)  # failed fit

    chisq_values = np.array(chisq_values)
    min_chisq = np.min(chisq_values)

    # Define Δχ² for 1σ confidence level in 1 parameter
    delta_chi2_1sigma = 1.0

    delta_chi2_99 = 6.63  # for 1 parameter at 99% confidence level

    # Find the range of p where RSS is within 99%
    f_gate_conf_int = gate_fidelity_values[chisq_values <= min_chisq + delta_chi2_99]
    f_gate_lower, f_gate_upper = f_gate_conf_int[0], f_gate_conf_int[-1]
    f_gate_best = gate_fidelity_values[np.argmin(chisq_values)]
    decay_constant_best = p_values[np.argmin(chisq_values)]
    f_gate_err = (f_gate_upper - f_gate_lower) / 2

    # Find the range of p where RSS is within 1σ
    f_gate_conf_int_1sigma = gate_fidelity_values[chisq_values <= min_chisq + delta_chi2_1sigma]
    f_gate_lower_1sigma, f_gate_upper_1sigma = f_gate_conf_int_1sigma[0], f_gate_conf_int_1sigma[-1]
    f_gate_best_1sigma = gate_fidelity_values[np.argmin(chisq_values)]
    f_gate_err_1sigma = (f_gate_upper_1sigma - f_gate_lower_1sigma) / 2

    model = RB_decay_fixed_p(N_Cliffords, None, None, decay_constant_best)
    popt_tmp, _ = curve_fit(model, N_Cliffords, z_avrg,
                                sigma=z_sigma,
                                p0=[amp, offset],
                                maxfev=1_000_000,
                                absolute_sigma=True)
    amp_fit, offset_fit = popt_tmp

    return {
        'f_gate': f_gate_best,
        'f_gate_err': f_gate_err,
        'f_gate_lower': f_gate_lower,
        'f_gate_upper': f_gate_upper,
        'f_gate_1sigma': f_gate_best_1sigma,
        'f_gate_err_1sigma': f_gate_err_1sigma,
        'chisq': min_chisq,
        'amp': amp_fit,
        'offset': offset_fit,
        'decay_constant': decay_constant_best
    }

# %% Using raw maximum likelihood

def neg_log_likelihood(params, N_Cliffords, Z_data):
    amp, offset, p, log_sigma = params
    sigma = np.exp(log_sigma)

    try:
        model = RB_decay(N_Cliffords[:, None], amp, offset, p)
        residuals = Z_data - model
        nll = 0.5 * np.sum((residuals / sigma)**2 + np.log(2 * np.pi * sigma**2))
    except Exception:
        return np.inf

    return nll

def numerical_hessian(f, x0, eps=1e-5):
    n = len(x0)
    H = np.zeros((n, n))
    fx = f(x0)
    for i in range(n):
        dx = np.zeros(n)
        dx[i] = eps
        grad1 = approx_fprime(x0 + dx, f, epsilon=eps)
        grad2 = approx_fprime(x0 - dx, f, epsilon=eps)
        H[:, i] = (grad1 - grad2) / (2 * eps)
    return H

def estimate_global_variance(Z_data):
    """
    Z_data: shape (n_cliffords, n_reps)
    Returns: estimated global variance
    """
    z_means = Z_data.T.mean(axis=1, keepdims=True)
    residuals = Z_data.T - z_means
    n, m = Z_data.T.shape
    pooled_var = np.sum(residuals**2) / (n * (m - 1))
    return pooled_var

def fit_rb_mle(N_Cliffords, Z_data, initial_guess=None):
    """
    N_Cliffords : array of shape (n,)
        Number of Clifford gates per data point
    Z_data : array of shape (n, m)
        Raw measurement data: n Clifford counts × m repetitions
    """
    N_Cliffords = np.asarray(N_Cliffords)
    Z_data = np.asarray(Z_data)

    sigma0 = estimate_global_variance(Z_data)

    if initial_guess is None:
        amp0 = np.max(Z_data) - np.min(Z_data)
        offset0 = np.min(Z_data)
        p0 = 0.99
        initial_guess = [amp0, offset0, p0]

    initial_guess += [np.log(sigma0)]  # log_sigma

    bounds = [
        (-1, 1),
        (0.2, 0.8),  # offset unbounded
        (0.9, 0.999),  # p ∈ (0, 1)
        (np.log(1e-6), np.log(0.2))  # log_sigma: sigma ≥ 1e-6
    ]

    result = minimize(neg_log_likelihood, initial_guess,
                      args=(N_Cliffords, Z_data),
                      bounds=bounds,
                      method='L-BFGS-B')

    if not result.success:
        raise RuntimeError("MLE fit did not converge:", result.message)

    amp, offset, p, log_sigma = result.x
    p_sigma = np.exp(log_sigma)

    # H_inv_dense = result.hess_inv.todense()
    # p_std = np.sqrt(H_inv_dense[2, 2])  # Index 2 is p
    hess = numerical_hessian(lambda params: neg_log_likelihood(params, N_Cliffords, Z_data), result.x)
    cov = np.linalg.inv(hess)
    p_std = np.sqrt(np.abs(cov[2, 2]))

    error = 1 - p
    F_gate = 1 - (error / 4)
    F_err = p_std / 4

    return {
        'amp': amp,
        'offset': offset,
        'p': p,
        'p_sigma': p_sigma,
        'f_gate': F_gate,
        'f_err': F_err,
        'neg_log_likelihood': result.fun,
        'hessian_inv': hess
    }

def get_max_likelihood_fidelity_with_error(N_Cliffords, z, p0):
    fit = fit_rb_mle(N_Cliffords, z.T, initial_guess=p0)

    return fit['f_gate'], fit['f_err']