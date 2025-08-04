# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 17:17:50 2023

@author: Francesco
"""
import inspect

import logging
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as constant

from scipy import ndimage
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter, find_peaks


# %% Fitting routines


def PolLine(kT, x, x0, t, y0, slope_l, slope_r, delta):
    x_data_center = x - x0
    Om = np.sqrt(x_data_center**2 + 4 * t**2)
    Q = 1 / 2 * (1 + x_data_center / Om * np.tanh(Om / (2 * kT)))
    slopes = slope_l + (slope_r - slope_l) * Q
    y_data = y0 + x_data_center * slopes + Q * delta
    return y_data


def PolLine_t(t,  x, x0, kT, y0, slope_l, slope_r, delta):
    x_data_center = x - x0
    Om = np.sqrt(x_data_center**2 + 4 * t**2)
    Q = 1 / 2 * (1 + x_data_center / Om * np.tanh(Om / (2 * kT)))
    slopes = slope_l + (slope_r - slope_l) * Q
    y_data = y0 + x_data_center * slopes + Q * delta
    return y_data


def diffN(x, y, N=1):
    dy = np.array(y[N:]) - np.array(y[:-N])
    dx = (np.array(x[N:]) + np.array(x[:-N]))/2
    return (dx, dy)


def Lorentz(x, A, FWHM, x0, y0):
    w = FWHM / 2
    return A / (((x - x0) / w)**2 + 1) + y0


def Gauss(x, A, FWHM, x0, y0):
    w = FWHM / (2 * np.sqrt(2 * np.log(2)))
    return A * np.exp(-(x - x0)**2 / (2 * w**2)) + y0


def NGauss(x, *args):
    N = int((len(args) - 1) / 3)
    # print(f'number of Gaussians: {N}')
    y0 = args[-1]
    for k in range(N):
        arg_set = args[3*k:3*k + 3]
        A = arg_set[0]
        FWHM = arg_set[1]
        x0 = arg_set[2]
        w = FWHM / (2 * np.sqrt(2 * np.log(2)))
        peak = A * np.exp(-(x - x0)**2 / (2 * w**2)) + y0

        if k == 0:
            result = peak
        else:
            result += peak
    # print(f'A ={A:.2}, FWHM = {FWHM:.2}, x0 = {x0:.2}, y0 = {y0:.2}')
    return result


def dGauss(x, A, FWHM, x0, y0):
    w = FWHM / (2 * np.sqrt(2 * np.log(2)))
    return (x0 - x) / w**2 * A * np.exp(-(x - x0)**2 / (2 * w**2)) + y0


def Gauss2(x, A, A2, FWHM, FWHM2, x0, x0_2):
    w = FWHM / (2 * np.sqrt(2 * np.log(2)))
    w2 = FWHM2 / (2 * np.sqrt(2 * np.log(2)))
    return A * np.exp(-(x - x0)**2 / (2 * w**2)) + A2 * np.exp(-(x - x0_2)**2 / (2 * w2**2))


def Rabi(t, A, f, alpha, y0, phi):
    return A*np.cos(2*np.pi*f*t+phi)/(2*np.pi*f*t)**alpha+y0

# def Ramsey(t, A, f, T2, y0, phi):
#    return A*np.cos(2*np.pi*t*f+phi)*np.exp(-t/T2)+y0


def Ramsey(t, A, f, phi, tau, C):
    return A * np.sin(2 * np.pi * f * t + phi) * np.exp(-t**2 / tau**2) + C


def Ramsey_shift(t, A, f, phi, tau, C, t0):
    return A * np.sin(2 * np.pi * f * (t) + phi) * np.exp(-(t-t0)**2 / tau**2) + C


def ExpDec(x, A, m, y0):
    return A * np.exp(-x / m) + y0


def ExpDec_offset(x, A, m, x0):
    return A * np.exp(-(x - x0) / m)


def OneOverf(f, alpha, f0):
    return (f0/f)**alpha


def Larmor_resonance(f, A, offset, t, f_Rabi, f_Larmor):
    Delta = f - f_Larmor
    P = offset + A*(f_Rabi**2 / (f_Rabi**2 + Delta**2)) * \
        np.sin(0.5 * t * np.sqrt((f_Rabi**2 + Delta**2)))**2
    return P


def PSD_CPMG(t_wait, A_CMPG):
    '''

    Parameters
    ----------
    t_wait : in second
    A_CMPG : normalized amplitude CPMG for a given t_wait

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    return -np.log(A_CMPG)/(2*t_wait*np.pi**2)


def freq_PSD_CPMG(t_wait, N_CPMG):
    return N_CPMG/(2*t_wait)


def sinusoidal_wave(x, amplitude, frequency, phase, offset):
    """
    Sinusoidal wave function for curve fitting.

    Parameters:
    x (array-like): Independent variable (in radians).
    amplitude (float): Amplitude of the sinusoid.
    frequency (float): Frequency of the sinusoid (in radians per second).
    phase (float): Phase shift of the sinusoid.
    offset (float): Vertical offset of the sinusoid.

    Returns:
    array-like: Evaluated sinusoidal wave.
    """
    return amplitude * np.cos(frequency * x + phase) + offset


def fit_one_oscillation(x_data, y_data):
    """
    Fits a dataset to a sinusoidal model considering one oscillation.

    Parameters:
    x_data (array-like): The independent variable data (in radians).
    y_data (array-like): The dependent variable data.

    Returns:
    tuple: Best-fit parameters and their standard deviations.
    """

    # Guess initial parameters (assuming frequency close to 1 Hz)
    amplitude_guess = (np.max(y_data) - np.min(y_data)) / 2
    frequency_guess = 1
    phase_guess = 0
    offset_guess = np.mean(y_data)

    initial_guess = [amplitude_guess,
                     frequency_guess, phase_guess, offset_guess]

    try:
        best_fit_params, cov_matrix = curve_fit(
            sinusoidal_wave, x_data, y_data, p0=initial_guess, bounds=([amplitude_guess - 0.05, frequency_guess * 0.5, phase_guess - 0.1, offset_guess - 0.1],  # f_Larmor_guess*0.8
                                                                       [amplitude_guess + 0.05, frequency_guess * 2, phase_guess + 0.1, offset_guess + 0.1])  # f_Larmor_guess*1.2

        )

        # Calculate standard deviations of the parameters
        perr = np.sqrt(np.diag(cov_matrix))

        # Print the best-fit parameters and their errors
        param_names = ["Amplitude",
                       "Frequency (rad/s)", "Phase (rad)", "Offset"]
        for i, (param, error) in enumerate(zip(best_fit_params, perr)):
            print(f"{param_names[i]}: {param:.3f} \u00B1 {error:.3f}")

    except Exception as e:
        print(f"Fit did not work: {e}")
        return None, None

    return best_fit_params, perr


def fit_Larmor_data(f, y):
    """
    Fit and plot Larmor resonance data.

    Parameters:
    f (array): Frequency data.
    y (array): Resonance data.

    Returns:
    tuple: Best-fit parameters and their standard deviations.
    """

    # Guess parameters
    if y[0] <= 0.5:
        A_guess = np.max(y) - np.min(y)
        offset_guess = np.min(y)
        f_Larmor_guess = f[np.argmax(y)]
    else:
        A_guess = np.min(y) - np.max(y)
        offset_guess = np.max(y)
        f_Larmor_guess = f[np.argmin(y)]
    t_guess = 2e3*1e-9
    f_Rabi_guess = 0.1e6

    # largest_y_indices = np.argsort(y)[-4:]
    # average_x = np.mean(f[largest_y_indices])
    # f_Larmor_guess = average_x

    initial_guess = [A_guess, offset_guess,
                     t_guess, f_Rabi_guess, f_Larmor_guess]

    try:
        # Perform curve fitting
        best_fit_params, cov_matrix = curve_fit(
            Larmor_resonance, f, y,
            p0=initial_guess,
            maxfev=1400*10,
            bounds=([A_guess - 0.10, offset_guess - 0.2, t_guess*0, 0e6, np.sign(f_Larmor_guess)*np.abs(f_Larmor_guess) - 0.2*np.abs(f_Larmor_guess)],  # f_Larmor_guess*0.8
                    [A_guess + 0.10, offset_guess + 0.2, t_guess*10, 50e6,  np.sign(f_Larmor_guess)*np.abs(f_Larmor_guess) + 0.2*np.abs(f_Larmor_guess)])  # f_Larmor_guess*1.2
        )

        # Calculate standard deviations (square root of the diagonal of the covariance matrix)
        perr = np.sqrt(np.diag(cov_matrix))

        # Print the best-fit parameters and their errors
        param_names = ["A", "offset",
                       "time (us)", "f_Rabi (MHz)", "f_Larmor (MHz)"]
        for i, (param, error) in enumerate(zip(best_fit_params, perr)):
            if i >= 3:  # Converting f_Rabi and f_Larmor from Hz to MHz and GHz respectively
                print(
                    f"{param_names[i]}: {1e-6*param:.3f} \u00B1 {1e-6*error:.3f}")
            elif i == 2:
                print(
                    f"{param_names[i]}: {1e6*param:.3f} \u00B1 {1e6*error:.3f}")
            else:
                print(f"{param_names[i]}: {param:.3f} \u00B1 {error:.3f}")

    except Exception as e:
        print('Fit did not work:', e)
        return None, None

    return best_fit_params, perr


def fit_Gaussian_data(f, y):
    """
    Fit and plot Larmor resonance data using a Gaussian function.

    Parameters:
    f (array): Frequency data.
    y (array): Resonance data.

    Returns:
    tuple: Best-fit parameters and their standard deviations.
    """

    # Guess parameters for Gaussian fit
    FWHM_guess = np.ptp(f) / 10  # Rough estimate of FWHM

    if np.mean(y) <= 0.5:
        A_guess = np.max(y) - np.min(y)
        x0_guess = f[np.argmax(y)]
        y0_guess = np.min(y)
    if np.mean(y) > 0.5:
        A_guess = np.min(y) - np.max(y)
        x0_guess = f[np.argmin(y)]
        y0_guess = np.max(y)

    initial_guess = [A_guess, FWHM_guess, x0_guess, y0_guess]
    print(initial_guess)

    try:
        # Perform curve fitting with the Gaussian function
        best_fit_params, cov_matrix = curve_fit(
            Gauss, f, y, p0=initial_guess, maxfev=14000
        )

        # Calculate standard deviations (square root of the diagonal of the covariance matrix)
        perr = np.sqrt(np.diag(cov_matrix))

        # Print the best-fit parameters and their errors
        param_names = ["A", "FWHM", "x0", "y0"]
        for i, (param, error) in enumerate(zip(best_fit_params, perr)):
            print(f"{param_names[i]}: {param:.3f} \u00B1 {error:.3f}")

        # Optional: Plotting the fit
        # plt.figure()
        # plt.plot(f, y, 'b-', label='Data')
        # plt.plot(f, Gauss(f, *best_fit_params), 'r--', label='Fit')
        # plt.xlabel('Frequency')
        # plt.ylabel('Intensity')
        # plt.title('Gaussian Fit to Larmor Data')
        # plt.legend()
        # plt.show()

    except Exception as e:
        print('Fit did not work:', e)
        return None, None

    return best_fit_params, perr


def Hahn_decay(t, A, offset, gamma_ge, T2H, alpha, a0, B):
    # f_ge = gamma_ge*B
    return offset + A*np.exp(-(t/T2H)**alpha)/np.abs(1-a0*np.cos(2*np.pi*gamma_ge*B*t/2))**2


def fit_Hahn_decay(t, y, B):
    """
    Fit and plot Hanh decay.

    Parameters:
    t (array): Time data.
    y (array): Probability data.

    Returns:
    tuple: Best-fit parameters and their standard deviations.
    """

    # Guess parameters
    if y[0] >= 0.5:
        A_guess = np.max(y) - np.min(y)
        offset_guess = np.min(y)
    else:
        A_guess = np.min(y) - np.max(y)
        offset_guess = np.max(y)
    gamma_ge_guess = 1.48e6  # in Hz
    B_guess = B
    T2H_guess = 20e-6  # s
    alpha = 1
    a0 = 0.5

    initial_guess = [A_guess, offset_guess,
                     gamma_ge_guess, T2H_guess, alpha, a0, B_guess]

    try:
        # Perform curve fittinggamma_ge_guess*100
        best_fit_params, cov_matrix = curve_fit(
            Hahn_decay, t, y,
            p0=initial_guess,
            bounds=([A_guess - 0.05, offset_guess - 0.05, 0.1e6, 1e-6, 0, 0.01,  B_guess*0.99],  # f_Larmor_guess*0.8
                    [A_guess + 0.05, offset_guess + 0.05, 2e6, 100e-6, 2.5, 1, B_guess*1.01])  # f_Larmor_guess*1.2
        )

        # Calculate standard deviations (square root of the diagonal of the covariance matrix)
        perr = np.sqrt(np.diag(cov_matrix))

        # Print the best-fit parameters and their errors
        param_names = ["A", "offset",
                       "gamma_ge73 (MHz)", "T2H (us)", "alpha", "a0", "B"]
        # Conversion factors for each parameter
        units_conversion = [1, 1, 1e-6, 1e6, 1, 1, 1]
        for i, (param, error) in enumerate(zip(best_fit_params, perr)):
            converted_param = param * units_conversion[i]
            converted_error = error * units_conversion[i]
            print(
                f"{param_names[i]}: {converted_param:.3f} \u00B1 {converted_error:.3f}")

    except Exception as e:
        print('Fit did not work:', e)
        return None, None

    return best_fit_params, perr


def fit_data(xdata, ydata, p0=None, func=None, plot=True,
             return_cov=False, verbose=0, fix_params={}, **kwargs):
    """
    Fit data using the specified function and initial parameters.

    Parameters:
    - xdata: numpy array
        X values.
    - ydata: numpy array
        Y values.
    - p0: list, optional
        Initial parameters for the fit.
    - func: function, optional
        The function used for fitting.
    - plot: bool, optional
        Whether to plot the fit or not.
    - return_cov: bool, optional
        Whether to return the covariance or not.
    - verbose: int, optional
        Verbosity level.
    - fix_params: dict, optional
        Parameters to fix during the fit.
    - **kwargs:
        Additional arguments to pass to the curve fitting function.

    Returns:
    - p1: numpy array
        Fitted parameters.
    - covar: numpy array, optional
        Covariance of the fit.
    """
    def initialize_p0dict(func):
        if func in [Gauss, dGauss, Lorentz]:
            return {
                'A': np.max(ydata) - np.mean(ydata),
                'FWHM': (x_range[-1] - x_range[0]) * 0.2,
                'x0': xdata[np.argmax(ydata)],
                'y0': np.mean(ydata)
            }
        elif func is Rabi:
            return {
                'A': np.max(ydata) - np.mean(ydata),
                'f': 1 / (x_range[-1] - x_range[0]) * 2,
                'alpha': 0.1,
                'y0': np.mean(ydata),
                'phi': 0
            }
        elif func is ExpDec:
            start = np.mean(ydata[:5])
            baseline = np.mean(ydata[int(3 * len(ydata) / 4):])
            return {
                'A': start - baseline,
                'm': (xdata[1] - xdata[-1]) / 2,
                'y0': baseline
            }

    x_range = np.linspace(np.min(xdata), np.max(xdata), num=500)

    if p0 is None and func:
        p0dict = initialize_p0dict(func)
        if verbose:
            logging.info(f'p0: {p0dict}')

    if fix_params:
        func = partial(func, **fix_params)

    try:
        if 'p0dict' in locals():
            p0 = [p0dict[arg] for arg in inspect.getfullargspec(func).args[1:]]

        if 'bounds' in kwargs.keys() and p0:
            for i, p in enumerate(p0):
                lower, upper = kwargs['bounds'][0][i], kwargs['bounds'][1][i]
                p0[i] = min([max([lower, p]), upper])

        p1, covar = curve_fit(func, xdata, ydata, p0=p0,
                              maxfev=1400*10, **kwargs)

        if plot:
            fig = plt.figure()
            plt.plot(xdata, ydata, 'b', label='Data')
            plt.plot(x_range, func(x_range, *p1), 'r', label='Fit')
            plt.legend()
            plt.show()

        if return_cov:
            return p1, np.sqrt(np.diag(covar))
        else:
            return p1

    except RuntimeError as Err:
        logging.warning(Err)
        return None


def estimate_double_gaussian_parameters(x_data, y_data, split=0.5):
    """
    Estimate the parameters for a double Gaussian model.

    Parameters:
    - x_data: numpy array
        Array of x-values.
    - y_data: numpy array
        Array of y-values corresponding to the x-values.
    - split: float (default: 0.5)
        The proportion at which to split the data for the two Gaussians.

    Returns:
    - initial_params: numpy array
        An array containing estimated parameters in the following order:
        [amplitude_left, amplitude_right, sigma_left, sigma_right, mean_left, mean_right].
    """

    # Calculate the split index once
    split_idx = int(len(y_data) * split)

    # Split the data
    data_left, data_right = y_data[:split_idx], y_data[split_idx:]
    x_data_left, x_data_right = x_data[:split_idx], x_data[split_idx:]

    # Calculate signal percentiles
    maxsignal = np.percentile(x_data, 98)
    minsignal = np.percentile(x_data, 2)
    sigma = (maxsignal - minsignal) / 20

    # Estimate parameters
    amplitude_left, amplitude_right = np.max(data_left), np.max(data_right)
    mean_left = np.sum(x_data_left * data_left) / np.sum(data_left)
    mean_right = np.sum(x_data_right * data_right) / np.sum(data_right)

    initial_params = np.array(
        [amplitude_left, amplitude_right, sigma, sigma, mean_left, mean_right])
    return initial_params


def thresholded_data(image, xaxis, split=0.5, max_diff=5, plot=True, sensor='sensor1'):
    """
    Apply thresholding on the data.

    Parameters:
    - image: numpy array
        2D array representing the image data.
    - xaxis: numpy array
        Array of x-values.
    - split: float, optional
        Split value for data. Default is 0.5.
    - max_diff: int, optional
        Maximum allowed difference for thresholding. Default is 5.
    - plot: bool, optional
        Whether to plot the thresholded data or not. Default is True.
    - sensor: str, optional
        Sensor name. Default is 'sensor1'.

    Returns:
    - nsub: numpy array
        Calculated fraction blocked.
    - thresholds: list
        List of threshold values.
    """
    def get_threshold_range(index, max_index, window=2):
        return max(0, index - window), min(max_index, index + window)

    xdata = np.arange(len(image[0]))
    thresholds = []
    threshold_prev = split

    for i in range(len(xaxis)):
        i_min, i_max = get_threshold_range(i, len(xaxis))

        ydata_guess = np.mean(image[i_min:i_max], axis=0)
        ydata = image[i]

        guess = estimate_double_gaussian_parameters(
            xdata, ydata_guess, split=split)
        p1_guess = fit_data(xdata, ydata_guess,
                            func=Gauss2, p0=guess, plot=False)

        try:
            p1 = fit_data(xdata, ydata, func=Gauss2, p0=p1_guess, plot=False)
            threshold = (p1[4] + p1[5]) / 2
            threshold_guess = (p1_guess[4] + p1_guess[5]) / 2

            if abs(threshold - threshold_guess) < max_diff:
                thresholds.append(threshold)
                threshold_prev = threshold
            else:
                thresholds.append(threshold_guess)
                threshold_prev = threshold_guess

        except Exception as e:
            print(f"Thresholding failed due to: {e}")
            thresholds.append(threshold_prev)

    if plot:
        plt.figure()
        plt.pcolor(image)
        plt.plot(thresholds, np.arange(len(thresholds)) + 0.5, 'r')
        plt.show()

    sup = [sum(image[i][:int(t)]) for i, t in enumerate(thresholds)]
    nsub = np.array(sup)

    return nsub, thresholds

# Function to fit Ramsey like data and return best-fit parameters


# def fit_Ramsey_data(x, data, freq_guess, short_measurement=False,
#                     tau_guess=1e-3, amplitude_guess=None,
#                     tau_min=None, tau_max=None, print_popt=True):
#     """
#     Fit Ramsey data and return best-fit parameters and their errors.

#     Parameters:
#     x (array): Independent variable (s)
#     data (array): Dependent variable (data to be fitted).
#     freq_guess (float): Initial guess for the frequency (MHz)
#     short_measurement (bool): Flag for short measurement.
#     tau_guess (float): Initial guess for decay time (s)

#     Returns:
#     tuple: Best-fit parameters and their standard deviations, or None if fitting fails.
#     """

#     # Guess amplitude based on measurement type
#     if amplitude_guess is None:
#         amplitude_guess = 2 * np.std(data) * 1.5
#         if short_measurement:
#             amplitude_guess = 0.5 * np.abs(np.max(data) - np.min(data))

#     offset_guess = np.average(data)
#     initial_guess = [amplitude_guess, freq_guess, 0,
#                      tau_guess, offset_guess]

#     try:
#         if tau_min is None:
#             tau_min = 0
#         if tau_max is None:
#             tau_max = 2*tau_guess
#         best_fit_params, cov_matrix = curve_fit(
#             Ramsey, x, data, p0=initial_guess,
#             bounds=([-1, freq_guess - 1e6, -1.05*np.pi, tau_min, 0.2],
#                     [+1, freq_guess + 1e6, 1.05*np.pi, tau_max, 0.8]), maxfev=14000 * 100)

#         # bounds=([amplitude_guess - 0.20, freq_guess - 1e6, -np.pi, 0.8*tau_guess, 0],
#         #        [amplitude_guess + 0.20, freq_guess + 1e6, np.pi, 1.2*tau_guess, 1.0])

#         # Calculate standard deviations (square root of the diagonal of the covariance matrix)
#         perr = np.sqrt(np.diag(cov_matrix))

#         # Calculate r squared
#         y_pred = Ramsey(x, *best_fit_params)

#         # Print the best-fit parameters and their errors
#         if print_popt:
#             print("Best-fit parameters and their errors:")
#             param_names = ["Amplitude (A)", "Frequency", "Phase",
#                            "Decay Time (us)", "Constant Offset (C)"]
#             for i, (param, error) in enumerate(zip(best_fit_params, perr)):
#                 if i == 3:
#                     print(
#                         f"{param_names[i]}: {1e6*param:.3f} \u00B1 {1e6*error:.3f}")
#                 else:
#                     print(f"{param_names[i]}: {param:.3f} \u00B1 {error:.3f}")

#             print(best_fit_params)
#         return best_fit_params, perr

#     except Exception as e:
#         if print_popt:
#             print(f"Error fitting data: {e}")
#         return None, None

def fit_Ramsey_data(x, data, freq_guess, short_measurement=False,
                    tau_guess=1e-3, amplitude_guess=None):
    """
    Fit Ramsey data and return best-fit parameters and their errors.

    Parameters:
    x (array): Independent variable (s)
    data (array): Dependent variable (data to be fitted).
    freq_guess (float): Initial guess for the frequency (MHz)
    short_measurement (bool): Flag for short measurement.
    tau_guess (float): Initial guess for decay time (s)

    Returns:
    tuple: Best-fit parameters and their standard deviations, or None if fitting fails.
    """

    # Guess amplitude based on measurement type
    if amplitude_guess is None:
        amplitude_guess = 2 * np.std(data) * 1.5
        if short_measurement:
            amplitude_guess = 0.5 * np.abs(np.max(data) - np.min(data))

    if data[0] > 0.5:
        phase_guess = np.pi/2
    else:
        phase_guess = -np.pi/2

    offset_guess = np.average(data)
    initial_guess = [amplitude_guess, freq_guess, phase_guess, tau_guess, offset_guess] #phase guess = 0

    try:
        best_fit_params, cov_matrix = curve_fit(
            Ramsey, x, data, p0=initial_guess,
            bounds=([0, freq_guess - 1e6, phase_guess-0.5*np.pi, 0.0*tau_guess, 0],
                    [1, freq_guess + 1e6, phase_guess+0.5*np.pi, 2*tau_guess, 100.0])
            , maxfev= 14000 * 100)

            #bounds=([amplitude_guess - 0.20, freq_guess - 1e6, -np.pi, 0.8*tau_guess, 0],
            #        [amplitude_guess + 0.20, freq_guess + 1e6, np.pi, 1.2*tau_guess, 1.0])

        # Calculate standard deviations (square root of the diagonal of the covariance matrix)
        perr = np.sqrt(np.diag(cov_matrix))

        # Print the best-fit parameters and their errors
        print("Best-fit parameters and their errors:")
        param_names = ["Amplitude (A)", "Frequency", "Phase", "Decay Time (us)", "Constant Offset (C)"]
        for i, (param, error) in enumerate(zip(best_fit_params, perr)):
            if i == 3:
                print(f"{param_names[i]}: {1e6*param:.3f} \u00B1 {1e6*error:.3f}")
            else:
                print(f"{param_names[i]}: {param:.3f} \u00B1 {error:.3f}")

        return best_fit_params, perr

    except Exception as e:
        print(f"Error fitting data: {e}")
        return None, None

def find_major_peak_frequency(x, data, height_threshold=0.1):
    zfft = np.fft.fft(data)
    u = x.size
    timestep = x[1] - x[0]
    freq = np.fft.fftfreq(u, d=timestep)

    freq = freq[1:u // 2]
    fft_norm = np.abs(zfft)[1:u // 2] / np.abs(zfft).max()

    # Find peaks in the FFT
    peaks, _ = find_peaks(fft_norm, height=height_threshold)

    # Get the peak frequency and amplitude
    peak_freq = freq[peaks]
    peak_amplitude = fft_norm[peaks]

    return freq, fft_norm, peak_freq, peak_amplitude

# basic functions


def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)


def derivative(x, y, z, N, method='sum'):
    '''
    Parameters
    ----------
    x : x axis

    y : y axis

    z : z data

    N: integer, n point along which the derivative is taken

    method: 'sum', 'square', 'gaussian', 'partial x', 'partial y'

    Returns
    -------
    dx : x axis of derivative

    dy : y axis of derivative

    dz : gradient

    '''
    dzdx = []
    for line in z:
        dx, dline = diffN(x, line, N)
        dzdx.append(dline)
    dzdy = []
    for line in z.T:
        dy, dline = diffN(y, line, N)
        dzdy.append(dline)
    dzdx = np.array(dzdx)[int(N/2):-int(N/2), :]
    dzdy = np.array(dzdy).T
    dzdy = dzdy[:, int(N/2):-int(N/2)]
    dz = dzdx + dzdy
    if method == 'sum':
        dz = dzdx + dzdy
    if method == 'partial x':
        dz = dzdx
    if method == 'partial y':
        dz = dzdy
    if method == 'square':
        dz = np.sqrt(dzdx**2 + dzdy**2)
    if method == 'gaussian':
        dz = ndimage.gaussian_gradient_magnitude(
            z, sigma=1, mode='nearest')  # sigma = 1
        dx = x
        dy = y
    # offset = np.median(dz)
    # dz -= offset
    dz -= np.min(dz)
    dz /= np.max(dz)
    return dx, dy, dz


def smooth(z, N, direction='horizontal'):
    '''
    Parameters
    ----------
    x : x axis

    y : y axis

    z : z data

    N: integer, n point along which average

    direction: horizontal or vertical

    Returns
    -------
    x : x axis

    y : y axis

    z : data - data_average

    '''
    if direction == 'horizontal':
        smooth_data = []
        for line in z:
            smooth_line = savgol_filter(line, N, 3
                                        )
            smooth_data.append(smooth_line)

        smooth_data = np.array(smooth_data)
        z = z - smooth_data

    # smooth along vertical
    if direction == 'vertical':
        smooth_data = []
        for line in z.T:
            smooth_line = savgol_filter(line, N, 3)
            smooth_data.append(smooth_line)

        smooth_data = np.array(smooth_data)
        z = z.T - smooth_data

    return z

# define a function to extract the data from a given id, channel, with margins, and derivative


def data_extraction(id, channel, i, j, N, method='sum', smooth=False, smooth_points=10):  # 5
    '''
    Parameters
    ----------
    id : id of the dataset (int)
    channel : channel of digitizer (integer from 1 to 4)
    i : margin of x (int)
    j : margin of y (int)
    N : number of points over which calculate the derivative (= 4)
    method: 'sum', 'square', 'gaussian'

    Returns
    -------
    x, y, z, x_label, y_label, z_label, x_unit, y_unit, z_unit, dx, dy, dz

    '''
    # dat = load_by_id(id)
    dat = load_by_uuid(id)
    try:
        dat = getattr(dat, f'm1_{channel}')
    except:
        dat = getattr(dat, f'm1')
    try:
        x, y, z = dat.y(), dat.x(), dat.z()
        x_label, y_label, z_label = dat.y.label, dat.x.label, dat.z.label
        x_unit, y_unit, z_unit = dat.y.unit, dat.x.unit, dat.z.unit
        x = x[i:]
        y = y[j:]
        z = z[j:, i:]
        dx, dy, dz = derivative(x, y, z, N, method)
        if smooth:
            z = z - ndimage.gaussian_filter(z, smooth_points, mode='nearest')
        results = x, y, z, x_label, y_label, z_label, x_unit, y_unit, z_unit, dx, dy, dz
    except:
        x, y = dat.x()[i:], dat.y()[i:]
        x_label, y_label = dat.x.label, dat.y.label
        x_unit, y_unit = dat.x.unit, dat.y.unit
        if smooth:
            y = y - ndimage.gaussian_filter(y, smooth_points, mode='nearest')
        results = x, y, x_label, y_label, x_unit, y_unit

    return results


def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def find_nearest_index(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx


def linear_func(x, m, b):
    return m*x+b


def Zeeman_energy(g, B):
    '''


    Parameters
    ----------
    g : g-factor
    B : magnetic fied in T

    Returns
    -------
    energy : in eV.

    '''
    mu_B = constant.physical_constants['Bohr magneton'][0]
    energy = g * mu_B * B  # energy in Joule
    energy = energy * \
        constant.physical_constants['joule-electron volt relationship'][0]  # energy in eV
    return energy


def Larmor_frequency(g, B):
    '''


    Parameters
    ----------
    g : g-factor
    B : magnetic fied in T

    Returns
    -------
    Larmor frequency : in Hz.

    '''
    mu_B = constant.physical_constants['Bohr magneton'][0]
    energy = g * mu_B * B  # energy in Joule

    energy = energy * \
        constant.physical_constants['joule-electron volt relationship'][0]  # energy in eV
    Planck = constant.physical_constants['Planck constant in eV/Hz'][0]
    frequency = energy / Planck  # frequency in Hz
    return frequency


def g_factor(m):
    Planck = constant.Planck
    mu_B = constant.physical_constants['Bohr magneton'][0]
    g = Planck/(mu_B * m)
    return g


def g_factor_from_frequency(f, B):
    '''


    Parameters
    ----------
    f : frequency
    B : field in Tesla

    Returns
    -------
    g : g-factor

    '''
    Planck = constant.Planck
    mu_B = constant.physical_constants['Bohr magneton'][0]
    g = (Planck * f)/(mu_B * B)
    return g


def slope(g_factor):
    Planck = constant.Planck
    mu_B = constant.physical_constants['Bohr magneton'][0]
    m = Planck/(mu_B * g)*1e9
    return m


alphabet = ['_1', '_2', '_3', '_4', '_5']


def get_measurements(dat):
    for letter in alphabet:
        try:
            getattr(dat, f'm1{letter}')
        except:
            break
    return alphabet.index(letter)


def exp_fit(x, a, b):
    return a*np.exp(-b * x)


def linear_fit(x, a, b):
    return -a*x + b


def model(V, b, c, V0, V1):
    '''

    Parameters
    ----------
    V : barrier voltage
    b : decay factor
    c : amplification factor
    V0 : Voltage when the barrier height is equal to the energy of the hole
    V1 : Voltage for which the barrier is zero

    Returns
    -------
    fitted_model : tunnel coupling trend in GHz
    '''
    fitted_model = 1e-9 * \
        np.sqrt(abs(16*c*(V-V0)/((V-V1)**2)))*np.exp(-b*np.sqrt(abs(V-V0)))
    return fitted_model


def function_fit(x, y, y_err, p0=[0.01, 2.6], bounds=((0, -5), (1, 0))):
    '''
    Parameters
    ----------
    x : gate voltage

    y : tunnel coupling points
    p0 : list of two guess values .
    bounds : list of bounds
    Returns/
    -------
    fitted data & plot

    '''

    xfit = np.linspace(-600, -200)
    popt, pcov = curve_fit(exp_fit, x, y, p0=p0,
                           # bounds = bounds,
                           sigma=y_err
                           )

    fit = exp_fit(xfit, popt[0], popt[1])
    errors = np.sqrt(np.diag(pcov))
    print('-----------------------------')
    print(f' a = {popt[0]:.3f} +- {errors[0]:.3f}')
    print(f' b = {popt[1]:.3f} +- {errors[1]:.3f}')

    fig = plt.figure(figsize=cm2inch(5, 5), constrained_layout=True)
    ax1 = plt.subplot2grid((1, 1), (0, 0))
    ax1.errorbar(x, y, yerr=y_err, fmt='o', ecolor='black',
                 ms=2, mfc='black', mec='black')
    ax1.plot(xfit, fit, c='grey', lw=1, linestyle='--', zorder=10)
    ax1.set_xlabel('gate (mV)')
    ax1.set_ylabel('$ t_{\mathrm{c}} \, \mathrm{(GHz)}$')
    ax1.set_xlim(np.min(x), np.max(x))
    ax1.set_ylim(0, 1.1*np.max(y))
    plt.legend(loc=0)
    plt.show()
    plt.close()

    return popt, errors


def Average(lst):
    return sum(lst) / len(lst)

def evaluate_fit(ydata, fitted_ydata):
    """
    Evaluate the goodness of fit based on the R^2 value.

    Parameters:
    - ydata: numpy.ndarray, the observed y-values.
    - fitted_ydata: numpy.ndarray, the predicted y-values from the model.
    - threshold: float, the R^2 threshold for determining an acceptable fit.

    Returns:
    - bool: True if the R^2 value exceeds the threshold, False otherwise.
    """
    # Calculate the sum of squares of residuals
    ss_res = np.sum((ydata - fitted_ydata) ** 2)

    # Calculate the total sum of squares
    ss_tot = np.sum((ydata - np.mean(ydata)) ** 2)

    # Calculate the R^2 value
    r_squared = 1 - (ss_res / ss_tot)

    return r_squared
