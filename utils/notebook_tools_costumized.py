# -*- coding: utf-8 -*-
"""
Created on Tue May 18 13:17:01 2021

@author: fvanriggelen, adapted from script by David Franke
"""

from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import logging
from functools import partial
import inspect

#%% fitting functions

def Lorentz(x, A, FWHM, x0, y0):
    w = FWHM / 2
    return A / (((x - x0) / w)**2 + 1) + y0


def Gauss(x, A, FWHM, x0, y0):
    w = FWHM / (2 * np.sqrt(2 * np.log(2)))
    return A * np.exp(-(x - x0)**2 / (2 * w**2)) + y0


def dGauss(x, A, FWHM, x0, y0):
    w = FWHM / (2 * np.sqrt(2 * np.log(2)))
    return (x0 - x) / w**2 * A * np.exp(-(x - x0)**2 / (2 * w**2)) + y0

def Rabi(t, A, f, alpha, y0, phi):
        return A*np.cos(2*np.pi*f*t+phi)/(2*np.pi*f*t)**alpha+y0

def Cos(t, A, f, y0, phi):
    return A*np.cos(2*np.pi*f*t+phi)+y0

def Ramsey(t, A, f, T2, y0, phi):
    return A*np.cos(2*np.pi*t*f+phi)*np.exp(-t/T2)+y0

def ExpDec(x, A, m, y0):
    return A * np.exp(-x / m) + y0

def DoubExpDec(x, A1, m1, A2, m2, y0):
    return A1 * np.exp(-x / m1) + A2 * np.exp(-x / m2) + y0

def Power(x,A,alpha,y0):
    return A*alpha**x+y0

def Line(x, m, y0):
    return m*x+y0

def LineOrigin(x, m):
    return m*x

def Hahn_decay(x, A, m, b, y0):
    return A * np.exp(-(x / m)**b) + y0

#%% fit routine


def fit_data(xdata, ydata, p0=None, func=dGauss,
             plot=True, return_cov=False, verbose=0,fix_params = {}, **kwargs):

    x_range = np.linspace(np.min(xdata), np.max(xdata), num=500)
    p0dict = {}
    if p0 is None:
        if func in [Gauss, dGauss, Lorentz]:

            p0dict = {
                    'A':np.max(ydata) - np.mean(ydata),
                    'FWHM': (x_range[-1] - x_range[0]) * 0.2,
                    'x0': xdata[np.argmax(ydata)],
                    'y0': np.mean(ydata)
                    }
            if verbose:
                logging.info('p0: ' + str(p0))
        elif func is Rabi:
            p0dict = {
                    'A':np.max(ydata) - np.mean(ydata),
                    'f' :1/(x_range[-1] - x_range[0])*2,
                    'alpha': 0.1,
                    'y0': np.mean(ydata),
                    'phi': 0
                    }

            if verbose:
                logging.info('p0: ' + str(p0))

        elif func is ExpDec:
            start = np.mean(ydata[:5])
            baseline = np.mean(ydata[int(3 * len(ydata) / 4):])
            p0dict = {'A': start - baseline,
                  'm': (xdata[1] - xdata[-1]) / 2,
                  'y0': baseline}
        elif func is DoubExpDec:
            start = np.mean(ydata[:5])
            baseline = np.mean(ydata[int(3 * len(ydata) / 4):])
            p0dict = {'A1': start - baseline,
                  'm1': (xdata[1] - xdata[0]) / 2,
                  'A2': start - baseline,
                  'm2': (xdata[1] - xdata[0]) / 4,
                  'y0': baseline}

    if fix_params:
        func = partial(func, **fix_params)

    try:
        if p0dict:
            p0 = [p0dict[arg] for arg in inspect.getfullargspec(func).args[1:]]

        # check p0 feasability
        if 'bounds' in kwargs.keys() and p0 is not None:
            for i,p in enumerate(p0):
                lower, upper = (kwargs['bounds'][0][i], kwargs['bounds'][1][i])
                if not lower <=p <= upper:
                    p=min([max([lower, p]),upper])
        p1, covar = curve_fit(func, xdata, ydata, p0=p0, **kwargs)
        if plot:
            plt.plot(xdata,ydata)
            plt.plot(x_range, func(x_range, *p1))

        if return_cov:
            return np.sqrt(np.diag(covar)), p1
        else:
            return p1
    except RuntimeError as Err:
        logging.warning(Err)

#%%
def pcolormesh_centre(x, y, im, xlabel, ylabel, figure = None, cmap= None, vmin=None, vmax=None):
    """ Wrapper for pcolormesh to plot pixel centres at data points.

    for documentation: matplotlib.pyplot.pcolormesh?
    """
    dx=np.diff(x)
    dy=np.diff(y)
    dx=np.hstack( (dx[0], dx, dx[-1]))
    dy=np.hstack( (dy[0], dy, dy[-1]))
    xx=np.hstack( (x, x[-1]+dx[-1]))-dx/2
    yy=np.hstack( (y, y[-1]+dy[-1]))-dy/2
    if figure == None:
        plt.figure()
    if (vmin is not None) and (vmax is not None):
        plt.pcolormesh(xx, yy, im, vmin=vmin, cmap= cmap, vmax= vmax)
    else:
        plt.pcolormesh(xx, yy, im, cmap= cmap)
    #plt.colorbar(label = colorbar_label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()

#%%
def derandomize(X, Y):
    Z = [x for _,x in sorted(zip(X,Y))]
    X_sorted = sorted(X)
    return X_sorted, Z


#%%
def root_mean_sqrt_error(x_data, y_data, func, fit_parameters):
    predictions = func(x_data, *fit_parameters)
    targets = y_data
    rmse = np.sqrt(((predictions - targets) ** 2).mean())
    return rmse

def chisq(x_data, y_data, sigma, func, fit_parameters):
    predictions = func(x_data, *fit_parameters)
    targets = y_data
    chisq = np.sqrt((((predictions - targets)/sigma) ** 2).mean())
    return chisq

#%%
def SDOM(values):
    mean = np.mean(values)
    len_n = np.shape(values)[0]
    std = []
    for value in values:
        std.append((value - mean)**2)
    sdom = np.sqrt(1/(len_n*(len_n-1)) * sum(std))
    return sdom
