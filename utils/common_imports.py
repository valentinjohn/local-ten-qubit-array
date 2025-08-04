# -*- coding: utf-8 -*-
"""
Created on Thu May 16 10:27:42 2024

@author: vjohn
"""

# %% imports

import numpy as np
import math
from math import ceil
import pandas as pd
import ast
import warnings
from shapely.geometry import Polygon
from descartes import PolygonPatch
from scipy import constants
from scipy.optimize import curve_fit
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import cm, colormaps
import matplotlib.colorbar as colorbar
from matplotlib.colors import (ListedColormap, TwoSlopeNorm, Normalize,
                               LogNorm, LinearSegmentedColormap, TABLEAU_COLORS)


# requires proper path management
from utils.notebook_tools_costumized import fit_data, Gauss, Ramsey, Line
import utils.analysis_tools as tools
import utils.package_style

# %% global variables

plungers = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10']
all_plungers = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10']
barriers = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11', 'B12']
all_barriers = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11', 'B12']
gates = plungers + barriers
all_gates = all_plungers + all_barriers

neighbour_barriers = {'P1': ['B1', 'B2'],
                      'P2': ['B3', 'B4'],
                      'P3': ['B5', 'B6'],
                      'P4': ['B1', 'B7'],
                      'P5': ['B2', 'B3', 'B8', 'B9'],
                      'P6': ['B4', 'B5', 'B10', 'B11'],
                      'P7': ['B6', 'B12'],
                      'P8': ['B7', 'B8'],
                      'P9': ['B9', 'B10'],
                      'P10': ['B11', 'B12']}

one_column_width_cm = 8.8
two_column_width_cm = 18

# %% barrier and plunger masks

# Define rows (Q1 to Q10) and columns (B1 to B12)
rows = [f'Q{i}' for i in range(1, 11)]
columns_barrier = [f'B{i}' for i in range(1, 13)]
columns_plunger = [f'P{i}' for i in range(1, 11)]
columns = columns_plunger + columns_barrier

# Initialize the DataFrame with all values set to False
df_mask_nearest_neighbour = pd.DataFrame(False, index=rows, columns=columns)
df_mask_qubit_plunger= pd.DataFrame(False, index=rows, columns=columns)

# Dictionary with True values
qubit_neighbour_barriers = {
    'Q1': ['B1', 'B2'],
    'Q2': ['B3', 'B4'],
    'Q3': ['B5', 'B6'],
    'Q4': ['B1', 'B7'],
    'Q5': ['B2', 'B3', 'B8', 'B9'],
    'Q6': ['B4', 'B5', 'B10', 'B11'],
    'Q7': ['B6', 'B12'],
    'Q8': ['B7', 'B8'],
    'Q9': ['B9', 'B10'],
    'Q10': ['B11', 'B12']
}

qubit_plunger = {
    'Q1': ['P1'],
    'Q2': ['P2'],
    'Q3': ['P3'],
    'Q4': ['P4'],
    'Q5': ['P5'],
    'Q6': ['P6'],
    'Q7': ['P7'],
    'Q8': ['P8'],
    'Q9': ['P9'],
    'Q10': ['P10']
}

# Update the DataFrame based on the neighbour_barriers dictionary
for key, gates in qubit_neighbour_barriers.items():
    for gate in gates:
        df_mask_nearest_neighbour.loc[key, gate] = True

for key, gates in qubit_plunger.items():
    for gate in gates:
        df_mask_qubit_plunger.loc[key, gate] = True