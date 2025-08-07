from pathlib import Path

import json
from config import DATA_DIR, PROJECT_DIR
from utils.utils import *
import numpy as np
from utils.utils_final_figures import get_fname, load_xr_hdf5, find_m1_fraction

# %%

data_path_gfactor_tunability = PROJECT_DIR / 'gfactor_tunability/uuids_gfactor_tunability.json'
with open(data_path_gfactor_tunability, 'r') as file:
    uuids_gfactor_tunability = json.load(file)

# %%
# def get_min_max_values(uuid):
#     y, x, z, y_label, x_label = get_data(uuid)
    # print(f'Max value of z: {np.nanmax(z)}')
    # print(f'Min value of z: {np.nanmin(z)}')
    # return np.nanmax(z), np.nanmin(z)

# fname = get_fname(uuid)
# uuid_data_path = DATA_DIR / 'uuid_datasets' / fname
# dat = load_xr_hdf5(uuid_data_path)
# zdata = dat.m1_3_fraction.data

# %%

uuids = {'Q1': 1_712_302_934_272_283_691,
         'Q2': 1_712_134_612_289_283_691,
         'Q3': 1_712_588_841_590_283_691,
         'Q4': 1_712_863_319_622_283_691,
         'Q5': 1_712_840_510_888_283_691,
         'Q6': 1_712_299_846_122_283_691,
         'Q7': 1_712_674_162_658_283_691,
         'Q8': 1_712_221_015_516_283_691,
         'Q9': 1_712_146_299_991_283_691,
         'Q10': 1_712_762_335_740_283_691}

qubits = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10']
drive_gates = ['P1', 'P2', 'B5', 'B7', 'P5', 'P6', 'B12', 'P8', 'P9', 'B12']
hole_occupation = ['3h', '5h', '1h', '1h', '1h', '3h', '1h', '3h', '3h', '1h']


# %% offset and amplitude for RB

# create empty dataframe
df = pd.DataFrame(index=['amplitude', 'offset', ], columns=qubits)

for qubit, uuid in uuids.items():
    fname = get_fname(uuid)
    uuid_data_path = DATA_DIR / 'uuid_datasets' / fname
    dat = load_xr_hdf5(uuid_data_path)

    for n in range(1, 5):
        n_fraction = find_m1_fraction(dat.attrs['keywords'], n)
        if n_fraction is not None:
            z = dat[n_fraction].data

    vmax, vmin = z.max(), z.min()

    offset = (vmax + vmin) / 2
    amplitude = (vmax - vmin) / 2
    # df.loc['max_value', qubit] = np.round(vmax, 2)
    # df.loc['min_value', qubit] = np.round(vmin, 2)
    df.loc['amplitude', qubit] = np.round(amplitude, 3)
    df.loc['offset', qubit] = np.round(offset, 3)

# %% visibility and read out time


# create empty dataframe
df = pd.DataFrame(index=['visibility', 't_measure', 't_settle'], columns=qubits)

for qubit, uuid in uuids.items():
    fname = get_fname(uuid)
    uuid_data_path = DATA_DIR / 'uuid_datasets' / fname
    dat = load_xr_hdf5(uuid_data_path)

    for n in range(1, 5):
        n_fraction = find_m1_fraction(dat.attrs['keywords'], n)
        if n_fraction is not None:
            z = dat[n_fraction].data

    vmax, vmin = z.max(), z.min()

    t_measure = dat.snapshot['measurement']['sequence']['settings']['q7,q10']['psb']['t_measure']
    t_settle = dat.snapshot['measurement']['sequence']['settings']['q7,q10']['psb']['t_settle']

    visibility = vmax - vmin
    df.loc['visibility', qubit] = np.round(visibility, 2)
    df.loc['t_measure', qubit] = np.round(t_measure, 2)
    df.loc['t_settle', qubit] = np.round(t_settle, 2)
