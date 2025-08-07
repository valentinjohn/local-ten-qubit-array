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

from utils.utils import *
import matplotlib.pyplot as plt

# %% core tools setup to load data

from core_tools.data.ds.ds_hdf5 import load_hdf5_uuid

uuids_data_path = DATA_DIR / 'uuid_datasets'


# %% import data (use RB calibration data to get integration time and visibility)

uuid_qubits = {'Q1': 1_712_302_934_272_283_691,
              'Q2': 1_712_134_612_289_283_691,
              'Q3': 1_712_588_841_590_283_691,
              'Q4': 1_712_863_319_622_283_691,
              'Q5': 1_712_840_510_888_283_691,
              'Q6': 1_712_299_846_122_283_691,
              'Q7': 1_712_674_162_658_283_691,
              'Q8': 1_712_221_015_516_283_691,
              'Q9': 1_712_146_299_991_283_691,
              'Q10': 1_712_762_335_740_283_691}

# %% plot data

# create empty dataframe
df = pd.DataFrame(index=['PSB pair', 'sensor', 'visibility', 't_measure', 't_settle'], columns=uuid_qubits.keys())

for qubit, uuid in uuid_qubits.items():
    print('----------')
    print(f'Qubit {qubit}')

    data = load_hdf5_uuid(uuid, uuids_data_path)

    # read_out_pair = df.loc['PSB pair', qubit]
    print(data.snapshot['measurement']['sequence']['circuit']['statements'])
    read_out_pair = data.snapshot['measurement']['sequence']['circuit']['statements'][-1].split( )[1][:-1]
    read_out_pair = f'q{read_out_pair[0]},q{read_out_pair[2:]}'
    df.loc['PSB pair', qubit] = read_out_pair

    z = data.m1_2()

    vmax, vmin = z.max(), z.min()

    sensor_acquisition = ['S_North_acq', 'S_East_acq', 'S_West_acq', 'S_South_acq']
    for sensor_acq in sensor_acquisition:
        if sensor_acq in data.snapshot['measurement']['sequence']['pc0'].keys():
            df.loc['sensor', qubit] = sensor_acq
            break

    t_measure = data.snapshot['measurement']['sequence']['settings'][read_out_pair]['psb']['t_measure']
    t_settle = data.snapshot['measurement']['sequence']['settings'][read_out_pair]['psb']['t_settle']

    visibility = vmax - vmin
    df.loc['visibility', qubit] = np.round(visibility, 2)
    df.loc['t_measure', qubit] = np.round(t_measure, 2)
    df.loc['t_settle', qubit] = np.round(t_settle, 2)

df.pr