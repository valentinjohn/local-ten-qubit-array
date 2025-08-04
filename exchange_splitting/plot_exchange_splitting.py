# %% Path management

import os
from pathlib import Path
from config import DATA_DIR

try:
    script_dir = Path(__file__).resolve().parent
except:
    script_dir = Path(os.getcwd()) / 'exchange_splitting'

fig_path = script_dir / "images"
data_path = DATA_DIR

# %% imports

from utils.utils import *
import matplotlib.pyplot as plt

# %% core tools setup to load data

from core_tools.data.ds.ds_hdf5 import load_hdf5_uuid
uuids_data_path = DATA_DIR / 'uuid_datasets'

#%% import data 
uuid_qubits = {} 
uuid_qubits['Q1'] = (1707743903086283691, 'J1')
uuid_qubits['Q2'] = (1707403925475283691, 'J4') 
uuid_qubits['Q3'] = (1707239639996283691, 'J5')
uuid_qubits['Q4'] = (1707744182252283691, 'J1')
uuid_qubits['Q5'] = (1707726484464283691, 'J8')
uuid_qubits['Q6'] = (1707404118788283691, 'J4')
uuid_qubits['Q7'] = (1706887485905283691, 'J6') 
uuid_qubits['Q8'] = (17077_52063_460283691, 'J7')
uuid_qubits['Q9'] = (17074_87708168_283691, 'J9')
uuid_qubits['Q10'] = (17072_99495940_283691, 'J12') 

#%% import data grouped into J gate 
uuid_exchange = {}
uuid_exchange['J1'] = {}
uuid_exchange['J4'] = {}
uuid_exchange['J7'] = {}
uuid_exchange['J9'] = {}

uuid_exchange['J1']['Q1'] = 1707743903086283691
uuid_exchange['J1']['Q4'] = 1707744182252283691
uuid_exchange['J4']['Q2'] = 1707403925475283691
uuid_exchange['J4']['Q6'] = 1707404118788283691
uuid_exchange['J7']['Q4'] = 1707744791249283691
uuid_exchange['J7']['Q8'] = 1707752063460283691
uuid_exchange['J9']['Q5'] = 1707489513898283691
uuid_exchange['J9']['Q9'] = 1707486630310283691

#%% plot data per J

for gate in ['J1', 'J4', 'J7', 'J9']:
    for qubit, uuid in uuid_exchange[gate].items():
        print('----------')
        print(f'Qubit {qubit}')
        data = load_hdf5_uuid(uuid, uuids_data_path)

        x, x_label, x_unit = data.m1_3.i(), data.m1_3.i.label, data.m1_3.i.unit
        y, y_label, y_unit  = data.m1_3.j(), data.m1_3.j.label, data.m1_3.j.unit
        z = data.m1_2()

        fig, ax = plt.subplots(1, 1, figsize = cm2inch(4.5,4))

        c = plt.pcolor(y*1e-6,x,1-z,
                        # vmin = 0.25,
                        # vmax = 0.76
                       )

        ax.set_ylabel(f'$\\Delta${gate} (mV)')
        ax.set_xlabel('$f$ (MHz)')
        # ax.set_xlim((336, 365))
        # ax.set_ylim((-36, -20))

        ax.annotate(f'{qubit}', xy=(338, -22), color = 'w')


        fig.colorbar(c, ax=ax,
                     location = 'top',
                     shrink = 0.7)


        fig.tight_layout()
        fig.savefig(os.path.join(fig_path, f'exchange_{qubit}{gate}.png'), dpi=300)
        fig.savefig(os.path.join(fig_path, f'exchange_{qubit}{gate}.pdf'), dpi=300, transparent=True)
        plt.show()

#%% plot data per qubits 

for qubit, (uuid, gate) in uuid_qubits.items(): 
    print('----------')
    print(f'Qubit {qubit}')
    
    data = load_hdf5_uuid(uuid, uuids_data_path)
    
    x, x_label, x_unit = data.m1_3.i(), data.m1_3.i.label, data.m1_3.i.unit
    y, y_label, y_unit  = data.m1_3.j(), data.m1_3.j.label, data.m1_3.j.unit
    z = data.m1_2()

    
    fig, ax = plt.subplots(1, 1, figsize = cm2inch(4.5,4))
    
    c = plt.pcolor(y*1e-6,x,z,
                    # vmin = 0.25,
                    # vmax = 0.76
                   )
    
    ax.set_ylabel(f'$\\Delta${gate} (mV)')
    ax.set_xlabel('$f$ (MHz)')
    if qubit == 'Q10':
      ax.set_xlim((300, 340))
    
    clb = fig.colorbar(c, ax=ax, 
                       location = 'right',
                       shrink = 0.6)
    clb.ax.set_title('$P_{even}$', loc = 'left')
    plt.title(f'Qubit {qubit}', fontsize = 8)
    
    fig.tight_layout()
    fig.savefig(os.path.join(fig_path, f'exchange_{qubit}.png'), dpi=300)
    fig.savefig(os.path.join(fig_path, f'exchange_pair_{qubit}.pdf'), dpi=300, transparent=True)
    plt.show()
