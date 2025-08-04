# %% Path management

import os
from pathlib import Path
from config import DATA_DIR

try:
    script_dir = Path(__file__).resolve().parent
except:
    script_dir = Path(os.getcwd()) / 'gfactor_tunability'

fig_path = script_dir / "images"
data_path = DATA_DIR
subfolder = fig_path / 'gfactor_tunability_extraction'

# %% imports

from utils import *
from gfactor_tunability.utils_statistics_gfactor_tunability import gfactor_tunability_plot, create_nested_dict


# %% load uuids
import json

data_path = script_dir / 'uuids_gfactor_tunability.json'
with open(data_path, 'r') as file:
    uuids = json.load(file)

# %% optional specified limits of specified uuids for extraction of frequency peak

# uuid: [min_amplitude, max_width, min_width]
uuid_limits = {1720018123947283691: [0.15, 5e6, 0.5e6],
               1719565062821283691: [0.04, 2e6, 0.3e6],
               1719565646638283691: [0.05, 2e6, 0.1e6],
               1718633016514283691: [0.5, 5e6, 0.5e6],
               1718808741475283691: [0.2, 5e6, 0.5e6],
               1721913882931283691: [-0.8, 5e6, 0.5e6],
               1721658286971283691: [0.5, 1e6, 0.5e6],
               1718808075878283691: [0.2, 2e6, 1e6],
               1721147469417283691: [0.1, 2e6, 0.5e6],
               1721913882931283691: [0.3, 2e6, 0.5e6]}

for uuid in uuids['Q4']['1h']['plunger'] :
    uuid_limits[uuid] = [0.2, 1e6, 0.1e6]

# %% extract LSES

plot = True

slopes = defaultdict(create_nested_dict)
for qubit in uuids:
    for num_holes in uuids[qubit]:
        slope_plungers = gfactor_tunability_plot(uuids[qubit][num_holes]['plunger'],
                                                 plungers, qubit, num_holes,
                                                 plot=plot,
                                                 uuid_limits=uuid_limits)
        slope_barriers = gfactor_tunability_plot(uuids[qubit][num_holes]['barrier'],
                                                 barriers, qubit, num_holes,
                                                 plot=plot,
                                                 uuid_limits=uuid_limits)
        slopes[qubit][num_holes] = slope_plungers + slope_barriers

# %% colorbar

cmap = 'viridis'
norm = colors.Normalize(vmin=0, vmax=1)
unit_label = r'$P_{\text{even}}$'
filename = fig_path / 'colorbar_gfactor_tunability'

save_colorbar(norm, cmap, unit_label, filename, orientation='horizontal', figsize=None,
                  ticklabelsize=None, labelsize=None, shrink=1, tick_number=3, location=None, labelpad=None)

# %% saving the results of the fits in a csv file, so you dont have to run the whole analysis again and again

qubits = slopes.keys()

slope_1h_gfactor = [slopes[q]['1h'] if not isinstance(slopes[q]['1h'], defaultdict) else [np.nan]*22 for q in slopes.keys()]
slope_3h_gfactor = [slopes[q]['3h'] if not isinstance(slopes[q]['3h'], defaultdict) else [np.nan]*22 for q in slopes.keys()]
slope_5h_gfactor = [slopes[q]['5h'] if not isinstance(slopes[q]['5h'], defaultdict) else [np.nan]*22 for q in slopes.keys()]

ds_gfactor_1hole = pd.DataFrame(slope_1h_gfactor, index=qubits, columns=gates)
ds_gfactor_3hole = pd.DataFrame(slope_3h_gfactor, index=qubits, columns=gates)
ds_gfactor_5hole = pd.DataFrame(slope_5h_gfactor, index=qubits, columns=gates)

ds_gfactor_1hole.to_csv(os.path.join(data_path, 'gfactor_tunability_1hole.csv'))
ds_gfactor_3hole.to_csv(os.path.join(data_path, 'gfactor_tunability_3hole.csv'))
ds_gfactor_5hole.to_csv(os.path.join(data_path, 'gfactor_tunability_5hole.csv'))

print('finished')
