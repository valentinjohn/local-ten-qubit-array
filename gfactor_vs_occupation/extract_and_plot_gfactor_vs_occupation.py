# %% Path management

import os
from pathlib import Path
from config import DATA_DIR

try:
    script_dir = Path(__file__).resolve().parent
except:
    script_dir = Path(os.getcwd()) / 'gfactor_vs_occupation'
fig_path = script_dir / "images"
subfolder = fig_path / 'driving_efficiency_extraction'
data_path = DATA_DIR

# %% imports

from utils import *

# %% core tools setup to load data

from core_tools.data.ds.ds_hdf5 import load_hdf5_uuid
uuids_data_path = DATA_DIR / 'uuid_datasets'

# %% methods

def create_dicts():
    uuids = {}
    slopes = {}

    for n in range(1,11):
        uuids[f'Q{n}'] = {}
        slopes[f'Q{n}'] = {}

    return uuids, slopes

def get_gfactor_from_uuid(uuid, qubit, field_factor=0.69):
    dat = load_hdf5_uuid(uuid, uuids_data_path)
    drive_label = f'q{qubit[1:]}'
    f_larmor = dat.snapshot['measurement']['sequence']['pc0'][f'{drive_label}_pulses']['p0']['frequency']
    b_field = field_factor*dat.snapshot['station']['instruments']['magnet']['parameters']['field']['value']
    g_factor = flarmor_to_gfactor_conversion(f_larmor, b_field)
    return g_factor

# %% lists of uuids

uuids, g_factors = create_dicts()

uuids['Q1']['1h'] = 1722277849383283691
uuids['Q1']['3h'] = 1709751306413283691

uuids['Q2']['5h'] = 1711018205078283691

uuids['Q3']['1h'] = 1718182630972283691
uuids['Q3']['3h'] = 1718100466578283691

uuids['Q4']['1h'] = 1709744104524283691
uuids['Q4']['3h'] = 1722265817693283691

uuids['Q5']['1h'] = 1711039785032283691
uuids['Q5']['3h'] = 1720000832365283691
uuids['Q5']['5h'] = 1720191222913283691

uuids['Q6']['1h'] = 1720796085240283691
uuids['Q6']['3h'] = 1711028778711283691
uuids['Q6']['5h'] = 1721140037923283691

print('using the new Q7(1h+) from the more isolated regime')
uuids['Q7']['1h'] = 1721933670702283691
uuids['Q7']['3h'] = 1721662593637283691

uuids['Q8']['1h'] = 1719480081221283691
uuids['Q8']['3h'] = 1709831512124283691

uuids['Q9']['1h'] = 1720006970702283691
uuids['Q9']['3h'] = 1711055562146283691

uuids['Q10']['1h'] = 1711136789885283691
uuids['Q10']['3h'] = 1721924247956283691

# %%

for qubit in uuids:
    g_factors[qubit] = {}
    for num_holes in uuids[qubit]:
        uuid = uuids[qubit][num_holes]
        g_factor = get_gfactor_from_uuid(uuid, qubit, field_factor=0.69)
        g_factors[qubit][num_holes] = g_factor

df_gfactors = pd.DataFrame(g_factors)
df_gfactors.to_csv(os.path.join(data_path, 'gfactor_vs_occupation.csv'))

# %% plot

file_name = 'gfactor_vs_occupation'

qubits = uuids.keys()

fig = plt.figure(figsize=(4,3))

for qubit in qubits:
    x = df_gfactors[qubit].dropna().keys()
    y = df_gfactors[qubit].dropna()
    plt.plot(x, y, marker='.', label=qubit)

plt.ylabel(r'$g^*$')
plt.xlabel('hole occupation')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()

fig.savefig(os.path.join(fig_path, f"{file_name}.png"), dpi=300, transparent=True)
fig.savefig(os.path.join(fig_path, f"{file_name}.pdf"), dpi=300, transparent=True)

plt.show()

# %% normalised plot

file_name = 'gfactor_vs_occupation_normalised'

qubits = uuids.keys()

fig = plt.figure(figsize=(3,3))

for qubit in qubits:
    x = df_gfactors[qubit].dropna().keys()
    y = df_gfactors[qubit].dropna() - df_gfactors[qubit].loc['1h']
    plt.plot(x, y, marker='.')

plt.ylabel(r'g - g$_{\rm{1h}}$')
plt.xlabel('hole_occupation')

plt.ylim(-0.06, 0.06)

plt.tight_layout()

fig.savefig(os.path.join(fig_path, f"{file_name}.png"), dpi=300, transparent=True)
fig.savefig(os.path.join(fig_path, f"{file_name}.pdf"), dpi=300, transparent=True)

plt.show()

# %%

# min_value = 0.5 #df_gfactors.min().min()
# max_value = 0.65 #df_gfactors.max().max()
#
# norm = Normalize(vmin=min_value, vmax=max_value, clip=True)
# cmap = mpl.colormaps['cividis']
#
# for num_holes in df_gfactors.index:
#     fig, ax = ten_qubit_plot_343(df_gfactors.loc[num_holes].dropna(),
#                                  qubits=df_gfactors.loc[num_holes].dropna().index,
#                                  cmap=cmap,
#                                  norm=norm,
#                                  plot_cmap=False)
#
#     save = True
#     if save:
#         file_name = f"gfactors_{num_holes}"
#         if not os.path.exists(fig_path):
#             os.makedirs(fig_path)
#         fig.savefig(os.path.join(fig_path, f"{file_name}.png"), dpi=300, transparent=True)
#         fig.savefig(os.path.join(fig_path, f"{file_name}.pdf"), dpi=300, transparent=True)
#
#     plt.show()
#
#     hcbar_path = os.path.join(fig_path, 'horizontal_colorbars')
#     vcbar_path = os.path.join(fig_path, 'vertical_colorbars')
#     if not os.path.exists(hcbar_path):
#         os.makedirs(hcbar_path)
#     if not os.path.exists(vcbar_path):
#         os.makedirs(vcbar_path)
#     save_colorbar(norm, cmap, r'$g^*$', os.path.join(hcbar_path, file_name),
#                   orientation='horizontal', figsize=tools.cm2inch(3.5, 2.5), tick_number=4)
#     save_colorbar(norm, cmap, r'$g^*$', os.path.join(vcbar_path, file_name),
#                   orientation='vertical', figsize=tools.cm2inch(2.5, 3.5), tick_number=4)

# %% mean and std of g factor

# all_gfactor = df_gfactors.to_numpy().flatten()
# all_gfactor = all_gfactor[~np.isnan(all_gfactor)]
#
# mean_gfactor = np.mean(all_gfactor)
# std_gfactor = np.std(all_gfactor)
#
# print(f"mean g factor: {mean_gfactor:.3f}")
# print(f"std g factor: {std_gfactor:.3f}")
