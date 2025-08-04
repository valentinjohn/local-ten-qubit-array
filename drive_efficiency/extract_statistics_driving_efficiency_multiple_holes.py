# %% Path management

from pathlib import Path
from config import DATA_DIR

script_dir = Path(__file__).resolve().parent
fig_path = script_dir / "images"
data_path = DATA_DIR
subfolder = fig_path / 'driving_efficiency_extraction'

# %% imports

from utils import *
from drive_efficiency.utils_statistics_driving_slope import (driving_speed_amp_fft_plot,
                                                                      replace_nan_with_zero,
                                                                      create_dicts)

# %% load uuids
import json

uuids_data_path = os.path.join(script_dir, 'uuids_driving_efficiency.json')
with open(uuids_data_path, 'r') as file:
    uuids = json.load(file)

# %% performing the analysis to find the driving strength as a function of amplitude

plot = True
plot_ramsey = False
check_only_single_dataset = False

do_plungers = True
do_barriers = True

plunger_gates = plungers
barrier_gates = barriers
gates = plunger_gates + barrier_gates

if check_only_single_dataset:
    qubit = 'Q5'
    num_holes = '1h'

    plot_ramsey = True

    uuids_plunger = uuids[qubit][num_holes]['plunger'].copy()
    uuids_barrier = uuids[qubit][num_holes]['barrier'].copy()
    full_uuids = uuids
    uuids, _ = create_dicts()
    uuids[qubit][num_holes]['plunger'] = uuids_plunger
    uuids[qubit][num_holes]['barrier'] = uuids_barrier

slopes = defaultdict(create_nested_dict)
for qubit in uuids:
    for num_holes in uuids[qubit]:
        if do_plungers:
            uuids_plunger = uuids[qubit][num_holes]['plunger']
            if np.shape(uuids_plunger) == (0,):
                slope_plungers = [np.nan]*10
            else:
                plunger_range = [0, 10]
                slope_plungers, _, fig_pl = driving_speed_amp_fft_plot(uuids[qubit][num_holes]['plunger'][plunger_range[0]:plunger_range[1]],
                                                                       plunger_gates[plunger_range[0]:plunger_range[1]],
                                                                       plot=plot, plot_ramsey=plot_ramsey,
                                                                       print_id=False)
                slope_plungers = replace_nan_with_zero(slope_plungers)

                file_name = f'EDSR_efficiency_extraction_{qubit}_{num_holes}_plungers'
                fig_pl.suptitle(f'{qubit} ({num_holes})', size=12, weight='bold')
                fig_pl.savefig(os.path.join(subfolder, f"{file_name}.png"), dpi=300, transparent=True)
                fig_pl.savefig(os.path.join(subfolder, f"{file_name}.pdf"), dpi=300, transparent=True)

        if do_barriers:
            uuids_barrier = uuids[qubit][num_holes]['barrier']
            if np.shape(uuids_barrier) == (0,):
                slope_barriers = [np.nan]*12
            else:
                barrier_range = [0, 12]
                # barrier_range = [4, 5]
                slope_barriers, _, fig_bar = driving_speed_amp_fft_plot(uuids[qubit][num_holes]['barrier'][barrier_range[0]: barrier_range[1]],
                                                                        barrier_gates[barrier_range[0]: barrier_range[1]],
                                                                        plot=plot, plot_ramsey=plot_ramsey,
                                                                        print_id=False)
                slope_barriers = replace_nan_with_zero(slope_barriers)

                file_name = f'EDSR_efficiency_extraction_{qubit}_{num_holes}_barriers'
                fig_bar.suptitle(f'{qubit} ({num_holes})', size=12, weight='bold')
                # plt.tight_layout()
                fig_bar.savefig(os.path.join(subfolder, f"{file_name}.png"), dpi=300, transparent=True)
                fig_bar.savefig(os.path.join(subfolder, f"{file_name}.pdf"), dpi=300, transparent=True)

        if do_plungers and do_plungers:
            slopes[qubit][num_holes] = slope_plungers + slope_barriers

# check_single_dataset = False
if check_only_single_dataset:
    uuids = full_uuids

# %% colorbar

cmap = 'viridis'
norm = colors.Normalize(vmin=0, vmax=1)
unit_label = 'FFT amplitude (a.u.)'
filename = fig_path / 'colorbar_driving_efficiency'

save_colorbar(norm, cmap, unit_label, filename, orientation='horizontal', figsize=None,
                  ticklabelsize=None, labelsize=None, shrink=1, tick_number=3, location=None, labelpad=None)


# %% saving the results of the fits in a csv file, so you dont have to run the whole analysis again and again

qubits = slopes.keys()

slope_1h_driving = [slopes[q]['1h'] for q in slopes.keys() if '1h' in slopes[q]]
slope_3h_driving = [slopes[q]['3h'] for q in slopes.keys() if '3h' in slopes[q]]
slope_5h_driving = [slopes[q]['5h'] for q in slopes.keys() if '5h' in slopes[q]]

ds_driving_1hole = pd.DataFrame(slope_1h_driving, index=qubits, columns=gates)
ds_driving_3hole = pd.DataFrame(slope_3h_driving, index=qubits, columns=gates)
ds_driving_5hole = pd.DataFrame(slope_5h_driving, index=qubits, columns=gates)

ds_driving_1hole.to_csv(os.path.join(data_path, 'driving_strength_1hole.csv'))
ds_driving_3hole.to_csv(os.path.join(data_path, 'driving_strength_3hole.csv'))
ds_driving_5hole.to_csv(os.path.join(data_path, 'driving_strength_5hole.csv'))
