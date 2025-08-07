from pathlib import Path
from config import DATA_DIR, PROJECT_DIR
script_dir = PROJECT_DIR

import sys
import subprocess

from utils.common_imports import *
from utils.utils_final_figures import (plot_infidelities, plot_boxplot_ratios,
                                       plot_gfactor_vs_occupation, load_LSES_data, plot_raw_data,
                                       plot_locality_boxplots, plot_locality_array)
import utils.analysis_tools as tools
from gfactor_tunability.utils_statistics_gfactor_tunability import plot_LSES_individual
from drive_efficiency.utils_statistics_driving_slope import plot_rabi_efficiency_individual, load_driving_slope_data

PYTHON = sys.executable

def main():
    print("Plotting Figure 1 ...")
    plot_infidelities(save_path=script_dir / 'figure1' / 'subfigures')

    print("Plotting Figure 2 ...")
    qubit = 'Q5'
    num_holes = '1h'
    gate = 'P5'
    plot_raw_data(qubit, num_holes, gate, properties=['LSES'], save_path=script_dir / 'figure2' / 'subfigures')

    plot_raw_data(qubit, num_holes, gate, properties=['drive_efficiency'],
                  save_path=script_dir / 'figure2' / 'subfigures')

    averages, medians = plot_boxplot_ratios(save_path=script_dir / 'figure2' / 'subfigures')
    print(averages)
    print(medians)

    plot_gfactor_vs_occupation(save_path=script_dir / 'figure2' / 'subfigures', hole_occupation=['1h', '3h', '5h'],
                               title=False)

    df_gfactor, unit = load_LSES_data()
    qubits_to_plot = ['Q5', 'Q6']
    num_holes_to_plot = ['1h', '3h', '5h']

    vmax = max(df_gfactor[num_holes].abs().max().max() for num_holes in ['1h', '3h', '5h'])
    quantity = r'$\partial{g^*}/\partial{V}$'
    cbar_label = rf'{quantity} ({unit})'
    for num_holes in num_holes_to_plot:
        for qubit in qubits_to_plot:
            plot_LSES_individual(qubit, num_holes, df_gfactor, vmax=vmax,
                                 save=True, show_plot=True,
                                 unit=cbar_label,
                                 dark_background=False,
                                 transparent_png=True,
                                 plot_with_cbar=False,
                                 figsize=tools.cm2inch(two_column_width_cm / 4 * 0.9, 3),
                                 save_path=script_dir / 'figure2' / 'subfigures')

    df = load_driving_slope_data(DATA_DIR)
    qubits_to_plot = ['Q5', 'Q6']
    num_holes_to_plot = ['1h', '3h', '5h']

    vmax = max(df[num_holes].abs().max().max() for num_holes in ['1h', '3h', '5h'])
    for num_holes in num_holes_to_plot:
        for qubit in qubits_to_plot:
            plot_rabi_efficiency_individual(qubit, num_holes, df,
                                            vmax=vmax,
                                            save=True,
                                            dark_background=False,
                                            dark_color='black',
                                            transparent_png=True,
                                            figsize=tools.cm2inch(two_column_width_cm / 4 * 0.9, 3),
                                            plot_with_cbar=False,
                                            save_path=script_dir / 'figure2' / 'subfigures')

    print("Plotting Figure 4 ...")
    plot_locality_boxplots(num_holes=['1h', '3h'], gate_types=['plunger', 'barrier'], title=False,
                           save_path=script_dir / 'figure4' / 'subfigures')

    plot_locality_array(num_holes=['1h', '3h'], gate_types=['plunger', 'barrier'], title=False,
                        save_path=script_dir / 'figure4' / 'subfigures')

    # print("Running script for Supplementary figures...")
    # subprocess.run([PYTHON, "supp_figures.py"], check=True)

if __name__ == '__main__':
    main()
