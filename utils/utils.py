# -*- coding: utf-8 -*-
"""
Created on Wed May 15 14:46:30 2024

@author: vjohn
"""
# %%
import sys
import os

# main_file = sys.modules['__main__'].__file__
# script_dir = os.path.dirname(os.path.abspath(main_file))
# parent_dir = os.path.dirname(script_dir)
#
# if parent_dir not in sys.path:
#     sys.path.append(parent_dir)
    
# %% imports

from collections import defaultdict
from utils.common_imports import *

# %% path management
import os
import sys
from pathlib import Path

def get_script_directory():
    # If __file__ is set, use it to find the script directory
    if '__file__' in globals():
        return Path(__file__).resolve().parent
    # Fallback for interactive environments (like Jupyter or single-cell executions)
    return Path(sys.argv[0] if sys.argv[0] else '.').resolve().parent

def get_fig_path():
    script_dir = get_script_directory()
    save_path = os.path.join(script_dir, 'images')

    # Create the 'Figures' folder if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    return save_path

def get_data_directory():
    script_dir = get_script_directory()
    parent_dir = os.path.dirname(script_dir)  # Get the parent directory
    data_dir = os.path.join(parent_dir, 'data')
    return data_dir

# %% Plotting

def create_343_subplot(style='plunger', num=1):
    # Plunger mosaic configuration
    plunger_mosaic = [['.', 'P1', '.', 'P2', '.', 'P3', '.'],
                      ['P4', '.', 'P5', '.', 'P6', '.', 'P7'],
                      ['.', 'P8', '.', 'P9', '.', 'P10', '.']]

    # Barrier mosaic configuration
    barrier_mosaic = [['B1', 'B2', 'B3', 'B4', 'B5', 'B6'],
                      ['B7', 'B8', 'B9', 'B10', 'B11', 'B12']]

    if style == 'plunger':
        fig, axes = plt.subplot_mosaic(
            plunger_mosaic, sharex=True, sharey=True, figsize=tools.cm2inch(two_column_width_cm, 8), rasterized=True)
        axes_left = [axes['P4']]
        axes_bottom = [axes['P8'], axes['P9'], axes['P10']]

    elif style == 'barrier':
        fig, axes = plt.subplot_mosaic(
            barrier_mosaic, sharex=True, sharey=True, figsize=tools.cm2inch(two_column_width_cm, 6.5), rasterized=True)
        axes_left = [axes['B1'], axes['B7']]
        axes_bottom = [axes['B7'], axes['B8'], axes['B9'],
                       axes['B10'], axes['B11'], axes['B12']]

    return fig, axes, axes_left, axes_bottom


# %% definitions

def create_nested_dict():
    return defaultdict(create_nested_dict)


def flarmor_to_gfactor_conversion(f_larmor, B_field):
    h = constants.Planck
    mu_B = constants.physical_constants['Bohr magneton'][0]

    g_factor = h*f_larmor / (mu_B*B_field)

    return g_factor

def gfactor_to_flarmor_conversion(g_factor, B_field):
    h = constants.Planck
    mu_B = constants.physical_constants['Bohr magneton'][0]

    f_larmor = g_factor*mu_B*B_field / h

    return f_larmor

def convert_string_to_array(string):
    # Evaluate the string to a Python list
    data_list = ast.literal_eval(string)

    # Convert the list to a NumPy array
    data_array = np.array(data_list)

    return data_array


def make_cmap(colors, cmap_name):
    n_bins = 500  # Number of bins in the colormap
    # Create the colormap
    custom_cmap = LinearSegmentedColormap.from_list(
        cmap_name, colors, N=n_bins)
    return custom_cmap


def cmap_wyr():
    colors = ["white", "#ffcc66", "#8b0000"]
    cmap_name = "white_yellow_darkred"
    return make_cmap(colors, cmap_name)



def cmap_yr():
    colors = ["#ffcc66", "#8b0000"]
    cmap_name = "yellow_darkred"
    return make_cmap(colors, cmap_name)

def cmap_seismic_y():
    seismic = plt.cm.get_cmap('seismic', 2)
    seismic_colors = seismic(np.linspace(0, 1, 2))
    colors = [seismic_colors[-1], "#ffdb92", seismic_colors[0]]
    cmap_name = "seismic_yellow"
    return make_cmap(colors, cmap_name)

# %% colorbar

def save_colorbar(norm, cmap, unit_label, filename, orientation='vertical', figsize=None,
                  ticklabelsize=None, labelsize=None, shrink=1, tick_number=3, location=None, labelpad=None):
    # Create a figure with the specified size
    if orientation == 'horizontal' and figsize is None:
        figsize = tools.cm2inch(4, 3)
    elif orientation == 'vertical' and figsize is None:
        figsize = tools.cm2inch(3, 4)

    fig, ax = plt.subplots(figsize=figsize)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    if orientation == 'horizontal':
        if location is None:
            location = 'bottom'
        cbar = fig.colorbar(sm, ax=ax,
                            orientation="horizontal", shrink=shrink, location=location)
    elif orientation == 'vertical':
        if location is None:
            location = 'left'
        cbar = fig.colorbar(sm, ax=ax,
                            orientation="vertical", shrink=shrink, location=location)


    # Customize the colorbar (optional)
    if labelsize is None:
        cbar.set_label(label=unit_label)
    else:
        cbar.set_label(label=unit_label, size=labelsize)
    if labelsize is not None:
        cbar.ax.tick_params(labelsize=ticklabelsize)  # Adjust the label size if needed
    cbar.ax.locator_params(nbins=tick_number)

    if labelpad is not None:
        cbar.ax.xaxis.labelpad = labelpad
        cbar.ax.yaxis.labelpad = labelpad

    # Remove axis for cleaner output
    ax.remove()

    # Save the colorbar as a PDF
    plt.tight_layout()
    plt.savefig(f'{filename}.pdf', bbox_inches='tight', pad_inches=0, transparent=True)
    plt.savefig(f'{filename}.png', bbox_inches='tight', pad_inches=0, transparent=True)
    plt.show()
    plt.close(fig)


# %% Function to add secondary axis to colorbar

def create_labels(xlabels, xunits):
    absolute_labels = []
    relative_labels = []

    for label, unit in zip(xlabels, xunits):
        if unit:
            absolute_label = f"{label} ({unit})"
        else:
            absolute_label = label
        relative_label = f"$\\Delta$ {label} (%)"

        absolute_labels.append(absolute_label)
        relative_labels.append(relative_label)

    return [absolute_labels, relative_labels]

def round_to_first_non_zero(values):
    values = np.asarray(values)
    non_zero_mask = values != 0
    magnitudes = np.floor(np.log10(np.abs(values[non_zero_mask])))
    scaled_values = values[non_zero_mask] / 10**magnitudes
    rounded_scaled_values = np.round(scaled_values, 1)
    rounded_values = rounded_scaled_values * 10**magnitudes
    output_values = np.zeros_like(values)
    output_values[non_zero_mask] = rounded_values

    return float(output_values)

def get_magnitude(value):
    if value == 0:
        return 0
    magnitude = int(np.floor(np.log10(abs(value))))
    return magnitude

def create_dual_axis_colorbar(data, label=['absolute value', r'$\Delta_{mean}$ (%)'] ,
                              cmap = None, show=False, orientation='horizontal'):
    # Compute the mean of the dataset
    mean_value = np.mean(data)
    std_value = np.std(data)
    min_value_abs = min(data)
    max_value_abs = max(data)
    std_value_rounded = np.round(std_value, -get_magnitude(std_value))
    min_value_abs_rounded = np.round(min_value_abs, -get_magnitude(std_value))
    max_value_abs_rounded = np.round(max_value_abs, -get_magnitude(std_value))
    ticklabels_abs = np.arange(min_value_abs_rounded, max_value_abs_rounded, std_value_rounded)

    # Calculate the percentage deviations from the mean
    deviations = ((data - mean_value) / mean_value) * 100

    max_value_dev = -max(np.abs(deviations))
    max_value = max(np.abs(deviations))

    # Create a colormap
    if cmap is None:
        cmap = cm.viridis
    norm = plt.Normalize(vmin=max_value_dev, vmax=max_value)

    # Create a figure and an axis for the colorbar
    if orientation == 'horizontal':
        fig, ax = plt.subplots(figsize=(2, 1))
        fig.subplots_adjust(bottom=0.5)
    elif orientation == 'vertical':
        fig, ax = plt.subplots(figsize=(1.2, 2))
        fig.subplots_adjust(left=0.5)

    # Create the colorbar
    cb1 = colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation=orientation)
    cb1.set_label(label[1])

    def percentage_ticks(x):
        deviation = ((x - mean_value) / mean_value) * 100
        return deviation

    # Add a secondary axis for percentage deviations
    def value_ticks(deviation):
        value = mean_value * (1 + deviation / 100)
        return value

    if orientation == 'horizontal':
        secax = ax.secondary_xaxis('top')
        secax.set_xlabel(label[0])
        secax.set_xticklabels(np.round(ticklabels_abs, -get_magnitude(std_value)))
    elif orientation == 'vertical':
        secax = ax.secondary_yaxis('left')
        secax.set_ylabel(label[0])
        secax.set_yticklabels(np.round(ticklabels_abs, -get_magnitude(std_value)))

    ticklabels_abs_dev_conversion = [percentage_ticks(x) for x in ticklabels_abs]
    secax.set_ticks(ticklabels_abs_dev_conversion)


    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig

# %%

def extract_rabi_driving_data(uuid, qubit, drive_gate, pc, drive_label, field_factor=0.69, kwargs={}):
    dat = load_by_uuid(uuid)
    b_field = field_factor*dat.snapshot['station']['instruments']['magnet']['parameters']['field']['value']
    A_rabi = dat.snapshot['measurement']['sequence'][pc][f'{drive_label}_pulses']['p0']['amplitude']
    f_larmor = dat.snapshot['measurement']['sequence'][pc][f'{drive_label}_pulses']['p0']['frequency']


    g_factor = flarmor_to_gfactor_conversion(f_larmor, b_field)


    x_data, x_label, x_unit = dat.m1_2.y(), dat.m1_2.y.label, dat.m1_2.y.unit
    y_data, y_label, y_unit = dat.m1_2.x(), dat.m1_2.x.label, dat.m1_2.x.unit

    y_data = 1e-9*y_data

    if 'rethreshold' in kwargs:
        z = dat.m1_3()
        if 'split' in kwargs:
            split = kwargs['split']
        else:
            split = 0.5
        z_data, _ = tools.thresholded_data(z, y_data, split = split, max_diff =10, plot = True, sensor = None)
    else:
        z_data = dat.m1_2()

    freq, fft_norm, peak_freq, peak_amplitude = tools.find_major_peak_frequency(
        y_data, z_data, height_threshold=0.001)

    plt.figure(figsize=(3,2))
    plt.plot(y_data, z_data)

    f_rabi_MHz = np.nan
    try:
        freq_guess = peak_freq[np.argmax(peak_amplitude)]
        if 'short_measurement' in kwargs:
            short_measurement = kwargs['short_measurement']
        else:
            short_measurement = False
        fit_params, fit_errors = tools.fit_Ramsey_data(y_data, z_data, freq_guess, short_measurement=short_measurement, tau_guess=10e-6, tau_max=200e-6)
        if fit_params is not None:
            A_fit, f_fit, phi_fit, tau_fit, C_fit = fit_params
            f_rabi_label = 'f_{Rabi}'
            plt.plot(y_data, tools.Ramsey(y_data, *fit_params), linestyle='--',
                      label=f'${f_rabi_label}={f_fit*1e-6:.3f}$ MHz',
                      lw=1, color='black')

            try:
                f_rabi = np.abs(fit_params[1])
            except:
                f_rabi = freq_guess
            f_rabi_MHz = f_rabi/1e6
    except:
        pass

    plt.xlabel('time (s)')
    plt.ylabel(x_label)
    f_larmor_label = '$f_{Larmor}$'
    plt.title(f'uuid: {uuid:_} \n {dat.sample_name} \n {qubit} ({drive_gate}) at {f_larmor_label} = {f_larmor/1e6:.1f} MHz, B = {b_field*1e3:.2f} mT')
    plt.legend()
    plt.tight_layout()
    plt.show()

    drive_efficiency_Bconst = f_rabi_MHz / (A_rabi*b_field)
    drive_efficiency_flarmorconst = f_rabi_MHz / (A_rabi*f_larmor*1e-9)

    new_data = {'uuid': uuid,
                'date': dat.completed_timestamp.date(),
                'sample':dat.sample_name,
                'qubit': qubit,
                'B_field': b_field,
                'drive_gate': drive_gate,
                'f_Larmor (MHz)': f_larmor*1e-6,
                'g': g_factor,
                'A_Rabi (mV)': A_rabi,
                'f_Rabi (MHz)': f_rabi_MHz,
                'drive_efficiency_Bconst (MHz/(mV⋅T))': drive_efficiency_Bconst,
                'drive_efficiency_flarmorconst (MHz/(mV GHz)': drive_efficiency_flarmorconst}

    df = pd.DataFrame(new_data, index=[0])
    df = df.set_index('uuid')

    pd.set_option('float_format', '{:.3f}'.format)
    print(df.T)
    return df.T

# %%

def apply_dark_background(fig, axes, label, cbar=None, dark_color='black'):
    # Set color bar properties for dark background
    if cbar is not None:
        cbar.ax.set_facecolor(None)
        cbar.ax.tick_params(colors='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        cbar.set_label(label, color='white')
        for spine in cbar.ax.spines.values():
            spine.set_edgecolor('white')

    # Set figure and axes properties for dark background
    fig.patch.set_facecolor(dark_color)
    if type(axes) == dict:
        loop_axes = axes.values()
    else:
        loop_axes = axes
    for ax in loop_axes:
        ax.set_facecolor(None)
        ax.tick_params(colors='white', which='both')
        for spine in ax.spines.values():
            spine.set_edgecolor('white')

def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

def cl():
    plt.close('all')

# %% general plotting functions

from matplotlib import colormaps
from matplotlib import colors

def ten_qubit_plot_343(qubit_results: list,
                       qubits = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10'],
                       cmap=colormaps['binary'], norm=None,
                       dark_background=False, accent_colour = 'black',
                       fig=None, ax=None, figsize=tools.cm2inch((5.5, 3)),
                       plot_cmap=True, cmap_label='', cmap_ticks=None, cmap_orientation='vertical', cmap_shrink=0.8,
                       annotate_qubits=None
                       ):
    df = pd.DataFrame(qubit_results, qubits)

    qubit_positions = [(-2, 1), (0, 1), (2, 1),
                       (-3, 0), (-1, 0), (1, 0), (3, 0),
                       (-2, -1), (0, -1), (2, -1)]

    if fig is None and ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    # plot qubit connectivity
    qubit_pairs = [
        (1, 4), (4, 8), (1, 5), (5, 8), (5, 9),
        (2, 5), (2, 6), (0, 3), (3, 7), (4, 7),
        (0, 4), (6, 9)
    ]
    for qubit_pair in qubit_pairs:
        plt.plot([qubit_positions[qubit_pair[0]][0], qubit_positions[qubit_pair[1]][0]],
                 [qubit_positions[qubit_pair[0]][1], qubit_positions[qubit_pair[1]][1]],
                 color=accent_colour, zorder=0)

    annotate_occupancy = False
    if annotate_qubits is not None:
        if len(annotate_qubits) == 10:
            annotate_occupancy = True
        else:
            print('The length of the annotate_qubits list should be 10')

    if norm is None:
        min_value = min(qubit_results)
        max_value = max(qubit_results)
        norm = colors.Normalize(vmin=min_value, vmax=max_value, clip=True)

    # Iterate through the dictionary to plot each polygon and annotate
    for n in range(0, 10):
        qubit = f'Q{n+1}'
        if qubit in qubits:
            value = df.loc[qubit]
            color = cmap(norm(value))
            lw = 1
        else:
            color = 'white'
            lw = 0.1

        r = 0.5
        circle = plt.Circle(qubit_positions[n], r, color=color, ec='black', lw=lw, zorder=1)
        if annotate_occupancy:
            plt.annotate(annotate_qubits[n], (qubit_positions[n][0] - 0.1, qubit_positions[n][1] - 0.1),
                         color='w')
        ax.add_artist(circle)

    # Create a colorbar
    if plot_cmap:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label=cmap_label,
                            orientation=cmap_orientation, shrink=cmap_shrink)
        if cmap_ticks is not None:
            cbar.set_ticks(cmap_ticks)
    ax.axis('off')

    ax.set_xlim(-3 - 2 * r, 3 + 2 * r)
    ax.set_ylim(-1 - 2 * r, 1 + 2 * r)

    ax.set_aspect('equal')

    if dark_background:
        cbar.ax.set_facecolor(None)  # Set color bar background to black
        cbar.ax.tick_params(colors='white')  # Set color of ticks on color bar to white
        cbar.ax.yaxis.set_tick_params(color='white')  # Set color of the ticks
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')  # Set color of the tick labels
        cbar.set_label(cmap_label, color='white')  # Set label color

        fig.patch.set_facecolor(None)
        ax.set_facecolor(None)
        ax.tick_params(colors='white', which='both')
        # Customize the spines to be visible
        for spine in ax.spines.values():
            spine.set_edgecolor('white')

    return fig, ax