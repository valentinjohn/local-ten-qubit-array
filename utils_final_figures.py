from pathlib import Path
from config import DATA_DIR

data_path = DATA_DIR
from utils.utils import *

# %%
import utils.analysis_tools as tools
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib import colors
from natsort import natsorted

from gfactor_tunability.utils_statistics_gfactor_tunability import load_gfactor_tunability_data, plot_LSES_individual

import json
from config import DATA_DIR, PROJECT_DIR

from mpl_toolkits.axes_grid1 import make_axes_locatable
from drive_efficiency.utils_statistics_driving_slope import fit_freq_2D, extract_slope
from gfactor_tunability.utils_statistics_gfactor_tunability import get_flarmor_2D, get_flarmor_slope
from drive_locality import utils_nearest_neighbours as unn

data_path_uuid_datasets = DATA_DIR / "uuid_datasets"

warnings.filterwarnings('ignore')

# %% load function

import h5py
import xarray as xr

def get_fname(uuid):
    return f'ds_{uuid}.hdf5'

def read_numpy_from_hdf5(fname: str, key: str = "data") -> np.ndarray:
    if not os.path.exists(fname):
        raise FileNotFoundError(fname)
    with h5py.File(fname, "r") as f:
        return f[key][()]

def load_xr_hdf5(fname: str) -> xr.Dataset:
    # Check existence. xarray gives a very confusing error when the file does not exist.
    if not os.path.exists(fname):
        raise FileNotFoundError(fname)
    xds = xr.load_dataset(fname)
    return xds

def find_m1_fraction(data_list, N):
    target = f"m1_{N}_fraction"
    return target if target in data_list else None

def get_data(uuid: int) -> xr.Dataset:
    fname = get_fname(uuid)
    uuid_data_path = DATA_DIR / 'uuid_datasets' / fname
    dat = load_xr_hdf5(uuid_data_path)

    for n in range(1, 5):
        n_fraction = find_m1_fraction(dat.attrs['keywords'], n)
        if n_fraction is not None:
            z = dat[n_fraction].data
            xdata, ydata = dat[n_fraction].indexes.values()
            x, xlabel = xdata.values, xdata.name
            y, ylabel = ydata.values, ydata.name

            return (x, y, z, xlabel, ylabel)


# %%

uuid = 1706887485905283691
x, y, z, xlabel, ylabel = get_data(uuid)

# %% general

def make_colorbar(norm, cmap, unit_label, orientation='vertical', figsize=None,
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
    plt.show()

# %% Figure 1

def plot_infidelities(save_path=None):
    qubit_positions = [(-2, 1), (0, 1), (2, 1),
                       (-3, 0), (-1, 0), (1, 0), (3, 0),
                       (-2, -1), (0, -1), (2, -1)]

    results = pd.read_csv(os.path.join(data_path, 'final_result_summary.csv'))
    results = results.set_index('label')

    num_colors = 7
    min_value = 0
    max_value = 0.6

    c = 'binary'
    continuous_cmap = colormaps[f'{c}']
    colors_list = continuous_cmap(np.linspace(0, 1, num_colors))
    cmap = colors.ListedColormap(colors_list)

    fig, ax = plt.subplots(1, 1, figsize=tools.cm2inch((4.5, 4)))

    for qubit_pairs in [(1, 4), (4, 8), (1, 5), (5, 8), (5, 9), (2, 5), (2, 6), (0, 3), (3, 7), (4, 7), (0, 4), (6, 9)]:
        plt.plot([qubit_positions[qubit_pairs[0]][0], qubit_positions[qubit_pairs[1]][0]],
                 [qubit_positions[qubit_pairs[0]][1], qubit_positions[qubit_pairs[1]][1]],
                 color='black', zorder=0)

    label = 'Fidelity (%)'
    xlabel = r'Infidelity (%)'

    results_fidelity = results.loc[label]

    norm = Normalize(vmin=min_value, vmax=max_value, clip=True)

    # Iterate through the dictionary to plot each polygon and annotate
    for n in range(1, 11):

        value = results_fidelity[f'Q{n}']
        value = 100 - value
        color = cmap(norm(value))

        r = 0.5
        circle = plt.Circle(qubit_positions[n - 1], r, color=color, ec='black')
        ax.add_artist(circle)

    # Create a colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label=xlabel, ticks=[0, 0.25, 0.5],
                        orientation="horizontal", shrink=0.5, pad=0.05)
    ax.axis('off')

    ax.set_xlim(-3 - 2 * r, 3 + 2 * r)
    ax.set_ylim(-1 - 2 * r, 1 + 2 * r)

    ax.set_aspect('equal')

    cbar.ax.xaxis.set_ticks([0, 0.3, 0.6])

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(os.path.join(save_path, f'qubit_fidelities_{c}.png'), dpi=300)
        fig.savefig(os.path.join(save_path, f'qubit_fidelities_{c}.pdf'), dpi=300, transparent=True)

    plt.show()

# %% Figure 2a

# Load the nested dictionary from the JSON file
data_path_edsr_uuids = PROJECT_DIR / 'drive_efficiency/uuids_driving_efficiency.json'
with open(data_path_edsr_uuids, 'r') as file:
    uuids_edsr_uuids = json.load(file)

data_path_gfactor_tunability = PROJECT_DIR / 'gfactor_tunability/uuids_gfactor_tunability.json'
with open(data_path_gfactor_tunability, 'r') as file:
    uuids_gfactor_tunability = json.load(file)

def plot_raw_data(qubit, num_holes, gate, properties=['LSES', 'drive_efficiency'], save_path=None):
    match gate[0]:
        case 'P':
            gate_type = 'plunger'
        case 'B':
            gate_type = 'barrier'
    n = int(gate[1:])

    uuids = {'LSES': uuids_gfactor_tunability,
             'drive_efficiency': uuids_edsr_uuids}

    for property in properties:
        figsize = tools.cm2inch(one_column_width_cm / 2.7, 3.9)
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        plt.locator_params(axis='x', nbins=3)

        uuid = uuids[property][qubit][num_holes][gate_type][n - 1]

        y, x, z, y_label, x_label = get_data(uuid)
        x_raw = x
        y_raw = y
        print(f'Max value of z: {np.nanmax(z)}')
        print(f'Min value of z: {np.nanmin(z)}')

        if property == 'LSES':
            x = (x - np.average(x)) / 1e6
            xlabel = r'$\Delta f_{\rm{mw}}$ (MHz)'
            ylabel = r'$\Delta \rm{P5}$ (mV)'
        elif property == 'drive_efficiency':
            x = x / 1e3
            ylabel = r'$A_{\rm{P5,mw}}$ (mV)'
            xlabel = r'$t_{\rm{P5,mw}}$ ($\rm{\mu}$s)'

        if property == 'LSES':
            z_transpose = z.T
            im = ax.pcolor(y, x, z_transpose, rasterized=True)

            larmor_freqs = get_flarmor_2D(z, x, uuid, 'peak')
            y_nonan = y[~np.isnan(larmor_freqs)]
            fit_larmor_freq_s = np.array(larmor_freqs)[~np.isnan(larmor_freqs)]
            slope, fit_param = get_flarmor_slope(fit_larmor_freq_s, y_nonan)

            ax.plot(y, Line(y, *fit_param), color='white',
                    label=f'{slope * 1e3:.2f} MHz/mV',
                    ls='--', lw=0.8)
            ax.legend(labelcolor='white')
            ax.set_xlabel(ylabel)
            ax.set_ylabel(xlabel)

            ax.legend(loc='upper center', frameon=True, bbox_to_anchor=(0.5, 1.3),
                      facecolor='black', framealpha=1, edgecolor='black', labelcolor='white')
        else:
            im = ax.pcolor(x, y, z, rasterized=True)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

        divider = make_axes_locatable(ax)
        cax = divider.new_vertical(size='10%', pad=0.2)
        fig.add_axes(cax)
        fig.colorbar(im, cax=cax, orientation='horizontal', location='top', label=r'$\rm{P}_{\uparrow}$')

        plt.subplots_adjust(left=0.33,
                            bottom=0.25,
                            right=0.83,
                            top=0.78)
        if save_path is not None:
            # fig.savefig(os.path.join(save_path, f'raw_data_{qubit}_{num_holes}_{gate}_{property}.png'), dpi=300)
            fig.savefig(os.path.join(save_path, f'raw_data_{qubit}_{num_holes}_{gate}_{property}.pdf'), dpi=300,
                        transparent=True)
        plt.show()

        if property == 'drive_efficiency':
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            z_fft = []
            for i in range(len(y)):
                zfft = np.fft.fft(z[i])
                z_fft.append(zfft)

            z_fft = np.array(z_fft)
            timestep = x[1] - x[0]
            freq = np.fft.fftfreq(x.size, d=timestep) * 1e3
            freq_fft = freq[1:round(len(freq) / 2)]
            fft_norm = np.abs(z_fft)[:, 1:round(len(freq) / 2)] / np.abs(zfft).max()

            fit_osc_freq, freq_fft, fft_norm = fit_freq_2D(x_raw, y_raw, z, plot_ramsey=False)
            slope, off_set, m_freq = extract_slope(fit_osc_freq, y)

            im = ax.pcolor(y, freq_fft, np.transpose(fft_norm), shading='auto', zorder=0, rasterized=True)
            ax.plot(y, (slope * y + off_set), label=f'{slope:.2f} MHz/mV', color='white', ls='--', lw=0.5, zorder=1)
            ax.legend(loc='upper center', frameon=True, bbox_to_anchor=(0.5, 1.3),
                      facecolor='black', framealpha=1, edgecolor='black', labelcolor='white')

            xlabel = r'$A_{\rm{P5,mw}}$ (mV)'
            ylabel = r'$f_{\rm{P5,mw}}$ (MHz)'
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

            # add color bar below chart
            divider = make_axes_locatable(ax)
            cax = divider.new_vertical(size='10%', pad=0.2)
            fig.add_axes(cax)
            fig.colorbar(im, cax=cax, orientation='horizontal', location='top', label='fft amplitude')

            plt.subplots_adjust(left=0.33,
                                bottom=0.25,
                                right=0.83,
                                top=0.78)

            if save_path is not None:
                # fig.savefig(os.path.join(save_path, f'raw_data_fft_{qubit}_{num_holes}_{gate}_{property}.png'), dpi=300)
                fig.savefig(os.path.join(save_path, f'raw_data_fft_{qubit}_{num_holes}_{gate}_{property}.pdf'), dpi=300,
                            transparent=True)
            plt.show()

# %% Figure 2b

def plot_boxplot_ratios(save_path=None):
    # %%% get Rabi
    df = {}
    df['1h'] = pd.read_csv(os.path.join(DATA_DIR, 'driving_strength_1hole.csv'),
                           header=[0], index_col=[0])
    df['3h'] = pd.read_csv(os.path.join(DATA_DIR, 'driving_strength_3hole.csv'),
                           header=[0], index_col=[0])
    qubits = df['1h'].index

    smallest_detectable_drive_efficiency = 0.008
    df['1h']['P3'].loc['Q3'] = smallest_detectable_drive_efficiency / 10

    mask_remove_zeros = (np.diag(df['1h']) != 0)
    df_rabi_difference_factor = np.diag(df['3h'])[mask_remove_zeros] / np.diag(df['1h'])[mask_remove_zeros]
    df_rabi_difference_factor = df_rabi_difference_factor[~np.isnan(df_rabi_difference_factor)]

    df_gfactors = pd.read_csv(os.path.join(DATA_DIR, 'gfactor_vs_occupation.csv'),
                              header=[0], index_col=[0])
    df_gfactors_difference_factor = df_gfactors.loc['3h'] / df_gfactors.loc['1h']
    df_gfactors_difference_factor = df_gfactors_difference_factor.dropna()
    df_gfactors_difference_factor = df_gfactors_difference_factor.to_numpy()

    df_p2 = pd.read_csv(os.path.join(DATA_DIR, 'gfactor_susceptibility_p2.csv'))
    df_p2 = df_p2.set_index('Unnamed: 0')
    df_p2 = df_p2.reindex(index=natsorted(df_p2.index))
    df_p2_difference_factor = df_p2['3h'] / df_p2['1h']
    df_p2_difference_factor = df_p2_difference_factor[~np.isnan(df_p2_difference_factor)]
    df_p2_difference_factor = df_p2_difference_factor.to_numpy()

    df_rabi_difference_factor_qf = np.diag(df['3h'])[mask_remove_zeros] / np.diag(df['1h'])[mask_remove_zeros]
    df_p2_difference_factor_qf = df_p2['3h'] / df_p2['1h']

    quality_factor = df_rabi_difference_factor_qf / df_p2_difference_factor_qf
    quality_factor = quality_factor.dropna().to_numpy()

    file_name = 'boxplot_3h_vs_1h'

    data = [list(df_gfactors_difference_factor),
            list(df_p2_difference_factor),
            list(df_rabi_difference_factor),
            list(quality_factor)
            ]

    averages = [np.mean(data[0]), np.mean(data[1]), np.mean(data[2]), np.mean(data[3])]
    medians = [np.median(data[0]), np.median(data[1]), np.median(data[2]), np.median(data[3])]

    labels = [r'$\frac{g^*_{\rm{3h}}}{g^*_{\rm{1h}}}$',
              r'$\frac{\xi_{3h}}{\xi_{1h}}$',
              r"$\frac{\rm f_{R,3h}}{f_{R,1h}}$",
              r'$\frac{Q_{\rm{3h,P}}}{Q_{\rm{1h,P}}}$',
              ]

    fig, ax = plt.subplots(1, 1, figsize=tools.cm2inch(two_column_width_cm / 2.5, 4))

    ax.boxplot(data, whis=1000, vert=True)
    ax.hlines(1, 0.5, 0.5 + len(data), color='black', ls='--', lw=1)
    ax.set_ylim(0.04, 25)

    ax.set_ylabel('ratio')
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yscale('log')

    plt.tight_layout()

    if save_path is not None:
        # fig.savefig(os.path.join(save_path, f"{file_name}.png"), dpi=300, transparent=True)
        fig.savefig(os.path.join(save_path, f"{file_name}.pdf"), dpi=300, transparent=True)

    plt.show()

    return averages, medians

# %% Figure 2c

def plot_gfactor_vs_occupation(save_path=None, hole_occupation=['1h', '3h', '5h'], title=False):
    df_gfactors = pd.read_csv(os.path.join(data_path, 'gfactor_vs_occupation.csv'))
    df_gfactors.set_index('Unnamed: 0', inplace=True)

    min_value = 0.5
    max_value = 0.65

    norm = Normalize(vmin=min_value, vmax=max_value, clip=True)
    cmap = mpl.colormaps['cividis']

    for num_holes in hole_occupation:
        fig, ax = ten_qubit_plot_343(df_gfactors.loc[num_holes].dropna(),
                                     qubits=df_gfactors.loc[num_holes].dropna().index,
                                     cmap=cmap,
                                     norm=norm,
                                     plot_cmap=False)
        if title:
            fig.suptitle(f'g-factors ({num_holes})')

        if save_path is not None:
            file_name = f"gfactors_{num_holes}"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            # fig.savefig(os.path.join(save_path, f"{file_name}.png"), dpi=300, transparent=True)
            fig.savefig(os.path.join(save_path, f"{file_name}.pdf"), dpi=300, transparent=True)

        plt.show()

        make_colorbar(norm, cmap, r'$g^*$',
                      orientation='horizontal', figsize=tools.cm2inch(3.5, 2.5), tick_number=4)


# %% Figure 2d

def load_LSES_data():
    b_field = 41.4e-3
    n_magnitude = 4
    unit = rf'10$^{{-{n_magnitude}}}$/mV'

    df = load_gfactor_tunability_data(DATA_DIR)

    for n in range(0, 11):
        qubit = f'Q{n}'
        df_gfactor = {}
        df_gfactor['1h'] = flarmor_to_gfactor_conversion(10 ** n_magnitude * df['1h'] * 1e3, b_field)
        df_gfactor['3h'] = flarmor_to_gfactor_conversion(10 ** n_magnitude * df['3h'] * 1e3, b_field)
        df_gfactor['5h'] = flarmor_to_gfactor_conversion(10 ** n_magnitude * df['5h'] * 1e3, b_field)

    return df_gfactor, unit

# %% Figure 3

df_drive_eff, drive_eff = unn.get_locality_data()

def plot_locality_boxplots(num_holes=['1h', '3h'], gate_types=['plunger', 'barrier'],
                           title=True, save_path=None):
    cmap = cmap_wyr()

    for hole in num_holes:
        for gate_type in gate_types:
            values_list = []
            mean_value_list = []
            mean_value_norm_list = []
            for step in df_drive_eff[hole][gate_type].keys():
                values = df_drive_eff[hole][gate_type][step].stack()
                values_list.append(values.tolist())
                mean_value_list.append(np.mean(values))

            mean_value_norm_list = np.array(mean_value_list) / max(mean_value_list)

            if title:
                height = 3
            else:
                height = 2.5
            fig = plt.figure(figsize=tools.cm2inch(3.5, height))

            # Create the boxplot
            boxplot = plt.boxplot(values_list, whis=1000, vert=True, patch_artist=True, showfliers=False,
                                  showmeans=True, meanline=True)

            colors = cmap(mean_value_norm_list)

            for patch, color in zip(boxplot['boxes'], colors):
                patch.set_facecolor(color)

            plt.xlabel('n-th nearest qubits')
            plt.ylabel(r'$f_{R}/A$ (MHz/mV)')

            if title:
                plt.title(rf'Rabi Eff. ({gate_type},{hole})')

            plt.tight_layout()

            if save_path is not None:
                file_name = f'locality_boxplot_{gate_type}_{hole}ole'
                # fig.savefig(os.path.join(save_path, f"{file_name}.png"), dpi=300, transparent=True)
                fig.savefig(os.path.join(save_path, f"{file_name}.pdf"), dpi=300, transparent=True)

            plt.show()

def plot_locality_array(num_holes=['1h', '3h'], gate_types=['plunger', 'barrier'], title=True, save_path=None):
    for hole in num_holes:
        for gate_type in gate_types:

            fig, ax = plt.subplots()

            max_mean_key = max(drive_eff[hole][gate_type], key=lambda k: drive_eff[hole][gate_type][k]['mean'])
            max_mean_value = drive_eff[hole][gate_type][max_mean_key]['mean']

            vmax = 1  # max_mean_value
            norm = Normalize(vmin=0, vmax=vmax, clip=True)
            accent_color = 'black'
            cmap = unn.cmap_wyr()

            for step, polygons in unn.qubit_drive_polygons[gate_type].items():
                for polygon in polygons:
                    if polygon[0][0] == unn.plunger_polygon[0][0] and polygon[0][1] == unn.plunger_polygon[0][1]:
                        color = 'red'
                        lw = 0.8
                    else:
                        color = 'grey'
                        lw = 0.2
                    poly = Polygon(polygon)
                    patch = PolygonPatch(poly, fc='white', ec=color, lw=lw)
                    ax.add_patch(patch)

            for orientation, positions in unn.barrier_coordinates[f'{gate_type}_drive'].items():
                for position in positions:
                    polygon = unn.barrier_polygon[orientation] + np.array(position)
                    if position[0] == 0 and position[1] == 0:
                        color = 'red'
                        lw = 0.8
                    else:
                        color = 'grey'
                        lw = 0.2
                    poly = Polygon(polygon)
                    patch = PolygonPatch(poly, fc='white', ec=color, lw=lw)
                    ax.add_patch(patch)

            for step, cords in unn.qubit_coordinates_drive[gate_type].items():
                value = drive_eff[hole][gate_type][step]['mean'] / max_mean_value
                for coordinates in cords:
                    color = cmap(norm(value))
                    qubit_position = unn.qubit_coordinates_drive[gate_type][step]
                    circle = plt.Circle(coordinates, 0.04, color=color, ls=None)
                    ax.add_artist(circle)

            plt.xlim(-0.6, 0.6)
            plt.ylim(-0.6, 0.6)

            plt.axis('scaled')
            plt.axis('off')

            if title:
                plt.title(f'Drive efficiency {gate_type} {hole}ole')

            plt.tight_layout()

            if save_path is not None:
                # plt.savefig(save_path / f'drive_efficiency_{gate_type}_{hole}ole.png')
                plt.savefig(save_path / f'drive_efficiency_{gate_type}_{hole}ole.pdf', transparent=True)

            plt.show()

    if save_path is not None:
        norm = Normalize(vmin=0, vmax=vmax, clip=True)
        cmap = unn.cmap_wyr()
        save_colorbar(norm, cmap, r'$f_{\rm{R}}/f_{\rm{R}}^{\rm{target}}$', os.path.join(save_path, 'cbar_locality_norm.pdf'),
                      orientation='vertical', figsize=tools.cm2inch(3.5, 2.5), location='right')