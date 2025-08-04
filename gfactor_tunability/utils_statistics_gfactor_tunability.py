# %% imports
import matplotlib.pyplot as plt

from utils import *
from utils.utils import apply_dark_background

# %% Path management

from pathlib import Path
from config import DATA_DIR

script_dir = Path(__file__).resolve().parent
fig_path = script_dir / "images"
subfolder = fig_path / "gfactor_tunability_extraction"

# %% core tools setup to load data

from core_tools.data.ds.ds_hdf5 import load_hdf5_uuid
uuids_data_path = DATA_DIR / 'uuid_datasets'

# %% load data

def load_gfactor_tunability_data(data_path):
    df = {}
    df['1h'] = pd.read_csv(os.path.join(data_path, 'gfactor_tunability_1hole.csv'),
                           header=[0], index_col=[0])
    df['3h'] = pd.read_csv(os.path.join(data_path, 'gfactor_tunability_3hole.csv'),
                           header=[0], index_col=[0])
    df['5h'] = pd.read_csv(os.path.join(data_path, 'gfactor_tunability_5hole.csv'),
                           header=[0], index_col=[0])

    df['1h'] = df['1h'].dropna()
    df['3h'] = df['3h'].dropna()
    df['5h'] = df['5h'].dropna()
    return df

# %%

def peak_or_dip(uuids):
    dat = load_hdf5_uuid(uuids[0], uuids_data_path)
    z_sum_all = np.zeros((1, dat.m1_2.z.shape[1]))

    for uuid in uuids:
        dat = load_hdf5_uuid(uuid, uuids_data_path)
        z_sum_all = z_sum_all + dat.m1_2().sum(axis=0)

    if (z_sum_all.max() - z_sum_all.mean()) > (z_sum_all.mean() - z_sum_all.min()):
        res = 'peak'
    else:
        res = 'dip'

    return res

def get_flarmor_2D(z, x, uuid, search_for='peak', uuid_limits={}):
    larmor_freqs = []
    for i in np.arange(0, len(z)):
        x_i = x[~np.isnan(z[i])]
        z_i = z[i][~np.isnan(z[i])]
        if search_for == 'peak':
            A_guess = max(z_i) - min(z_i)
            x_guess = x[np.argmax(z[i])]
        else:
            A_guess = min(z_i) - max(z_i)
            x_guess = x[np.argmin(z[i])]
        # print(A_guess)
        fwhm_gues = 1.50912654e+06
        y_guess = np.mean(z[i][0:10])
        fit_param = fit_data(
            x_i, z_i, p0=[A_guess, fwhm_gues, x_guess, y_guess], func=Gauss, plot=False)
        # print(fit_param)

        if uuid in uuid_limits.keys():
            if np.abs(fit_param[0]) < uuid_limits[uuid][0] or fit_param[1] > uuid_limits[uuid][1] or fit_param[1] < \
                    uuid_limits[uuid][2]:
                larmor_freq = float('nan')
            else:
                larmor_freq = fit_param[2]
        else:
            if np.abs(fit_param[0]) < 0.05 or fit_param[1] > 5e6 or fit_param[1] < 0.5e6:
                larmor_freq = float('nan')
            else:
                larmor_freq = fit_param[2]
        larmor_freqs.append(larmor_freq)
        # print(f'{gate}: {i} - {fit_param}')

    return larmor_freqs

def get_flarmor_slope(fit_larmor_freq_s, y_nonan):
    # try:
    if len(fit_larmor_freq_s) == 0 or len(y_nonan) == 0:
        fit_param = None
        slope = np.nan
    else:
        fit_param = fit_data(y_nonan, fit_larmor_freq_s, p0=[(fit_larmor_freq_s[-1] - fit_larmor_freq_s[0]) / (
                y_nonan[-1] - y_nonan[0]), fit_larmor_freq_s[0]], func=Line, plot=False)
        slope = fit_param[0] / 1000

    return slope, fit_param

# %% Fitting functions


def gfactor_tunability_plot(uuids, gates, qubit, num_holes, plot=True, uuid_limits={}):

    if gates[0][0] in ['P', 'N'] or gates[0][:2] in ['vP']:
        subplot_plungers = True
        subplot_barriers = False
    if gates[0][0] in ['J', 'B'] or gates[0][:2] in ['vB']:
        subplot_barriers = True
        subplot_plungers = False

    gate_name_length = len(gates[0])-1

    if plot:
        if gates[0][0] in ['P'] or gates[0][:2] in ['vP']:
            style = 'plunger'
            fig, axes, axes_left, axes_bottom = create_343_subplot(style=style, num=1)
        elif gates[0][0] in ['J', 'B'] or gates[0][:2] in ['vB']:
            style = 'barrier'
            fig, axes, axes_left, axes_bottom = create_343_subplot(style=style, num=1)
        else:
            fig, axes = plt.subplots(
                2, ceil(len(uuids)/2), sharex=True, figsize=(12, 5.5))
            axes_left = axes[:, 0]
            axes_bottom = axes[-1, :]
            axes = axes.flatten()

    slopes = []
    search_for = peak_or_dip(uuids)
    for n, (gate, uuid) in enumerate(zip(gates, uuids)):
        dat = load_hdf5_uuid(uuid, uuids_data_path)

        x, x_label, x_unit = dat.m1_2.y(), dat.m1_2.y.label, dat.m1_2.y.unit
        y, y_label, y_unit = dat.m1_2.x(), dat.m1_2.x.label, dat.m1_2.x.unit
        z = dat.m1_2()

        larmor_freqs = get_flarmor_2D(z, x, uuid, search_for, uuid_limits)

        y_nonan = y[~np.isnan(larmor_freqs)]
        fit_larmor_freq_s = np.array(larmor_freqs)[~np.isnan(larmor_freqs)]
        slope, fit_param = get_flarmor_slope(fit_larmor_freq_s, y_nonan)
        slopes.append(slope)

        if plot:
            cmap = 'viridis'
            norm = Normalize(vmin=0,
                             vmax=1, clip=True)

            if subplot_plungers:
                ax = axes[f'P{gate[gate_name_length:]}']
            elif subplot_barriers:
                ax = axes[f'B{gate[gate_name_length:]}']
            else:
                ax = axes[n]

            title_name = f'{gate}' # \n id: {dat.exp_id}'
            ax.set_title(title_name)
            ax.pcolor(y, x*1e-6, np.transpose(z), shading='auto', cmap=cmap, norm=norm)
            ax.scatter(y_nonan, fit_larmor_freq_s*1e-6, color='white', s=5)
            if fit_param is not None:
                ax.plot(y, Line(y, *fit_param)*1e-6, color='white',
                    ls='--', label=f'{slope*1e-3:.2f} MHz/mV')
            ax.legend(labelcolor='white')

            if ax in axes_left:
                ax.set_ylabel(f'f (MHz)')
            if ax in axes_bottom:
                ax.set_xlabel(f'{y_label} ({y_unit})')

        match style:
            case 'plunger':
                plt.subplots_adjust(left=0.1,
                                    bottom=0.12,
                                    right=0.9,
                                    top=0.85,
                                    wspace=0.1,
                                    hspace=0.1)
            case 'barrier':
                plt.subplots_adjust(left=0.1,
                                    bottom=0.15,
                                    right=0.9,
                                    top=0.85,
                                    wspace=0.3,
                                    hspace=0.4)

    file_name = f'gfactor_tunability_extraction_{qubit}_{num_holes}_{style}'
    fig.suptitle(f'{qubit} ({num_holes})', size=12, weight='bold')
    fig.savefig(os.path.join(subfolder, f"{file_name}.png"), dpi=300, transparent=True)
    fig.savefig(os.path.join(subfolder, f"{file_name}.pdf"), dpi=300, transparent=True)

    plt.show()

    return slopes


def plot_in_343_config_gfactor(qubit, plunger, slope_plunger, slope_barrier, min_plus=True):
    gates = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10',
             'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11', 'B12']

    x_coordinate = [3, 7, 11, 1, 5, 9, 13, 3, 7,
                    11, 2, 4, 6, 8, 10, 12, 2, 4, 6, 8, 10, 12]
    y_coordinate = [5, 5, 5,  3, 3, 3, 3,  1, 1,
                    1,  4, 4, 4, 4, 4,  4,  2, 2, 2, 2, 2,  2]
    size = [100]*10+[40]*12

    slopes_Q1 = np.array(slope_plunger + slope_barrier)

    if min_plus:
        norm = norm = TwoSlopeNorm(
            vmin=slopes_Q1.min(), vcenter=0, vmax=slopes_Q1.max())
        line_color = ['k' if x != plunger else 'green' for x in gates]
    else:
        slopes_Q1 = abs(slopes_Q1)
        line_color = ['k' if x != plunger else 'red' for x in gates]

    plt.figure()
    plt.title('Larmor freq susceptibility ' + qubit, color='green')
    if min_plus:
        plt.scatter(x_coordinate, y_coordinate, c=slopes_Q1,
                    s=size, ec=line_color, cmap='seismic', norm=norm)
    else:
        plt.scatter(x_coordinate, y_coordinate,
                    c=slopes_Q1, s=size, ec=line_color)
    plt.xlim(0, 16)
    plt.ylim(-1, 6)
    plt.axis(False)
    plt.colorbar(label='kHz/mV',
                 ticks=[min(slopes_Q1), 0, max(slopes_Q1)], shrink=0.7)
    plt.tight_layout()

# %% final plotting functions

def get_gfactors_from_flarmor_data(df, B_field, flarmor_unit='kHz', unit_exponent=0):
    unit_conversion = {'Hz': 1,
                       'kHz': 1e3,
                       'MHz': 1e6,
                       'GHz': 1e9,
                       }
    slopes_gates_larmor_input = df.to_numpy()
    slopes_gates_larmor_Hz = slopes_gates_larmor_input*unit_conversion[flarmor_unit]

    factor = 10**unit_exponent
    # exponent = int(math.log10(1/factor))

    if unit_exponent == 0:
        gfactor_unit = ''
    else:
        gfactor_unit = r'$10^{{{}}}$'.format(-unit_exponent)

    slopes_gates_gfactor = flarmor_to_gfactor_conversion(factor*slopes_gates_larmor_Hz, B_field)

    return slopes_gates_gfactor, gfactor_unit

# %%% horseshoe plots

def overview_plot_gfactor_tunability(slope_gates, unit='1/mV',
                                     save=True, show_plot=True,
                                     dark_background=False, dark_color='black',
                                     transparent_png=False):
    plungers = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10']
    qubits = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10']

    qubit_mosaic = [['.', 'Q1', '.', 'Q2', '.', 'Q3', '.'],
                    ['Q4', '.', 'Q5', '.', 'Q6', '.', 'Q7'],
                    ['.', 'Q8', '.', 'Q9', '.', 'Q10', '.']]

    gates = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10',
             'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11', 'B12']

    gate_positions = pd.read_csv(DATA_DIR / 'array_343_gate_positions.csv')
    gate_positions = gate_positions.set_index('Key')

    largest_val = np.max(np.abs(slope_gates))
    min_val = -largest_val
    max_val = largest_val

    cmap = plt.cm.get_cmap('seismic')
    norm = TwoSlopeNorm(vmin=min_val, vcenter=0, vmax=max_val)

    if dark_background:
        accent_color = 'white'
    else:
        accent_color = 'black'

    fig, axes = plt.subplot_mosaic(
        qubit_mosaic, sharex=True, sharey=True, figsize=tools.cm2inch(18, 8))

    for n, qubit in enumerate(qubits):
        try:
            ax = axes[qubit]
        except:
            ax = axes[n]

        slopes_Qn = slope_gates[n]
        qubit_plunger = plungers[n]

        ax.set_title(qubits[n], color=accent_color)
        for gate, slope in zip(gates, slopes_Qn):
            polygon = Polygon(np.array(gate_positions.loc[f'{gate}']))
            color = cmap(norm(slope))
            patch = PolygonPatch(polygon, fc=color, ec='black', lw=0.1)
            ax.add_patch(patch)

        ax.set_xlim(gate_positions['X'].min(), gate_positions['X'].max())
        ax.set_ylim(gate_positions['Y'].min(), gate_positions['Y'].max())
        ax.set_axis_off()
        ax.set_aspect('equal')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axes['Q9'], label=unit,
                 orientation="horizontal", shrink=1)

    plt.subplots_adjust(left=0,
                        bottom=0.15,
                        right=1,
                        top=0.9,
                        wspace=-0.3,
                        hspace=-0.3)

    file_name = 'gfactor_tunability_horseshoe'
    if dark_background:
        apply_dark_background(fig, cbar, axes, unit, dark_color)
        file_name = file_name+'_dark'

    if save:
        fig.savefig(os.path.join(fig_path, f"{file_name}.png"), dpi=300, transparent=transparent_png)
        fig.savefig(os.path.join(fig_path, f"{file_name}.pdf"), dpi=300, transparent=True)

    if show_plot:
        plt.show()
    else:
        plt.close()


def overview_plot_gfactor_tunability_indv(slope_gates, unit='1/mV',
                                         save=True, show_plot=True,
                                         dark_background=False, dark_color='black',
                                         transparent_png=False):
    plungers = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10']
    qubits = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10']

    qubit_mosaic = [['.', 'Q1', '.', 'Q2', '.', 'Q3', '.'],
                    ['Q4', '.', 'Q5', '.', 'Q6', '.', 'Q7'],
                    ['.', 'Q8', '.', 'Q9', '.', 'Q10', '.']]

    gates = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10',
             'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11', 'B12']

    file_name = 'gfactor_tunability_horseshoe_ind'

    gate_positions = pd.read_csv(DATA_DIR / 'array_343_gate_positions.csv')
    gate_positions = gate_positions.set_index('Key')

    if dark_background:
        accent_color = 'white'
    else:
        accent_color = 'black'

    fig, axes = plt.subplot_mosaic(
        qubit_mosaic, sharex=True, sharey=True, figsize=tools.cm2inch(18, 9))

    cmap = plt.cm.get_cmap('seismic')
    for n, qubit in enumerate(qubits):
        try:
            ax = axes[qubit]
        except:
            ax = axes[n]

        slopes_Qn = slope_gates[n]
        qubit_plunger = plungers[n]

        largest_val = np.max(np.abs(slopes_Qn))
        min_val = -largest_val
        max_val = largest_val
        norm = TwoSlopeNorm(vmin=min_val, vcenter=0, vmax=max_val)

        ax.set_title(qubits[n], color=accent_color)
        for gate, slope in zip(gates, slopes_Qn):
            polygon = Polygon(np.array(gate_positions.loc[f'{gate}']))
            color = cmap(norm(slope))
            patch = PolygonPatch(polygon, fc=color, ec=accent_color, lw=0.1)
            ax.add_patch(patch)

        ax.set_xlim(gate_positions['X'].min(), gate_positions['X'].max())
        ax.set_ylim(gate_positions['Y'].min(), gate_positions['Y'].max())
        ax.set_axis_off()
        ax.set_aspect('equal')

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label=unit,
                     orientation="horizontal", shrink=0.7)

        if dark_background:
            apply_dark_background(fig, cbar, axes, unit, dark_color)
            file_name = file_name + '_dark'

    plt.subplots_adjust(left=0,
                        bottom=0.15,
                        right=1,
                        top=0.9,
                        wspace=0,
                        hspace=0)

    if save:
        fig.savefig(os.path.join(fig_path, f"{file_name}.png"), dpi=300, transparent=transparent_png)
        fig.savefig(os.path.join(fig_path, f"{file_name}.pdf"), dpi=300, transparent=True)

    if show_plot:
        plt.show()
    else:
        plt.close()

# %%% different hole occupation

def plot_LSES(qubit, df, save=True, show_plot=True, unit='kHz/mV',
              dark_background=False, transparent_png=False, dark_color='black',
              vertical_plot=True, plot_with_cbar=False, figsize=None):

    plunger_gates = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10']
    barrier_gates = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11', 'B12']

    hole_fillings = []
    for num_holes in ['1h', '3h', '5h']:
        if qubit in df[num_holes].dropna().index:
            hole_fillings.append(num_holes)

    if dark_background:
        accent_color = 'white'
        background_color = 'black'
    else:
        accent_color = 'black'
        background_color = 'white'

    if len(hole_fillings):
        gate_positions = pd.read_csv(DATA_DIR / 'array_343_gate_positions.csv')
        gate_positions = gate_positions.set_index('Key')

        gates = plunger_gates + barrier_gates

        vmax = 0
        for num_holes in hole_fillings:
            vmax = max(vmax, df[num_holes].max(axis=0).max())
        min_val = -vmax
        max_val = vmax

        cmap = plt.cm.get_cmap('seismic')
        norm = TwoSlopeNorm(vmin=min_val, vcenter=0, vmax=max_val)

        height_ratios = list(3*np.ones(len(hole_fillings))) + [1.5]
        width_ratios = list(6*np.ones(len(hole_fillings))) + [1.5]

        if vertical_plot:
            if plot_with_cbar:
                fig, axes = plt.subplots(len(hole_fillings)+1, 1,
                                         figsize=tools.cm2inch(6, 2+sum(height_ratios)),
                                         gridspec_kw={'height_ratios': height_ratios})
            else:
                if figsize is None:
                    figsize = tools.cm2inch(6, 3 * len(hole_fillings))
                fig, axes = plt.subplots(len(hole_fillings), 1, figsize=figsize)
        else:
            if plot_with_cbar:
                fig, axes = plt.subplots(1, len(hole_fillings)+1,
                                         figsize=tools.cm2inch(2+sum(width_ratios), 5),
                                         gridspec_kw={'width_ratios': width_ratios})
            else:
                if figsize is None:
                    figsize = tools.cm2inch(6 * len(hole_fillings), 5)
                fig, axes = plt.subplots(1, len(hole_fillings), figsize=figsize)

        if type(axes) is np.ndarray:
            axes[-1].axis('off')
        else:
            axes = np.array([axes])
            axes[-1].axis('off')

        for num_holes, ax in zip(hole_fillings, axes):
            slopes_Qn = np.array(df[num_holes][gates].loc[qubit])
            qubit_plunger = f'P{qubit[1:]}'

            # vmax = df[num_holes].loc[qubit].max()
            # norm = Normalize(vmin=0,
            #                  vmax=vmax, clip=True)

            where_are_NaNs = np.isnan(slopes_Qn)
            slopes_Qn[where_are_NaNs] = 0

            ax.set_title(f'{qubit} ({num_holes})', color=accent_color)
            for gate, slope in zip(gates, list(slopes_Qn)):
                polygon = Polygon(np.array(gate_positions.loc[f'{gate}']))
                color = cmap(norm(slope))
                patch = PolygonPatch(polygon, fc=color, ec=accent_color, lw=0.1)
                ax.add_patch(patch)

            ax.set_xlim(gate_positions.loc[gates]['X'].min(), gate_positions.loc[gates]['X'].max())
            ax.set_ylim(gate_positions.loc[gates]['Y'].min(), gate_positions.loc[gates]['Y'].max())
            ax.set_axis_off()
            ax.set_aspect('equal')

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        unit_label = unit

        file_name = f'gfactor_tunability_{qubit}'
        if vertical_plot:
            orientation = 'vertical'
        else:
            orientation = 'horizontal'
        fig_path = f'images/gfactor_tunability/{orientation}_plot/{background_color}'

        if plot_with_cbar:
            if vertical_plot:
                plt.subplots_adjust(left=0,
                                    bottom=0.2,
                                    right=1,
                                    top=0.9,
                                    hspace=0.4)
                cbar = plt.colorbar(sm, ax=axes[-1], label=unit_label,
                              orientation="horizontal", shrink=5)
            else:
                plt.subplots_adjust(left=0,
                                    bottom=0,
                                    right=0.8,
                                    top=0.9,
                                    wspace=0.2)
                cbar = plt.colorbar(sm, ax=axes[-1], label=unit_label,
                              orientation="vertical", shrink=0.4)
            # cbar.ax.tick_params(axis='x', labelrotation=30)
            if dark_background:
                apply_dark_background(fig, cbar, axes, unit_label, dark_color)

        plt.show()
        if save:
            if not os.path.exists(fig_path):
                os.makedirs(fig_path)
            fig.savefig(os.path.join(fig_path, f"{file_name}.png"), dpi=300, transparent=transparent_png)
            fig.savefig(os.path.join(fig_path, f"{file_name}.pdf"), dpi=300, transparent=True)

        if show_plot:
            plt.show()
        else:
            plt.close()

        hcbar_path = os.path.join(fig_path, 'horizontal_colorbars')
        vcbar_path = os.path.join(fig_path, 'vertical_colorbars')
        if not os.path.exists(hcbar_path):
            os.makedirs(hcbar_path)
        if not os.path.exists(vcbar_path):
            os.makedirs(vcbar_path)
        save_colorbar(norm, cmap, unit_label, os.path.join(hcbar_path, file_name),
                      orientation='horizontal', figsize=tools.cm2inch(3.5, 2.5))
        save_colorbar(norm, cmap, unit_label, os.path.join(vcbar_path, file_name),
                      orientation='vertical', figsize=tools.cm2inch(2.5, 3.5))

def plot_LSES_individual(qubit, num_holes, df, vmax=None, save=True, show_plot=True, unit='kHz/mV',
                         dark_background=False, transparent_png=False, dark_color='black',
                         plot_with_cbar=False, figsize=tools.cm2inch(6, 3), save_path=None):

    plunger_gates = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10']
    barrier_gates = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11', 'B12']
    gates = plunger_gates + barrier_gates

    if dark_background:
        accent_color = 'white'
        background_color = 'black'
    else:
        accent_color = 'black'
        background_color = 'white'

    gate_positions = pd.read_csv(DATA_DIR / 'array_343_gate_positions.csv')
    gate_positions = gate_positions.set_index('Key')

    if vmax is None:
        vmax = df[num_holes].loc[qubit].abs().max()
    vmin = - vmax


    cmap = plt.cm.get_cmap('seismic')
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    slopes_Qn = np.array(df[num_holes][gates].loc[qubit])

    where_are_NaNs = np.isnan(slopes_Qn)
    slopes_Qn[where_are_NaNs] = 0

    ax.set_title(f'{qubit} ({num_holes})', color=accent_color)
    for gate, slope in zip(gates, list(slopes_Qn)):
        polygon = Polygon(np.array(gate_positions.loc[f'{gate}']))
        color = cmap(norm(slope))
        patch = PolygonPatch(polygon, fc=color, ec=accent_color, lw=0.1)
        ax.add_patch(patch)

        ax.set_xlim(gate_positions.loc[gates]['X'].min(), gate_positions.loc[gates]['X'].max())
        ax.set_ylim(gate_positions.loc[gates]['Y'].min(), gate_positions.loc[gates]['Y'].max())
        ax.set_axis_off()
        ax.set_aspect('equal')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    unit_label = unit

    file_name = f'gfactor_tunability_{qubit}_{num_holes}'
    fig_path = save_path

    if plot_with_cbar:
        plt.subplots_adjust(left=0,
                            bottom=0.2,
                            right=1,
                            top=0.9,
                            hspace=0.4)
        cbar = plt.colorbar(sm, ax=ax, label=unit_label,
                      orientation="horizontal", shrink=5)
    else:
        plt.tight_layout()
        cbar = None

    if dark_background:
        apply_dark_background(fig, [ax], unit_label, cbar, dark_color)

    plt.show()
    if save:
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
        fig.savefig(os.path.join(fig_path, f"{file_name}.png"), dpi=300, transparent=transparent_png)
        fig.savefig(os.path.join(fig_path, f"{file_name}.pdf"), dpi=300, transparent=True)

    if show_plot:
        plt.show()
    else:
        plt.close()

    hcbar_path = os.path.join(fig_path, 'horizontal_colorbars')
    vcbar_path = os.path.join(fig_path, 'vertical_colorbars')
    if not os.path.exists(hcbar_path):
        os.makedirs(hcbar_path)
    if not os.path.exists(vcbar_path):
        os.makedirs(vcbar_path)
    save_colorbar(norm, cmap, unit_label, os.path.join(hcbar_path, file_name),
                  orientation='horizontal', figsize=tools.cm2inch(3.5, 2.5))
    # save_colorbar(norm, cmap, unit_label, os.path.join(vcbar_path, file_name),
    #               orientation='vertical', figsize=tools.cm2inch(2.5, 3.5))