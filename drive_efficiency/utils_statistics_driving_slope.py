# %% imports

from utils.notebook_tools_costumized import LineOrigin
from utils import *

# %% paths

from pathlib import Path
from config import DATA_DIR

script_dir = Path(__file__).resolve().parent
fig_path = script_dir / "images"

# %% core tools setup to load data

from core_tools.data.ds.ds_hdf5 import load_hdf5_uuid
uuids_data_path = DATA_DIR / 'uuid_datasets'

# %% load data

def load_driving_slope_data(data_path=DATA_DIR):
    df = {}
    df['1h'] = pd.read_csv(os.path.join(data_path, 'driving_strength_1hole.csv'),
                           header=[0], index_col=[0])
    df['3h'] = pd.read_csv(os.path.join(data_path, 'driving_strength_3hole.csv'),
                           header=[0], index_col=[0])
    df['5h'] = pd.read_csv(os.path.join(data_path, 'driving_strength_5hole.csv'),
                           header=[0], index_col=[0])

    df['1h'] = df['1h'].dropna()
    df['3h'] = df['3h'].dropna()
    df['5h'] = df['5h'].dropna()
    return df

# %% helper functions

def get_plunger_qubit_drive(df, qubit):
    plunger_qubit_drive = []
    for num_holes in df.keys():
        if qubit in df[num_holes][f'P{qubit[1:]}']:
            plunger_qubit_drive.append(df[num_holes][f'P{qubit[1:]}'].loc[qubit])
        else:
            plunger_qubit_drive.append(np.nan)
    return plunger_qubit_drive

def create_dicts():
    uuids = {}
    slopes = {}

    for n in range(1,11):
        uuids[f'Q{n}'] = {}
        slopes[f'Q{n}'] = {}
        for num_holes in ['1h', '3h', '5h']:
            uuids[f'Q{n}'][num_holes] = {}
            uuids[f'Q{n}'][num_holes]['plunger'] = []
            uuids[f'Q{n}'][num_holes]['barrier'] = []

    return uuids, slopes

def replace_nan_with_zero(input_list):
    return [0 if np.isnan(x) else x for x in input_list]

# %% fitting functions

def fit_freq_2D(x, y, z, plot_ramsey=False, num=2):
    z_fft = []
    for i in range(len(y)):
        zfft = np.fft.fft(z[i])
        z_fft.append(zfft)

    z_fft = np.array(z_fft)
    n = x.size
    timestep = x[1] - x[0]
    freq = np.fft.fftfreq(n, d=timestep) * 1e3
    freq_fft = freq[1:round(len(freq) / 2)]
    fft_norm = np.abs(z_fft)[:, 1:round(len(freq) / 2)] / np.abs(zfft).max()

    # skip_lines = 3
    peaks = []
    for zline in fft_norm:
        peaks.append(np.argmax(zline[1:]))
    peaks = np.array(peaks)

    freq_guess = freq_fft[peaks]

    fit_osc_freq = []
    for i in np.arange(0, len(fft_norm)):
        x_nonan = x[~np.isnan(z[i])]
        z_i = z[i][~np.isnan(z[i])]
        amp_guess = max(z_i) - min(z_i)
        # print('____________________')
        # print(x_nonan)
        # print(z_i)
        # print(amp_guess)
        # print(freq_guess[i])
        # print('____________________')
        fit_params = fit_data(
            x_nonan, z_i, p0=[amp_guess, freq_guess[i] * 1e-3, 10000, 0.5, 3], func=Ramsey, plot=plot_ramsey)
        if fit_params is None:
            fit_osc_freq.append(float('nan'))
        else:
            low_amp_filter = (abs(fit_params[0]) < 0.1)
            low_freq_filter = (fit_params[1] < 0.1e-6)
            low_T2_filter = (fit_params[2] < 300)
            offset_filter = (fit_params[3] < 0.2) or (fit_params[3] > 0.8)
            failed_conditions = [
                "Low Amplitude" if low_amp_filter else "",
                "Low Frequency" if low_freq_filter else "",
                "Low T2" if low_T2_filter else "",
                "Offset" if offset_filter else ""
            ]
            # Filter out the empty strings and join the failed conditions
            failed_conditions = ', '.join(filter(None, failed_conditions))

            if low_amp_filter or offset_filter or low_T2_filter or low_freq_filter:
                fit_osc_freq.append(float('nan'))
                if plot_ramsey:
                    title_text = f"Fit discarded: {failed_conditions}" if failed_conditions else "Fit discarded"
                    plt.title(title_text, color='red', style='oblique')
            else:
                fit_osc_freq.append(fit_params[1])
                if plot_ramsey:
                    plt.title('Fit accepted', color='green', style='oblique')

        if plot_ramsey:
            plt.xlabel(r'$t_{\rm{mw}}$ (ns)')
            plt.tight_layout()
            plt.show()

    return fit_osc_freq, freq_fft, fft_norm

def extract_slope(fit_osc_freq, y):
    fit_osc_freq_s = np.array(fit_osc_freq)[~np.isnan(fit_osc_freq)] * 1e3
    amp_set = y[~np.isnan(fit_osc_freq)]

    if len(fit_osc_freq_s) > 3:
        fit_param = fit_data(amp_set, fit_osc_freq_s,
                             p0=[fit_osc_freq_s[-1] / amp_set[-1]],
                             func=LineOrigin, plot=False)
        off_set = 0
        slope = fit_param[0]

        m_freq = fit_osc_freq_s[-1]
        if slope < 0 or off_set > 0.2:
            slope = float('nan')
            off_set = float('nan')
            m_freq = float('nan')
    else:
        slope = float('nan')
        off_set = float('nan')
        m_freq = float('nan')

    return slope, off_set, m_freq

# %% plotting functions

def driving_speed_amp_fft_plot(uuids, gates, plot=True, plot_ramsey=False, print_id=True,
                               colorbar=False, filename_cbar=None):
    """
    plots a subplot of driving freq vs amplitude

    """

    max_freq = []
    off_set_list = []
    slope_list = []

    results = {}

    for n, (gate, id) in enumerate(zip(gates, uuids)):
        dat = load_hdf5_uuid(id, uuids_data_path)

        x, x_label, x_unit = dat.m1_2.y(), dat.m1_2.y.label, dat.m1_2.y.unit
        y, y_label, y_unit = dat.m1_2.x(), dat.m1_2.x.label, dat.m1_2.x.unit
        z = dat.m1_2()

        fit_osc_freq, freq_fft, fft_norm = fit_freq_2D(x, y, z, plot_ramsey=plot_ramsey)
        slope, off_set, m_freq = extract_slope(fit_osc_freq, y)

        max_freq.append(m_freq)
        off_set_list.append(off_set)
        slope_list.append(slope)

        results[gate] = {}
        results[gate]['fit_freq_2D'] = fit_osc_freq, freq_fft, fft_norm
        results[gate]['slope'] = slope, off_set, m_freq

    if plot:
        if gates[0][0] in ['P'] or gates[0][:2] in ['vP']:
            style = 'plunger'
            fig, axes, axes_left, axes_bottom = create_343_subplot(style=style, num=1)
        if gates[0][0] in ['J', 'B'] or gates[0][:2] in ['vB']:
            style = 'barrier'
            fig, axes, axes_left, axes_bottom = create_343_subplot(style=style, num=1)

        for gate in gates:
            fit_osc_freq, freq_fft, fft_norm = results[gate]['fit_freq_2D']
            slope, off_set, m_freq = results[gate]['slope']

            amp_set = y[~np.isnan(fit_osc_freq)]

            ax = axes[gate]

            if print_id:
                title = f'{gate} \n id: {dat.exp_id}'
            else:
                title = f'{gate}'
            ax.set_title(title)
            ax.pcolor(y, freq_fft, np.transpose(fft_norm),  shading='auto')

            if not np.isnan(slope):
                ax.plot(amp_set, Line(amp_set, slope, off_set),
                        color='white', ls='--', label=f'{slope:.3f} MHz/mV')
                ax.set_ylim(min(freq_fft), max(freq_fft))
                ax.legend(labelcolor='white')

            if ax in axes_left:
                ax.set_ylabel(f'f (MHz)')
            if ax in axes_bottom:
                ax.set_xlabel(f'{y_label} ({y_unit})')

            if style == 'plunger':
                plt.subplots_adjust(left=0.1,
                                    bottom=0.12,
                                    right=0.9,
                                    top=0.85,
                                    wspace=0.1,
                                    hspace=0.1)
            else:
                plt.subplots_adjust(left=0.1,
                                    bottom=0.15,
                                    right=0.9,
                                    top=0.85,
                                    wspace=0.3,
                                    hspace=0.4)

        plt.show()

        if colorbar:
            unit_label = r'$P_{\text{even}}$'
            save_colorbar(norm, cmap, unit_label, filename_cbar, orientation='vertical', figsize=None,
                          ticklabelsize=None, labelsize=None, shrink=1, tick_number=3, location=None, labelpad=None)


    return slope_list, max_freq, fig

def overview_plot_cross_talk(slope_gates, dark_background=False, dark_color='black', transparent_png=False,
                             figsize=tools.cm2inch(18, 8)):

    plungers = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10']
    qubits = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10']

    qubit_mosaic = [['.', 'Q1', '.', 'Q2', '.', 'Q3', '.'],
                    ['Q4', '.', 'Q5', '.', 'Q6', '.', 'Q7'],
                    ['.', 'Q8', '.', 'Q9', '.', 'Q10', '.']]

    gates = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10',
             'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11', 'B12']

    gate_positions = pd.read_csv(DATA_DIR / 'array_343_gate_positions.csv')
    gate_positions = gate_positions.set_index('Key')

    cmap = cmap_wyr()
    norm = Normalize(vmin=0,
                     vmax=max(slope_gates.flatten()), clip=True)

    if dark_background:
        accent_color = 'white'
    else:
        accent_color = 'black'

    fig, axes = plt.subplot_mosaic(
        qubit_mosaic, sharex=True, sharey=True, figsize=figsize)

    for n, qubit in enumerate(qubits):
        try:
            ax = axes[qubit]
        except:
            ax = axes[n]

        slopes_Qn = slope_gates[n]
        qubit_plunger = plungers[n]

        where_are_NaNs = np.isnan(slopes_Qn)
        where_are_neg = slopes_Qn < 0
        slopes_Qn[where_are_NaNs] = 0
        slopes_Qn[where_are_neg] = 0

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
    unit_label = 'MHz/mV'
    cbar = fig.colorbar(sm, ax=axes['Q9'], label=unit_label,
                        orientation="horizontal", shrink=1)

    plt.subplots_adjust(left=0,
                        bottom=0.15,
                        right=1,
                        top=0.9,
                        wspace=-0.3,
                        hspace=-0.3)

    file_name = 'EDSR_efficiency_horseshoe'
    if dark_background:
        apply_dark_background(fig, cbar, axes, unit_label, dark_color)
        file_name = file_name+'_dark'

    plt.savefig(os.path.join(
        fig_path, f'{file_name}.png'), dpi=300, transparent=transparent_png)
    plt.savefig(os.path.join(
        fig_path, f'{file_name}.pdf'), dpi=300, transparent=True)
    plt.show()


def overview_plot_cross_talk_indv(slope_gates, figsize=tools.cm2inch(18, 9)):
    plungers = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10']
    qubits = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10']

    qubit_mosaic = [['.', 'Q1', '.', 'Q2', '.', 'Q3', '.'],
                    ['Q4', '.', 'Q5', '.', 'Q6', '.', 'Q7'],
                    ['.', 'Q8', '.', 'Q9', '.', 'Q10', '.']]

    gates = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10',
             'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11', 'B12']

    gate_positions = pd.read_csv(DATA_DIR / 'array_343_gate_positions.csv')
    gate_positions = gate_positions.set_index('Key')

    fig, axes = plt.subplot_mosaic(
        qubit_mosaic, sharex=True, sharey=True, figsize=figsize)

    for n, qubit in enumerate(qubits):
        try:
            ax = axes[qubit]
        except:
            ax = axes[n]

        slopes_Qn = slope_gates[n]
        qubit_plunger = plungers[n]

        cmap = cmap_wyr()
        norm = Normalize(vmin=min(slopes_Qn),
                         vmax=max(slopes_Qn), clip=True)

        where_are_NaNs = np.isnan(slopes_Qn)
        where_are_neg = slopes_Qn < 0
        slopes_Qn[where_are_NaNs] = 0
        slopes_Qn[where_are_neg] = 0

        ax.set_title(qubits[n], color='black')
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
        plt.colorbar(sm, ax=ax, label='MHz/mV',
                     orientation="horizontal", shrink=0.7)

    # cbar_ax = fig.add_axes([0.9, 0.15, 0.01, 0.7])

    # ax.set_colorbar(label = 'MHz/mV', shrink=0.7)
    plt.subplots_adjust(left=0,  # 0.2
                        bottom=0.15,
                        right=1,
                        top=0.9,
                        wspace=0,
                        hspace=0)

    plt.savefig(os.path.join(
        fig_path, 'EDSR_efficiency_horseshoe_ind.png'), dpi=300)
    plt.savefig(os.path.join(
        fig_path, 'EDSR_efficiency_horseshoe_ind.pdf'), dpi=300)
    plt.show()

def plot_rabi_efficiency_occupation_comparison(qubit, df,
                                               shared_color_scale=True,
                                               save=True,
                                               dark_background=False,
                                               dark_color='black',
                                               transparent_png=False,
                                               vertical_plot=True,
                                               figsize=None,
                                               plot_with_cbar=False):
    hole_fillings = []
    for num_holes in ['1h', '3h', '5h']:
        if qubit in df[num_holes].index:
            hole_fillings.append(num_holes)

    if figsize is not None and not vertical_plot:
            figsize = (figsize[0] / 4 * (len(hole_fillings)+1),
                       figsize[1])

    gate_positions = pd.read_csv(DATA_DIR / 'array_343_gate_positions.csv')
    gate_positions = gate_positions.set_index('Key')

    plunger_gates = ['P1', 'P2', 'P3', 'P4', 'P5',
                     'P6', 'P7', 'P8', 'P9', 'P10']
    barrier_gates = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6',
                     'B7', 'B8', 'B9', 'B10', 'B11', 'B12']
    gates = plunger_gates + barrier_gates

    if dark_background:
        accent_color = 'white'
        background_color = 'black'
    else:
        accent_color = 'black'
        background_color = 'white'

    cmap = cmap_wyr()

    height_ratios = list(3 * np.ones(len(hole_fillings))) + [1.]
    width_ratios = list(6 * np.ones(len(hole_fillings))) + [1.5]
    if shared_color_scale:
        vmax = 0
        for num_holes in hole_fillings:
            vmax = max(vmax, df[num_holes].loc[qubit].max(axis=0).max())
        norm = Normalize(vmin=0, vmax=vmax, clip=True)

        if vertical_plot:
            if plot_with_cbar:
                if figsize is None:
                    figsize = tools.cm2inch(6, sum(height_ratios) * 1.5)
                fig, axes = plt.subplots(len(hole_fillings)+1, 1,
                                         figsize=figsize,
                                         gridspec_kw={'height_ratios': height_ratios})
            else:
                if figsize is None:
                    figsize = tools.cm2inch(6, 3 * len(hole_fillings))
                fig, axes = plt.subplots(len(hole_fillings), 1, figsize=figsize)
        else:
            if plot_with_cbar:
                if figsize is None:
                    figsize = tools.cm2inch(2 + sum(width_ratios), 5)
                fig, axes = plt.subplots(1, len(hole_fillings)+1,
                                         figsize=figsize,
                                         gridspec_kw={'width_ratios': width_ratios})
            else:
                if figsize is None:
                    figsize = tools.cm2inch(5, 6 * len(hole_fillings))
                fig, axes = plt.subplots(1, len(hole_fillings), figsize=figsize)
        if type(axes) is np.ndarray:
            axes[-1].axis('off')
        else:
            axes = np.array([axes])
            axes[-1].axis('off')
    else:
        if figsize is None:
            figsize = tools.cm2inch(6, sum(height_ratios) * 1)
        fig, axes = plt.subplots(len(hole_fillings), 1,
                                 figsize=figsize,
                                 gridspec_kw={'height_ratios': height_ratios})

    for num_holes, ax in zip(hole_fillings, axes):
        slopes_Qn = np.array(df[num_holes][gates].loc[qubit])

        if not shared_color_scale:
            vmax = df[num_holes].loc[qubit].max()
            norm = Normalize(vmin=0, vmax=vmax, clip=True)

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

        unit_label = 'MHz/mV'
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        if not shared_color_scale and plot_with_cbar:
            cbar = plt.colorbar(sm, ax=ax, label='MHz/mV',
                          orientation="horizontal", shrink=0.5)

    file_name = f"EDSR_efficiency_{qubit}_{'_'.join(hole_fillings)}"
    if vertical_plot:
        orientation = 'vertical'
    else:
        orientation = 'horizontal'
    fig_path = f'images/EDSR_efficiency/{orientation}_plot/{background_color}_background'

    if shared_color_scale and plot_with_cbar:
        if vertical_plot:
            cbar = plt.colorbar(sm, ax=axes[-1], label=unit_label,
                          orientation="horizontal", shrink=1)
            plt.subplots_adjust(left=0,
                                bottom=0.15,
                                right=1,
                                top=0.9,
                                hspace=0.4)
        else:
            plt.subplots_adjust(left=0,
                                bottom=0,
                                right=0.8,
                                top=0.9,
                                wspace=0.2)
            cbar = plt.colorbar(sm, ax=axes[-1], label=unit_label,
                          orientation="vertical", shrink=0.4)
    else:
        cbar = None
        plt.tight_layout()

    if dark_background:
        apply_dark_background(fig, axes, unit_label, cbar, dark_color)

    if save:
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
        fig.savefig(os.path.join(fig_path, f"{file_name}.png"), dpi=300, transparent=transparent_png)
        fig.savefig(os.path.join(fig_path, f"{file_name}.pdf"), dpi=300, transparent=True)
    plt.show()

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

def plot_rabi_efficiency_individual(qubit, num_holes, df,
                                    vmax=None,
                                    save=True,
                                    dark_background=False,
                                    dark_color='black',
                                    transparent_png=False,
                                    figsize=tools.cm2inch(6,3),
                                    plot_with_cbar=False,
                                    save_path=None):

    gate_positions = pd.read_csv(DATA_DIR / 'array_343_gate_positions.csv')
    gate_positions = gate_positions.set_index('Key')

    plunger_gates = ['P1', 'P2', 'P3', 'P4', 'P5',
                     'P6', 'P7', 'P8', 'P9', 'P10']
    barrier_gates = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6',
                     'B7', 'B8', 'B9', 'B10', 'B11', 'B12']
    gates = plunger_gates + barrier_gates

    if dark_background:
        accent_color = 'white'
        background_color = 'black'
    else:
        accent_color = 'black'
        background_color = 'white'

    cmap = cmap_wyr()

    if vmax is None:
        vmax = df[num_holes].loc[qubit].max()
    norm = Normalize(vmin=0, vmax=vmax, clip=True)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    slopes_Qn = np.array(df[num_holes][gates].loc[qubit])

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

    unit_label = r'$f_{\rm{R}}/A$ (MHz/mV)'
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    if plot_with_cbar:
        cbar = plt.colorbar(sm, ax=ax, label='MHz/mV',
                      orientation="horizontal", shrink=0.5)

    file_name = f"EDSR_efficiency_{qubit}_{num_holes}"
    fig_path = save_path

    if plot_with_cbar:
        cbar = plt.colorbar(sm, ax=ax, label=unit_label,
                      orientation="horizontal", shrink=1)
        plt.subplots_adjust(left=0,
                            bottom=0.15,
                            right=1,
                            top=0.9,
                            hspace=0.4)
    else:
        cbar = None
        plt.tight_layout()

    if dark_background:
        apply_dark_background(fig, [ax], unit_label, cbar, dark_color)

    if save:
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
        fig.savefig(os.path.join(fig_path, f"{file_name}.png"), dpi=300, transparent=transparent_png)
        fig.savefig(os.path.join(fig_path, f"{file_name}.pdf"), dpi=300, transparent=True)
    plt.show()

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


def plot_rabi_efficiency_gate(drive_gate, df, shared_color_scale = True,
                              save=True,
                              dark_background=False, dark_color='black',
                              transparent_png=False,
                              figsize=tools.cm2inch(6,8)):


    plunger_gates = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10']
    barrier_gates = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11', 'B12']
    gates = plunger_gates + barrier_gates

    cmap = cmap_wyr()
    if shared_color_scale:
        vmax = max(df['1h'][drive_gate].max(axis=0).max(), df['3h'][drive_gate].max(axis=0).max())
        norm = Normalize(vmin=0, vmax=vmax, clip=True)

        fig, axes = plt.subplots(3,1,figsize=figsize, gridspec_kw={'height_ratios': [3, 3, 1]})
        axes[2].axis('off')
    else:
        fig, axes = plt.subplots(2,1,figsize=figsize)

    if dark_background:
        accent_color = 'white'
    else:
        accent_color = 'black'

    for num_holes, ax in zip(['1h', '3h'], axes):
        if not shared_color_scale:
            vmax = df[num_holes][drive_gate].max()
            norm = Normalize(vmin=0,
                              vmax=vmax, clip=True)

        ax.set_title(f'{drive_gate} ({num_holes})', color=accent_color, fontsize=10)
        for gate in plunger_gates:
            qubit = f'Q{gate[1:]}'
            if qubit in df[num_holes][drive_gate].dropna().index:
                slope = df[num_holes][drive_gate][qubit]
                ls = '-'
                lw = 1
            else:
                slope = 0
                ls = ':'
                lw = 0.1
            if gate == drive_gate:
                ec = 'red'
                lw = 1
            else:
                ec = accent_color

            polygon = Polygon(np.array(gate_positions.loc[gate]))
            color = cmap(norm(slope))
            patch = PolygonPatch(polygon, fc=color, ec=ec, lw=lw, ls=ls)
            ax.add_patch(patch)

        for gate in barrier_gates:
            slope = 0
            ls = ':'

            if gate == drive_gate:
                ec = 'red'
                lw = 1
            else:
                ec = accent_color
                lw = 0.1

            polygon = Polygon(np.array(gate_positions.loc[gate]))
            color = cmap(norm(slope))
            patch = PolygonPatch(polygon, fc=color, ec=ec, lw=lw, ls=ls)
            ax.add_patch(patch)

        ax.set_xlim(gate_positions.loc[gates]['X'].min(), gate_positions.loc[gates]['X'].max())
        ax.set_ylim(gate_positions.loc[gates]['Y'].min(), gate_positions.loc[gates]['Y'].max())
        ax.set_axis_off()
        ax.set_aspect('equal')

        if not shared_color_scale:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            unit_label = 'MHz/mV'
            # cbar = plt.colorbar(sm, ax=ax, label=unit_label,
            #               orientation="horizontal", shrink=0.5)

    if shared_color_scale:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        unit_label = 'MHz/mV'
        cbar = plt.colorbar(sm, ax=axes[2], label=unit_label,
                      orientation="horizontal", shrink=1)
        plt.subplots_adjust(left=0,
                            bottom=0.15,
                            right=1,
                            top=0.9,
                            hspace=0.4)
    else:
        plt.subplots_adjust(left=0,
                            bottom=0.15,
                            right=1,
                            top=0.9,
                            hspace=0.6)

    file_name = f"EDSR_locality_{drive_gate}"

    fig_path = get_fig_path()
    fig_path = os.path.join(fig_path, 'EDSR_locality')
    if dark_background:
        apply_dark_background(fig, cbar, axes, unit_label, dark_color)
        fig_path = os.path.join(fig_path, 'dark_background')
    else:
        fig_path = os.path.join(fig_path, 'white_background')

    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    if save:
        fig.savefig(os.path.join(fig_path, f"{file_name}.png"), dpi=300, transparent=transparent_png)
        fig.savefig(os.path.join(fig_path, f"{file_name}.pdf"), dpi=300, transparent=True)
    plt.show()