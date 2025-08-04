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

from single_qubit_rb.utils_RB import (RB_decay, get_value_with_error, RB_decay_fixed_p, RB_decay_fixed_AB,
                                      get_max_likelihood_fidelity_with_error, get_gate_fidelity_from_decay,
                                      get_fidelity_from_least_squares, fit_rb_mle)
from utils.utils import *
import matplotlib.pyplot as plt

# %% core tools setup to load data

from core_tools.data.ds.ds_hdf5 import load_hdf5_uuid
uuids_data_path = DATA_DIR / 'uuid_datasets'

#%% import data 
uuid_qubits = {} 
uuid_qubits['Q1'] = 1_712_304_198_732_283_691
uuid_qubits['Q2'] = 1_712_135_103_826_283_691
uuid_qubits['Q3'] = 1_712_589_170_243_283_691
uuid_qubits['Q4'] = 1_712_865_506_219_283_691
uuid_qubits['Q5'] = 1_712_841_924_920_283_691
uuid_qubits['Q6'] = 1_712_300_223_914_283_691 # 1_712_306_894_198_283_691
uuid_qubits['Q7'] =  1_712_675_030_039_283_691
uuid_qubits['Q8'] = 17122_21770_844283691
uuid_qubits['Q9'] = 1_712_146_779_055_283_691
uuid_qubits['Q10'] = 1_712_763_230_584_283_691

# additional Q6 RB datasets


#%% plot data

method = 'max_likelihood'  # 'max_likelihood' or 'least_squares'

amp_estimation = [-0.22, -0.33, 0.34, -0.24, -0.33, -0.31, -0.42, -0.34, -0.29, -0.25]
offset_estimation = [0.41, 0.39, 0.57, 0.47, 0.44, 0.40, 0.47, 0.40, 0.27, 0.44]
decay_constant_estimation = [0.998, 0.996, 0.995, 0.996, 0.995, 0.997, 0.996, 0.997, 0.994, 0.994]

for n, (qubit, uuid) in enumerate(uuid_qubits.items()):
    print('----------')
    print(f'Qubit {qubit}')
    
    data = load_hdf5_uuid(uuid, uuids_data_path)

    f_rabi_MHz = 1e3 / (4*data.snapshot['measurement']['sequence']['settings'][f'q{qubit[1:]}']['x90']['t_pulse'])
    print(f'f_Rabi = {f_rabi_MHz:.2f} MHz')

    x, x_label, x_unit = data.m1_3.i(), data.m1_3.i.label, data.m1_3.i.unit
    y, y_label, y_unit  = data.m1_3.j(), data.m1_3.j.label, data.m1_3.j.unit
    z = data.m1_2()

    N_Cliffords = y
    z_avrg = np.average(z, axis = 0)
    z_std = np.std(z, axis=0)
    z_sigma = z_std / np.sqrt(z.shape[0])  # standard error of the mean

    if method == 'least-squares':
        # Fit the data using least squares
        fit_lsq = get_fidelity_from_least_squares(N_Cliffords=y, Z_data=z, amp=amp_estimation[n],
                                                  offset=offset_estimation[n], f_gate_start=0.99,
                                                  f_gate_end=0.999)
        F_gate = fit_lsq['f_gate']
        F_err = fit_lsq['f_gate_err']
        amp_fit = fit_lsq['amp']
        offset_fit = fit_lsq['offset']
        popt = [amp_fit, offset_fit, fit_lsq['decay_constant']]
    elif method == 'max_likelihood':
        # Fit the data using maximum likelihood estimation
        p0 = [amp_estimation[n], offset_estimation[n], decay_constant_estimation[n]]
        initial_guess = [p0[0], p0[1], p0[2]]
        fit = fit_rb_mle(N_Cliffords, z.T, initial_guess=initial_guess)
        F_gate = fit['f_gate']
        F_err = fit['f_err']
        amp_fit = fit['amp']
        offset_fit = fit['offset']
        popt = [amp_fit, offset_fit, fit['p']]


    # print(f'A_fit = {amp_fit:.3f}({amp_err:.3f}), ')
    # print(f'offset = {offset_fit:.3f}({offset_err:.3f}), ')
    # print(f'p = {p_fit:.3f}({p_err:.3f})')

    P_9 = 1 - 4 * (1 - 0.9)
    P_99 = 1 - 4*(1-0.99)
    P_999 = 1 - 4*(1-0.999)
    P_9999 = 1 - 4*(1-0.9999)

    popt_9 = [amp_fit, offset_fit, P_9]
    popt_99 = [amp_fit, offset_fit, P_99]
    popt_999 = [amp_fit, offset_fit, P_999]
    popt_9999 = [amp_fit, offset_fit, P_9999]

    # Calculate qubit gate fidelities
    # error = 1 - p_fit
    # F_gate = 1-(error/(2*2))
    # F_err = p_err/(2*2)
    # std_dev = np.std(z_std)*1e2
    
    fig, ax = plt.subplots(1, 1, figsize = cm2inch(4.8,4.2))
    
    if qubit == 'Q3':
        ax.fill_between(N_Cliffords, 1-RB_decay(N_Cliffords, *popt_9), 1-RB_decay(N_Cliffords, *popt_99),
                        color = 'grey',
                        alpha = 0.2)
        ax.fill_between(N_Cliffords, 1-RB_decay(N_Cliffords, *popt_99), 1-RB_decay(N_Cliffords, *popt_999),
                        color = 'grey',
                        alpha = 0.4)
        ax.fill_between(N_Cliffords, 1-RB_decay(N_Cliffords, *popt_999), 1-RB_decay(N_Cliffords, *popt_9999),
                        color = 'grey',
                        alpha = 0.6)
        ax.plot(N_Cliffords, 1-RB_decay(N_Cliffords, *popt_99), color = 'black',
                linestyle='--', lw=1,
                # label=f'$F_{{gate}} = 0.99$'
                )
        ax.plot(N_Cliffords, 1-RB_decay(N_Cliffords, *popt_999), color = 'black',
                linestyle=':', lw=1,
                # label=f'$F_{{gate}} = 0.999$'
                )
        ax.fill_between(N_Cliffords, 1- (z_avrg - z_std), 1- (z_avrg + z_std), color='tab:blue',
                        alpha=0.2)
        ax.scatter(N_Cliffords, 1- z_avrg , s = 2, color = 'tab:blue', alpha = 1)
        ax.plot(N_Cliffords, 1- RB_decay(N_Cliffords, *popt), color = 'orangered')
    else:    
        ax.fill_between(N_Cliffords, RB_decay(N_Cliffords, *popt_9), RB_decay(N_Cliffords, *popt_99),
                        color = 'grey',
                        alpha = 0.2)
        ax.fill_between(N_Cliffords, RB_decay(N_Cliffords, *popt_99), RB_decay(N_Cliffords, *popt_999),
                        color = 'grey',
                        alpha = 0.4)
        ax.fill_between(N_Cliffords, RB_decay(N_Cliffords, *popt_999), RB_decay(N_Cliffords, *popt_9999),
                        color = 'grey',
                        alpha = 0.6)
        ax.plot(N_Cliffords, RB_decay(N_Cliffords, *popt_99), color = 'black',
                linestyle='--', lw=1,
                # label=f'$F_{{gate}} = 0.99$'
                )
        ax.plot(N_Cliffords, RB_decay(N_Cliffords, *popt_999), color = 'black',
                linestyle=':', lw=1,
                # label=f'$F_{{gate}} = 0.999$'
                )
        ax.fill_between(N_Cliffords, z_avrg - z_std, z_avrg + z_std, color='tab:blue',
                        alpha=0.2)
        ax.scatter(N_Cliffords, z_avrg , s = 2., color = 'tab:blue', alpha = 1)
        ax.plot(N_Cliffords, RB_decay(N_Cliffords, *popt), color = 'orangered')
        
    ax.set_xlabel('Number of Cliffords')
    ax.set_ylabel('$P_{even}$')
    ax.set_title(rf'{qubit}: $F_\mathrm{{gate}} = {get_value_with_error(F_gate, F_err)}$')
    ax.legend()
    
    
    fig.tight_layout()
    fig.savefig(os.path.join(fig_path, f'RB_{qubit}.png'), dpi=300)
    fig.savefig(os.path.join(fig_path, f'RB_{qubit}.pdf'), dpi=300, transparent=True)
    plt.show()

# %% estimate error via Profile Likelihood (or RSS/R² scan)

amp_estimation = [-0.22, -0.33, 0.34, -0.24, -0.33, -0.31, -0.42, -0.34, -0.29, -0.25]
offset_estimation = [0.41, 0.39, 0.57, 0.47, 0.44, 0.40, 0.47, 0.40, 0.27, 0.44]

fidelity_estimation = [0.998, 0.996, 0.995, 0.996, 0.995, 0.997, 0.996, 0.997, 0.994, 0.994]
decay_constant_estimation = list(1-4*(1-np.array(fidelity_estimation)))
decay_constant_99 = 1-4*(1-0.99)

for n, (qubit, uuid) in enumerate(uuid_qubits.items()):
    print('----------')
    print(f'Qubit {qubit}')

    data = load_hdf5_uuid(uuid, uuids_data_path)

    f_rabi_MHz = 1e3 / (4 * data.snapshot['measurement']['sequence']['settings'][f'q{qubit[1:]}']['x90']['t_pulse'])
    print(f'f_Rabi = {f_rabi_MHz:.2f} MHz')

    x, x_label, x_unit = data.m1_3.i(), data.m1_3.i.label, data.m1_3.i.unit
    y, y_label, y_unit = data.m1_3.j(), data.m1_3.j.label, data.m1_3.j.unit
    z = data.m1_2()

    print(f'Number of Cliffords: {max(y)}')
    print(f'Number of steps: {z.shape[1]}')
    print(f'Number of randomisations: {z.shape[0]}')

    # averaging over randomisations
    N_Cliffords = y
    z_avrg = np.average(z, axis=0)
    z_std = np.std(z, axis=0)
    z_sigma = z_std / np.sqrt(z.shape[0])  # standard error of the mean

    # Define grid of p values
    gate_fidelity_values = np.linspace(0.99, 0.9999, 200)
    # gate_fidelity_values = 1 - np.logspace(-3, -2, 101)
    p_values = 1-4 * (1 - gate_fidelity_values)  # convert gate fidelity to p
    chisq_values = []

    for p_fixed in p_values:
        # Define model with p fixed
        model = RB_decay_fixed_p(N_Cliffords, None, None, p_fixed)

        # Fit only amp and offset
        try:
            popt_tmp, _ = curve_fit(model, N_Cliffords, z_avrg,
                                    sigma=z_sigma,
                                    p0=[amp_estimation[n], offset_estimation[n]],
                                    maxfev=1_000_000,
                                    absolute_sigma=True)

            # Compute residuals and RSS
            residuals = z_avrg - model(N_Cliffords, *popt_tmp)
            chisq = np.sum((residuals / z_sigma) ** 2)  # chi-squared
            chisq_values.append(chisq)
        except RuntimeError:
            chisq_values.append(np.nan)  # failed fit

    chisq_values = np.array(chisq_values)
    min_chisq = np.min(chisq_values)

    # Define Δχ² for 1σ confidence level in 1 parameter
    delta_chi2_1sigma = 1.0

    delta_chi2_99 = 6.63  # for 1 parameter at 99% confidence level

    # Find the range of p where RSS is within 99%
    p_conf_int = gate_fidelity_values[chisq_values <= min_chisq + delta_chi2_99]
    p_lower, p_upper = p_conf_int[0], p_conf_int[-1]
    p_best = gate_fidelity_values[np.argmin(chisq_values)]
    p_err = (p_upper - p_lower) / 2

    # Find the range of p where RSS is within 1σ
    p_conf_int_1sigma = gate_fidelity_values[chisq_values <= min_chisq + delta_chi2_1sigma]
    p_lower_1sigma, p_upper_1sigma = p_conf_int_1sigma[0], p_conf_int_1sigma[-1]
    p_best_1sigma = gate_fidelity_values[np.argmin(chisq_values)]
    p_err_1sigma = (p_upper_1sigma - p_lower_1sigma) / 2

    # Find fidelity at previously expected amplitude and offset
    model_AB = RB_decay_fixed_AB(N_Cliffords, amp_estimation[n],
                                offset_estimation[n], None)
    popt_ab, _ = curve_fit(model_AB, N_Cliffords, z_avrg,
                           sigma=z_sigma,
                           p0=[decay_constant_estimation[n]],
                           maxfev=1_000_000,
                           absolute_sigma=True)
    gate_fidelity_ab = 1-(1-popt_ab[0])/4
    residuals_ab = z_avrg - model_AB(N_Cliffords, *popt_ab)
    chisq_ab = np.sum((residuals_ab / z_sigma) ** 2)  # chi-squared

    fig, ax = plt.subplots(1, 1, figsize=cm2inch(4.8, 4.2))
    plt.plot(gate_fidelity_values, chisq_values, color='blue')
    plt.axvline(p_best, color='green', linestyle='--',
                label=rf'$F_{{gate}}$ = {get_value_with_error(p_best_1sigma, p_err_1sigma)}'
                )
    plt.axvspan(p_lower_1sigma, p_upper_1sigma, color='gray', alpha=0.6,
                # label='1σ confidence level'
                )
    plt.axvspan(p_lower, p_upper, color='gray', alpha=0.3,
                # label='99% confidence level'
                )
    # plt.scatter([gate_fidelity_ab], [chisq_ab], marker='*', color='black')
    plt.xlabel(r'$F_{gate}$')
    plt.ylabel(r'chi^2') # chisq
    # plt.title(f'Qubit {qubit} - Profile Likelihood for p')
    plt.ylim(0, 500)
    # plt.xscale('log')
    plt.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(fig_path, f'RB_profile_likelihood_{qubit}.png'), dpi=300)
    fig.savefig(os.path.join(fig_path, f'RB_profile_likelihood_{qubit}.pdf'), dpi=300, transparent=True)
    plt.show()


# %% using raw likelihood to estimate fidelity

p0 = [amp_estimation[n], offset_estimation[n], decay_constant_estimation[n]]
F_gate, F_err = get_max_likelihood_fidelity_with_error(N_Cliffords, z, p0)

print(get_value_with_error(F_gate, F_err))

# %% Dataframe comparing fidelities from least square fits and MLE fits

amp_estimation = [-0.22, -0.33, 0.34, -0.24, -0.33, -0.31, -0.42, -0.34, -0.29, -0.25]
offset_estimation = [0.41, 0.39, 0.57, 0.47, 0.44, 0.40, 0.47, 0.40, 0.27, 0.44]
fidelity_estimation = [0.998, 0.996, 0.995, 0.996, 0.995, 0.997, 0.996, 0.997, 0.994, 0.994]
decay_constant_estimation = list(1-4*(1-np.array(fidelity_estimation)))

import pandas as pd
fidelities_and_errors = {
    'Qubit': [],
    'F_lsq': [],
    'A_lsq': [],
    'Offset_lsq': [],
    'F_mle': [],
    'F_mle_round': [],
    'A_mle': [],
    'Offset_mle': [],
}

for n, (qubit, uuid) in enumerate(uuid_qubits.items()):
    fidelities_and_errors['Qubit'].append(qubit)

    # Load data
    data = load_hdf5_uuid(uuid, uuids_data_path)
    N_Cliffords = data.m1_3.j()
    z = data.m1_2()

    # Fit using least squares
    fit_lsq = get_fidelity_from_least_squares(N_Cliffords, z, amp_estimation[n], offset_estimation[n],
                                             0.99, 0.999)
    F_gate_lsq = fit_lsq['f_gate']
    F_err_lsq = fit_lsq['f_gate_err']
    amp_lsq = fit_lsq['amp']
    offset_lsq = fit_lsq['offset']

    # Fit using MLE
    p0 = [amp_estimation[n], offset_estimation[n], decay_constant_estimation[n]]
    fit = fit_rb_mle(N_Cliffords, z.T, initial_guess=p0)
    F_gate_mle = fit['f_gate']
    F_err_mle = fit['f_err']
    amp_mle = fit['amp']
    offset_mle = fit['offset']

    fidelities_and_errors['F_lsq'].append(get_value_with_error(F_gate_lsq, F_err_lsq))
    fidelities_and_errors['F_mle'].append(get_value_with_error(F_gate_mle, F_err_mle))
    fidelities_and_errors['F_mle_round'].append(np.round(F_gate_mle, 3))
    fidelities_and_errors['A_lsq'].append(np.round(amp_lsq, 3))
    fidelities_and_errors['Offset_lsq'].append(np.round(offset_lsq, 3))
    fidelities_and_errors['A_mle'].append(np.round(amp_mle, 3))
    fidelities_and_errors['Offset_mle'].append(np.round(offset_mle, 3))

# convert to DataFrame
fidelities_df = pd.DataFrame(fidelities_and_errors)
fidelities_df = fidelities_df.set_index('Qubit')
print(fidelities_df)


# %% plot data with 99% and 99.9% fit

amp_estimation = [-0.22, -0.33, 0.34, -0.24, -0.33, -0.31, -0.42, -0.34, -0.29, -0.25]
offset_estimation = [0.41, 0.39, 0.57, 0.47, 0.44, 0.40, 0.47, 0.40, 0.27, 0.44]

fidelity_estimation = [0.998, 0.996, 0.995, 0.996, 0.995, 0.997, 0.996, 0.997, 0.994, 0.994]
decay_constant_estimation = list(1-4*(1-np.array(fidelity_estimation)))
decay_constant_99 = 1-4*(1-0.99)

for n, (qubit, uuid) in enumerate(uuid_qubits.items()):
    print('----------')
    print(f'Qubit {qubit}')

    data = load_hdf5_uuid(uuid, uuids_data_path)

    f_rabi_MHz = 1e3 / (4 * data.snapshot['measurement']['sequence']['settings'][f'q{qubit[1:]}']['x90']['t_pulse'])
    print(f'f_Rabi = {f_rabi_MHz:.2f} MHz')

    x, x_label, x_unit = data.m1_3.i(), data.m1_3.i.label, data.m1_3.i.unit
    y, y_label, y_unit = data.m1_3.j(), data.m1_3.j.label, data.m1_3.j.unit
    z = data.m1_2()

    # averaging over randomisations
    N_Cliffords = y
    z_avrg = np.average(z, axis=0)
    z_std = np.std(z, axis=0)

    # Fit the data
    popt, pcov = curve_fit(RB_decay, N_Cliffords, z_avrg, sigma=z_std,
                           p0=[amp_estimation[n], offset_estimation[n], decay_constant_estimation[n]], maxfev=int(1e6))

    popt_99, pcov_99 = curve_fit(RB_decay, N_Cliffords, z_avrg, sigma=z_std,
                                   p0=[amp_estimation[n], offset_estimation[n], decay_constant_99],
                                   bounds=([amp_estimation[n]-0.3, offset_estimation[n]-0.3, 0.99999*decay_constant_99],
                                           [amp_estimation[n]+0.3, offset_estimation[n]+0.3, 1.00001*decay_constant_99]),
                                   maxfev=int(1e6)
                                   )

    # popt_999, pcov_999 = curve_fit(RB_decay, N_Cliffords, z_avrg, sigma=z_std,
    #                                   p0=[amp_estimation[n], offset_estimation[n], 0.999],
    #                                     bounds=([amp_estimation[n]-0.3, offset_estimation[n]-0.3, 0.99899999],
    #                                             [amp_estimation[n]+0.3, offset_estimation[n]+0.3, 0.99900001]),
    #                                     maxfev=int(1e6))

    # Extract fitted parameters
    amp_fit, offset_fit, p_fit = popt
    amp_err, offset_err, p_err = np.sqrt(np.diag(pcov))
    print(f'A_fit = {amp_fit:.3f}({amp_err:.3f}), ')
    print(f'offset = {offset_fit:.3f}({offset_err:.3f}), ')
    print(f'p = {p_fit:.3f}({p_err:.3f})')

    P_9 = 1 - 4 * (1 - 0.9)
    P_99 = 1 - 4 * (1 - 0.99)
    P_999 = 1 - 4 * (1 - 0.999)
    P_9999 = 1 - 4 * (1 - 0.9999)

    std_dev = np.std(z_std) * 1e2

    fig, ax = plt.subplots(1, 1, figsize=cm2inch(4.8, 4.2))
    
    if max(N_Cliffords) < 300:
        N_Cliffords_fit = np.arange(0, 301, 1)
    else:
        N_Cliffords_fit = N_Cliffords

    if qubit == 'Q3':
        ax.plot(N_Cliffords_fit, 1 - RB_decay(N_Cliffords_fit, *popt_99), color='black',
                linestyle='--', lw=1,
                label=f'$F_{{gate}} = 0.99$')
        # ax.plot(N_Cliffords_fit, 1 - RB_decay(N_Cliffords_fit, *popt_999), color='black',
        #         linestyle=':', lw=1,
        #         label=f'$F_{{gate}} = {get_value_with_error(*get_gate_fidelity_from_decay(popt_999[2], np.sqrt(pcov_999[2,2])))})$')
        ax.fill_between(N_Cliffords, 1 - (z_avrg - z_std), 1 - (z_avrg + z_std), color='tab:blue',
                        alpha=0.2)
        ax.scatter(N_Cliffords, 1 - z_avrg, s=2, color='tab:blue', alpha=1)
        ax.plot(N_Cliffords_fit, 1 - RB_decay(N_Cliffords_fit, *popt), color='orangered',
                label=f'$F_{{gate}} = {get_value_with_error(*get_gate_fidelity_from_decay(popt[2], np.sqrt(pcov[2,2])))}$')
    else:
        ax.plot(N_Cliffords_fit, RB_decay(N_Cliffords_fit, *popt_99), color='black',
                linestyle='--', lw=1,
                label=f'$F_{{gate}} = 0.99$'
                )
        # ax.plot(N_Cliffords_fit, RB_decay(N_Cliffords_fit, *popt_999), color='black',
        #         linestyle=':', lw=1,
        #         label=f'$F_{{gate}} = 0.999$'
        #         )
        ax.fill_between(N_Cliffords, z_avrg - z_std, z_avrg + z_std, color='tab:blue',
                        alpha=0.2)
        ax.scatter(N_Cliffords, z_avrg, s=2., color='tab:blue', alpha=1)
        ax.plot(N_Cliffords_fit, RB_decay(N_Cliffords_fit, *popt), color='orangered',
                label=f'$F_{{gate}} = {F_gate:.4f}({F_err:.0f})$')

    ax.set_xlabel('Number of Cliffords')
    ax.set_ylabel('$P_{even}$')
    ax.set_title(f'Qubit {qubit} ')
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(fig_path, f'RB_{qubit}_99.png'), dpi=300)
    fig.savefig(os.path.join(fig_path, f'RB_{qubit}_99.pdf'), dpi=300, transparent=True)
    plt.show()

# %% legend

fig, ax = plt.subplots(1, 1, figsize = cm2inch(14,8))

ax.fill_between(N_Cliffords, RB_decay(N_Cliffords, *popt_99), RB_decay(N_Cliffords, *popt_99),
                color='grey',
                alpha=0.2,
                label=f'$0.9 < F_{{gate}} < 0.99$')
ax.fill_between(N_Cliffords, RB_decay(N_Cliffords, *popt_99), RB_decay(N_Cliffords, *popt_999),
                color='grey',
                alpha=0.4,
                label=f'$0.99 < F_{{gate}} < 0.999$')
ax.fill_between(N_Cliffords, RB_decay(N_Cliffords, *popt_999), RB_decay(N_Cliffords, *popt_999),
                color='grey',
                alpha=0.6,
                label=f'$0.999 < F_{{gate}} < 0.9999$')
ax.plot(N_Cliffords, RB_decay(N_Cliffords, *popt_99), color='black',
        linestyle='--', lw=1,
        label=f'$F_{{gate}} = 0.99$'
        )
ax.plot(N_Cliffords, RB_decay(N_Cliffords, *popt_999), color='black',
        linestyle=':', lw=1,
        label=f'$F_{{gate}} = 0.999$'
        )

ax.plot(N_Cliffords, RB_decay(N_Cliffords, *popt), color='orangered',
        label=f'Fitted gate fidelity')

ax.fill_between(N_Cliffords, z_avrg - z_std, z_avrg + z_std, color='tab:blue',
                alpha=0.2,
                label='RB data points $\pm$ std deviation')
ax.scatter(N_Cliffords, z_avrg, s=2., color='tab:blue', alpha=1,
           label='RB data points')
plt.axis('off')

plt.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.8), fontsize=8)

plt.tight_layout()
fig.savefig(os.path.join(fig_path, f'RB_{qubit}_legend.pdf'), dpi=300, transparent=True)

plt.show()

