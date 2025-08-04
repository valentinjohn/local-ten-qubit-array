# %% Path management

from pathlib import Path
from config import DATA_DIR

try:
    script_dir = Path(__file__).resolve().parent
except NameError:
    script_dir = Path().resolve() / 'theory'
fig_path = script_dir / "images"

# %% imports

import numpy as np
import matplotlib.pyplot as plt
import utils.analysis_tools as tools
from utils.common_imports import one_column_width_cm, two_column_width_cm

# %%

theta = np.linspace(85,95,101)
phi = np.linspace(0,360,101)

n = 3 # choose either 1 or 3 depending on which map you want to load.
Full_data = {}
for n in [1, 3]:
    data_path = DATA_DIR / f'{n}ParticleMap.dat'
    Full_data[f'{n}_holes'] = np.loadtxt(data_path)
c = 2 # choose the column you want to plot (0, 1 are theta and phi, so those can be ignored)

# %%

fig, axes = plt.subplots(1, 4, figsize=tools.cm2inch(16, 4), sharex=True, sharey=True)

cmap = plt.get_cmap('Blues')
norm = plt.Normalize(0, 0.606906491)

for n, m in enumerate(['1_holes', '3_holes']):
    c = 2 # corresponding to the plunger drive efficiency
    data = Full_data[m][:, c].reshape(len(theta), len(phi))
    print(max(data.flatten()))
    axes[n].imshow(data, extent=[phi[0], phi[-1], theta[0], theta[-1]],
                   aspect='auto', cmap=cmap, norm=norm)
    axes[n].set_xlabel(r'$\phi$ (deg)')
    axes[n].set_xticks([0, 90, 180, 270, 360])
    axes[n].set_title(rf'$Qn_{{{m[0]}h}}$ ($P_n$)')

    c = 10 # corresponding to the fR_North-West [MHz/mV]
    data = Full_data[m][:, c].reshape(len(theta), len(phi))
    print(max(data.flatten()))
    axes[n+2].imshow(data, extent=[phi[0], phi[-1], theta[0], theta[-1]],
                   aspect='auto', cmap=cmap, norm=norm)
    axes[n+2].set_xlabel(r'$\phi$ (deg)')
    axes[n+2].set_xticks([0, 90, 180, 270, 360])
    axes[n+2].set_title(rf'$Qn_{{{m[0]}h}}$ ($B_{{NW}}$)')

axes[0].set_ylabel(r'$\theta$ (deg)')

plt.tight_layout()

plt.savefig(fig_path / 'drive_eff_field.pdf', dpi=300, transparent=True)
plt.show()

# %%
from utils.utils import save_colorbar

filename = fig_path / 'drive_eff_field_cbar.pdf'
unit_label = '$f_{\mathrm{Rabi}}/A$ (MHz/mV)'
save_colorbar(norm, cmap, unit_label, filename, orientation='vertical', figsize=tools.cm2inch(3, 2.7),
              ticklabelsize=None, labelsize=None, shrink=1, tick_number=3, location='right', labelpad=None)


fig, axes = plt.subplots(5, 4, figsize=tools.cm2inch(16, 20), sharex=True, sharey=True)

cmap = plt.get_cmap('Blues')
norm = plt.Normalize(0, 0.606906491)
cmap_lses = plt.get_cmap('bwr')
norm_lses = plt.Normalize(-20.2787044, 20.2787044)
# cmap_lses = plt.get_cmap('Blues')
# norm_lses = plt.Normalize(0, 20.2787044**2)

for n, m in enumerate(['1_holes', '3_holes']):
    drive_eff_assignment = {2: rf'$Qn_{{{m[0]}h}}$ ($P_n$)',
                            4: rf'$Qn_{{{m[0]}h}}$ ($B_{{NW}}$)',
                            6: rf'$Qn_{{{m[0]}h}}$ ($B_{{NE}}$)',
                            8: rf'$Qn_{{{m[0]}h}}$ ($B_{{SW}}$)',
                            10: rf'$Qn_{{{m[0]}h}}$ ($B_{{SE}}$)'}
    for i, (c, title) in enumerate(drive_eff_assignment.items()):
        data = Full_data[m][:, c].reshape(len(theta), len(phi))
        print(max(data.flatten()))
        axes[i][n+2].imshow(data, extent=[phi[0], phi[-1], theta[0], theta[-1]],
                       aspect='auto', cmap=cmap, norm=norm)
        axes[i][n+2].set_xticks([0, 90, 180, 270, 360])
        axes[i][n+2].set_title(title)

    lses_assignment = {3: rf'$Qn_{{{m[0]}h}}$ ($P_n$)',
                       5: rf'$Qn_{{{m[0]}h}}$ ($B_{{NW}}$)',
                       7: rf'$Qn_{{{m[0]}h}}$ ($B_{{NE}}$)',
                       9: rf'$Qn_{{{m[0]}h}}$ ($B_{{SW}}$)',
                       11: rf'$Qn_{{{m[0]}h}}$ ($B_{{SE}}$)'}
    for i, (c, title) in enumerate(lses_assignment.items()):
        data = Full_data[m][:, c].reshape(len(theta), len(phi))
        print(max(data.flatten()))
        axes[i][n].imshow(data, extent=[phi[0], phi[-1], theta[0], theta[-1]],
                       aspect='auto', cmap=cmap_lses, norm=norm_lses)
        axes[i][n].set_xticks([0, 90, 180, 270, 360])
        axes[i][n].set_title(title)

for n in range(5):
    axes[n][0].set_ylabel(r'$\theta$ (deg)')
for n in range(4):
    axes[4][n].set_xlabel(r'$\phi$ (deg)')

plt.tight_layout()

plt.savefig(fig_path / 'drive_eff_field.pdf', dpi=300, transparent=True)
plt.show()

# %% LSES sum

xi_square = {}
xi = {}
plunger_eff = {}
quality_factor = {}

for n, m in enumerate(['1_holes', '3_holes']):
    xi_square[m] = 0
    for key in [3, 5, 7, 9, 11]:
        xi_square[m] += Full_data[m][:, key].reshape(len(theta), len(phi))**2
    xi[m] = np.sqrt(xi_square[m])
    plunger_eff[m] = Full_data[m][:, 2].reshape(len(theta), len(phi))
    quality_factor[m] = plunger_eff[m] / xi[m]

cmap_name = 'viridis'

norm_xi = plt.Normalize(0, max([max(xi['1_holes'].flatten()), max(xi['3_holes'].flatten())]))
cmap_xi = plt.get_cmap(cmap_name)

norm_plunger_eff = plt.Normalize(0, max([max(plunger_eff['1_holes'].flatten()), max(plunger_eff['3_holes'].flatten())]))
cmap_plunger_eff = plt.get_cmap(cmap_name)

norm_quality_factor = plt.Normalize(0, max([max(quality_factor['1_holes'].flatten()), max(quality_factor['3_holes'].flatten())]))
cmap_quality_factor = plt.get_cmap(cmap_name)

# cmap_xi = parula_cmap
# cmap_plunger_eff = parula_cmap
# cmap_quality_factor = parula_cmap

fig, axes = plt.subplots(2, 4, figsize=tools.cm2inch(16, 8), sharex='col')

for n, m in enumerate(['1_holes', '3_holes']):
    axes[n][0].imshow(xi[m], extent=[phi[0], phi[-1], theta[0], theta[-1]],
                     aspect='auto', cmap=cmap_xi, norm=norm_xi)
    if n == 1:
        axes[n][0].set_xlabel(r'$\phi$ (deg)')
    axes[n][0].set_xticks([0, 90, 180, 270, 360])
    axes[n][0].set_title(rf'$\xi_{{{m[0]}h}}$')
    axes[n][0].set_ylabel(r'$\theta$ (deg)')

    axes[n][1].imshow(plunger_eff[m], extent=[phi[0], phi[-1], theta[0], theta[-1]],
                   aspect='auto', cmap=cmap_plunger_eff, norm=norm_plunger_eff)
    if n == 1:
        axes[n][1].set_xlabel(r'$\phi$ (deg)')
    axes[n][1].set_xticks([0, 90, 180, 270, 360])
    axes[n][1].set_title(rf'$f_{{Rabi,{m[0]}h}}$')
    axes[n][1].yaxis.set_ticklabels([])

    axes[n][2].imshow(quality_factor[m], extent=[phi[0], phi[-1], theta[0], theta[-1]],
                     aspect='auto', cmap=cmap_quality_factor, norm=norm_quality_factor)
    if n == 1:
        axes[n][2].set_xlabel(r'$\phi$ (deg)')
    axes[n][2].set_xticks([0, 90, 180, 270, 360])
    axes[n][2].set_title(rf'$Q_{{{m[0]}h}}$')
    axes[n][2].yaxis.set_ticklabels([])

    axes[n][3].plot(theta, quality_factor[m].max(axis=1), label='max')
    axes[n][3].plot(theta, quality_factor[m].mean(axis=1), label='mean')
    axes[n][3].plot(theta, quality_factor[m].min(axis=1), label='min')
    if n == 1:
        axes[n][3].set_xlabel(r'$\theta$ (deg)')
    axes[n][3].set_ylabel(rf'$Q_{{{m[0]}h}}$')
    axes[n][3].yaxis.tick_right()
    axes[n][3].yaxis.set_label_position("right")
    axes[n][3].set_ylim(0, 0.2)
    if m == '3_holes':
        axes[n][3].vlines(87, 0, quality_factor[m].max(), color='black', ls='--', lw=1)
        axes[n][3].vlines(88, 0, quality_factor[m].max(), color='black', ls=':', lw=1)
        axes[n][3].vlines(92, 0, quality_factor[m].max(), color='black', ls=':', lw=1)
        axes[n][3].vlines(93, 0, quality_factor[m].max(), color='black', ls='--', lw=1)
    else:
        axes[n][3].vlines(87, 0, quality_factor[m].max(), color='black', ls='--', lw=1)
        axes[n][3].vlines(88, 0, quality_factor[m].max(), color='black', ls=':', lw=1)
        axes[n][3].vlines(92, 0, quality_factor[m].max(), color='black', ls=':', lw=1)
        axes[n][3].vlines(93, 0, quality_factor[m].max(), color='black', ls='--', lw=1)
        axes[n][3].legend()
    # axes[n][3].set_title(rf'$Q_{{{m[0]}h}}$')

for m in range(2):
    for n in range(0, 3):
        axes[m][n].hlines(92, 0, 360, color='white', ls=':', lw=0.5)
        axes[m][n].hlines(93, 0, 360, color='white', ls='--', lw=0.5)
        axes[m][n].hlines(88, 0, 360, color='white', ls=':', lw=0.5, label='2 deg out-of-plane')
        axes[m][n].hlines(87, 0, 360, color='white', ls='--', lw=0.5, label='3 deg out-of-plane')
# axes[1][2].legend()

plt.tight_layout()
plt.savefig(fig_path / 'drive_eff_field_xi_f_Rabi_Q.pdf', dpi=300, transparent=True)
plt.show()

figsize = tools.cm2inch(2.5, 2.5)
unit_label = r'$\xi$ ($10^{-4}$/mV)'
filename = fig_path / 'drive_eff_field_xi_cbar.pdf'
save_colorbar(norm_xi, cmap_xi, unit_label, filename, orientation='vertical', figsize=figsize,
              ticklabelsize=None, labelsize=None, shrink=1, tick_number=3, location='right', labelpad=None)

unit_label = r'$f_{\mathrm{R}}/A$ (MHz/mV)'
filename = fig_path / 'drive_eff_field_f_Rabi_cbar.pdf'
save_colorbar(norm_plunger_eff, cmap_plunger_eff, unit_label, filename, orientation='vertical', figsize=figsize,
              ticklabelsize=None, labelsize=None, shrink=1, tick_number=3, location='right', labelpad=None)

unit_label = r'$Q$ (MHz)'
filename = fig_path / 'drive_eff_field_Q_cbar.pdf'
save_colorbar(norm_quality_factor, cmap_quality_factor, unit_label, filename, orientation='vertical', figsize=figsize,
              ticklabelsize=None, labelsize=None, shrink=1, tick_number=3, location='right', labelpad=None)


# %% min and max value for each line

fig, axes = plt.subplots(2, 3, figsize=tools.cm2inch(12, 7), sharex='col')

for n, m in enumerate(['1_holes', '3_holes']):
    axes[n][0].plot(theta, xi[m].max(axis=1), label='max')
    axes[n][0].plot(theta, xi[m].mean(axis=1), label='mean')
    axes[n][0].plot(theta, xi[m].min(axis=1), label='min')
    if n == 1:
        axes[n][0].set_xlabel(r'$\phi$ (deg)')
    axes[n][0].set_title(rf'$\xi_{{{m[0]}h}}$')
    axes[n][0].set_ylabel(r'$\theta$ (deg)')
    axes[n][0].legend()

    axes[n][1].plot(theta, plunger_eff[m].max(axis=1), label='max')
    axes[n][1].plot(theta, plunger_eff[m].mean(axis=1), label='mean')
    axes[n][1].plot(theta, plunger_eff[m].min(axis=1), label='min')
    if n == 1:
        axes[n][1].set_xlabel(r'$\phi$ (deg)')
    axes[n][1].set_title(rf'$f_{{Rabi,{m[0]}h}}$')

    axes[n][2].plot(theta, quality_factor[m].max(axis=1), label='max')
    axes[n][2].plot(theta, quality_factor[m].mean(axis=1), label='mean')
    axes[n][2].plot(theta, quality_factor[m].min(axis=1), label='min')
    if n == 1:
        axes[n][2].set_xlabel(r'$\phi$ (deg)')
    axes[n][2].set_title(rf'$Q_{{{m[0]}h}}$')

    ymax = quality_factor[m].flatten().max()
    axes[n][2].vlines(92, 0, ymax, color='black', ls=':', lw=1)
    axes[n][2].vlines(93, 0, ymax, color='black', ls=':', lw=1)

plt.tight_layout()
# plt.savefig(fig_path / 'drive_eff_field_xi_f_Rabi_Q.pdf', dpi=300, transparent=True)
plt.show()

# %% Xi, f_Rabi barrier, f_Rabi plunger, Q barrier, Q plunger

xi_square = {}
xi = {}
plunger_eff = {}
barrier_eff = {'1_holes': Full_data['1_holes'][:, 6].reshape(len(theta), len(phi)),
               '3_holes': Full_data['3_holes'][:, 6].reshape(len(theta), len(phi))}
quality_factor = {}
quality_factor_bar = {}

for n, m in enumerate(['1_holes', '3_holes']):
    xi_square[m] = 0
    for key in [3, 5, 7, 9, 11]:
        xi_square[m] += Full_data[m][:, key].reshape(len(theta), len(phi))**2
    xi[m] = np.sqrt(xi_square[m])
    plunger_eff[m] = Full_data[m][:, 2].reshape(len(theta), len(phi))
    quality_factor[m] = plunger_eff[m] / xi[m]
    quality_factor_bar[m] = barrier_eff[m] / xi[m]

cmap_name = 'viridis'

norm_xi = plt.Normalize(0, max([max(xi['1_holes'].flatten()), max(xi['3_holes'].flatten())]))
cmap_xi = plt.get_cmap(cmap_name)

norm_plunger_eff = plt.Normalize(0, max([max(plunger_eff['1_holes'].flatten()), max(plunger_eff['3_holes'].flatten())]))
cmap_plunger_eff = plt.get_cmap(cmap_name)

norm_quality_factor = plt.Normalize(0, max([max(quality_factor['1_holes'].flatten()), max(quality_factor['3_holes'].flatten())]))
cmap_quality_factor = plt.get_cmap(cmap_name)

# cmap_xi = parula_cmap
# cmap_plunger_eff = parula_cmap
# cmap_quality_factor = parula_cmap

fig, axes = plt.subplots(2, 5, figsize=tools.cm2inch(16, 8), sharex='col')

y_lim = 0.2
for n, m in enumerate(['1_holes', '3_holes']):
    axes[n][0].imshow(xi[m], extent=[phi[0], phi[-1], theta[0], theta[-1]],
                     aspect='auto', cmap=cmap_xi, norm=norm_xi)
    if n == 1:
        axes[n][0].set_xlabel(r'$\phi$ (deg)')
    axes[n][0].set_xticks([0, 90, 180, 270, 360])
    axes[n][0].set_title(rf'$\xi_{{{m[0]}h}}$')
    axes[n][0].set_ylabel(r'$\theta$ (deg)')

    axes[n][1].imshow(plunger_eff[m], extent=[phi[0], phi[-1], theta[0], theta[-1]],
                   aspect='auto', cmap=cmap_plunger_eff, norm=norm_plunger_eff)
    if n == 1:
        axes[n][1].set_xlabel(r'$\phi$ (deg)')
    axes[n][1].set_xticks([0, 90, 180, 270, 360])
    axes[n][1].set_title(rf'$f_{{Rabi,{m[0]}h, plunger}}$')
    axes[n][1].yaxis.set_ticklabels([])

    axes[n][2].imshow(barrier_eff[m], extent=[phi[0], phi[-1], theta[0], theta[-1]],
                   aspect='auto', cmap=cmap_plunger_eff, norm=norm_plunger_eff)
    if n == 1:
        axes[n][2].set_xlabel(r'$\phi$ (deg)')
    axes[n][2].set_xticks([0, 90, 180, 270, 360])
    axes[n][2].set_title(rf'$f_{{Rabi,{m[0]}h, barrier}}$')
    axes[n][2].yaxis.set_ticklabels([])

    # axes[n][3].plot(theta, quality_factor[m].max(axis=1), label='max')
    # axes[n][3].plot(theta, quality_factor[m].mean(axis=1), label='mean')
    # axes[n][3].plot(theta, quality_factor[m].min(axis=1), label='min')
    # if n == 1:
    #     axes[n][3].set_xlabel(r'$\theta$ (deg)')
    # axes[n][3].yaxis.tick_right()
    # axes[n][3].yaxis.set_label_position("right")
    # axes[n][3].yaxis.set_ticklabels([])
    # axes[n][3].set_ylim(0, 0.2)
    # axes[n][3].vlines(87, 0, y_lim, color='black', ls='--', lw=1)
    # axes[n][3].vlines(88, 0, y_lim, color='black', ls=':', lw=1)
    # axes[n][3].vlines(92, 0, y_lim, color='black', ls=':', lw=1)
    # axes[n][3].vlines(93, 0, y_lim, color='black', ls='--', lw=1)
    # if m == '1_holes':
    #     axes[n][3].legend(frameon=True)
    # axes[n][3].set_title(rf'$Q_{{{m[0]}h, plunger}}$')
    axes[n][3].imshow(quality_factor[m], extent=[phi[0], phi[-1], theta[0], theta[-1]],
                   aspect='auto', cmap=cmap_plunger_eff, norm=norm_plunger_eff)
    if n == 1:
        axes[n][3].set_xlabel(r'$\phi$ (deg)')
    axes[n][3].set_xticks([0, 90, 180, 270, 360])
    axes[n][3].set_title(rf'$Q_{{{m[0]}h, plunger}}$')
    axes[n][3].yaxis.set_ticklabels([])

    # axes[n][4].plot(theta, quality_factor_bar[m].max(axis=1), label='max')
    # axes[n][4].plot(theta, quality_factor_bar[m].mean(axis=1), label='mean')
    # axes[n][4].plot(theta, quality_factor_bar[m].min(axis=1), label='min')
    # if n == 1:
    #     axes[n][4].set_xlabel(r'$\theta$ (deg)')
    # axes[n][4].set_ylabel(rf'$Q_{{{m[0]}h}}$')
    # axes[n][4].yaxis.tick_right()
    # axes[n][4].yaxis.set_label_position("right")
    # axes[n][4].set_ylim(0, y_lim)
    # axes[n][4].vlines(87, 0, y_lim, color='black', ls='--', lw=1)
    # axes[n][4].vlines(88, 0, y_lim, color='black', ls=':', lw=1)
    # axes[n][4].vlines(92, 0, y_lim, color='black', ls=':', lw=1)
    # axes[n][4].vlines(93, 0, y_lim, color='black', ls='--', lw=1)
    # axes[n][4].set_title(rf'$Q_{{{m[0]}h, barrier}}$')
    axes[n][4].imshow(quality_factor_bar[m], extent=[phi[0], phi[-1], theta[0], theta[-1]],
                   aspect='auto', cmap=cmap_plunger_eff, norm=norm_plunger_eff)
    if n == 1:
        axes[n][4].set_xlabel(r'$\phi$ (deg)')
    axes[n][4].set_xticks([0, 90, 180, 270, 360])
    axes[n][4].set_title(rf'$Q_{{{m[0]}h, barrier}}$')
    axes[n][4].yaxis.set_ticklabels([])


for m in range(2):
    for n in range(0, 5):
        axes[m][n].hlines(92, 0, 360, color='white', ls=':', lw=0.5)
        axes[m][n].hlines(93, 0, 360, color='white', ls='--', lw=0.5)
        axes[m][n].hlines(88, 0, 360, color='white', ls=':', lw=0.5, label='2 deg out-of-plane')
        axes[m][n].hlines(87, 0, 360, color='white', ls='--', lw=0.5, label='3 deg out-of-plane')
# axes[1][2].legend()

plt.tight_layout()
plt.savefig(fig_path / 'drive_eff_field_xi_f_Rabi_pl_bar_Q.pdf', dpi=300, transparent=True)
plt.show()

# %% plot only quality factor

fig, axes = plt.subplots(1, 4, figsize=tools.cm2inch(12, 4), sharey=True)

y_lim = 0.2
for n, m in enumerate(['1_holes', '3_holes']):
    axes[n].plot(theta, quality_factor[m].max(axis=1), label='max')
    axes[n].plot(theta, quality_factor[m].mean(axis=1), label='mean')
    axes[n].plot(theta, quality_factor[m].min(axis=1), label='min')
    axes[n].set_xlabel(r'$\theta$ (deg)')
    axes[n].set_ylim(0, 0.2)
    axes[n].vlines(87, 0, y_lim, color='black', ls='--', lw=1)
    axes[n].vlines(88, 0, y_lim, color='black', ls=':', lw=1)
    axes[n].vlines(92, 0, y_lim, color='black', ls=':', lw=1)
    axes[n].vlines(93, 0, y_lim, color='black', ls='--', lw=1)
    if m == '1_holes':
        axes[n].legend(frameon=True)
    axes[n].set_title(rf'$Q_{{{m[0]}h, plunger}}$')

    axes[n+2].plot(theta, quality_factor_bar[m].max(axis=1), label='max')
    axes[n+2].plot(theta, quality_factor_bar[m].mean(axis=1), label='mean')
    axes[n+2].plot(theta, quality_factor_bar[m].min(axis=1), label='min')
    axes[n+2].set_xlabel(r'$\theta$ (deg)')
    axes[n+2].set_ylim(0, y_lim)
    axes[n+2].vlines(87, 0, y_lim, color='black', ls='--', lw=1)
    axes[n+2].vlines(88, 0, y_lim, color='black', ls=':', lw=1)
    axes[n+2].vlines(92, 0, y_lim, color='black', ls=':', lw=1)
    axes[n+2].vlines(93, 0, y_lim, color='black', ls='--', lw=1)
    axes[n+2].set_title(rf'$Q_{{{m[0]}h, barrier}}$')

axes[0].set_ylabel(rf'$Q$ (MHz)')
plt.tight_layout()
plt.savefig(fig_path / 'drive_eff_field_quality_factor.pdf', dpi=300, transparent=True)
plt.show()

# %% plot only Rabi drive efficiency

fig, axes = plt.subplots(1, 4, figsize=tools.cm2inch(12, 4), sharey=True)

y_lim = 0.63
for n, m in enumerate(['1_holes', '3_holes']):
    axes[n].plot(theta, plunger_eff[m].max(axis=1), label='max')
    axes[n].plot(theta, plunger_eff[m].mean(axis=1), label='mean')
    axes[n].plot(theta, plunger_eff[m].min(axis=1), label='min')
    axes[n].set_xlabel(r'$\theta$ (deg)')
    axes[n].set_ylim(0, y_lim)
    axes[n].vlines(87, 0, y_lim, color='black', ls='--', lw=1)
    axes[n].vlines(88, 0, y_lim, color='black', ls=':', lw=1)
    axes[n].vlines(92, 0, y_lim, color='black', ls=':', lw=1)
    axes[n].vlines(93, 0, y_lim, color='black', ls='--', lw=1)
    if m == '1_holes':
        axes[n].legend(frameon=True)
    axes[n].set_title(rf'$f_{{Rabi, {m[0]}h, plunger}}$')

    axes[n+2].plot(theta, barrier_eff[m].max(axis=1), label='max')
    axes[n+2].plot(theta, barrier_eff[m].mean(axis=1), label='mean')
    axes[n+2].plot(theta, barrier_eff[m].min(axis=1), label='min')
    axes[n+2].set_xlabel(r'$\theta$ (deg)')
    axes[n+2].set_ylim(0, y_lim)
    axes[n+2].vlines(87, 0, y_lim, color='black', ls='--', lw=1)
    axes[n+2].vlines(88, 0, y_lim, color='black', ls=':', lw=1)
    axes[n+2].vlines(92, 0, y_lim, color='black', ls=':', lw=1)
    axes[n+2].vlines(93, 0, y_lim, color='black', ls='--', lw=1)
    axes[n+2].set_title(rf'$f_{{Rabi, {m[0]}h, barrier}}$')

axes[0].set_ylabel(r'$f_{Rabi}$ (MHz/mV)')
plt.tight_layout()
plt.savefig(fig_path / 'drive_eff_field_f_rabi.pdf', dpi=300, transparent=True)
plt.show()

# %% plot only xi

fig, axes = plt.subplots(1, 2, figsize=tools.cm2inch(8, 4), sharey=True)

y_lim = 30 #0.63

for n, m in enumerate(['1_holes', '3_holes']):
    axes[n].plot(theta, xi[m].max(axis=1), label='max')
    axes[n].plot(theta, xi[m].mean(axis=1), label='mean')
    axes[n].plot(theta, xi[m].min(axis=1), label='min')
    axes[n].set_xlabel(r'$\theta$ (deg)')
    axes[n].set_ylim(0, y_lim)
    axes[n].vlines(87, 0, y_lim, color='black', ls='--', lw=1)
    axes[n].vlines(88, 0, y_lim, color='black', ls=':', lw=1)

    axes[n].vlines(92, 0, y_lim, color='black', ls=':', lw=1)
    axes[n].vlines(93, 0, y_lim, color='black', ls='--', lw=1)
    if m == '1_holes':
        axes[n].legend(frameon=True)
    axes[n].set_title(rf'$\xi_{{{m[0]}h}}$')

axes[0].set_ylabel(r'$\xi$ ($10^{-4}$/mV)')
plt.tight_layout()
plt.savefig(fig_path / 'drive_eff_field_xi.pdf', dpi=300, transparent=True)
plt.show()

# %% Xi, f_Rabi barrier, f_Rabi plunger
xi_square = {}
xi = {}
plunger_eff = {}
barrier_eff = {'1_holes': Full_data['1_holes'][:, 6].reshape(len(theta), len(phi)),
               '3_holes': Full_data['3_holes'][:, 6].reshape(len(theta), len(phi))}
quality_factor = {}
quality_factor_bar = {}

for n, m in enumerate(['1_holes', '3_holes']):
    xi_square[m] = 0
    for key in [3, 5, 7, 9, 11]:
        xi_square[m] += Full_data[m][:, key].reshape(len(theta), len(phi))**2
    xi[m] = np.sqrt(xi_square[m])
    plunger_eff[m] = Full_data[m][:, 2].reshape(len(theta), len(phi))
    quality_factor[m] = plunger_eff[m] / xi[m]
    quality_factor_bar[m] = barrier_eff[m] / xi[m]

cmap_name = 'viridis'

norm_xi = plt.Normalize(0, max([max(xi['1_holes'].flatten()), max(xi['3_holes'].flatten())]))
cmap_xi = plt.get_cmap(cmap_name)

norm_plunger_eff = plt.Normalize(0, max([max(plunger_eff['1_holes'].flatten()), max(plunger_eff['3_holes'].flatten())]))
cmap_plunger_eff = plt.get_cmap(cmap_name)

norm_quality_factor = plt.Normalize(0, max([max(quality_factor['1_holes'].flatten()), max(quality_factor['3_holes'].flatten())]))
cmap_quality_factor = plt.get_cmap(cmap_name)

fig, axes = plt.subplots(3, 2, figsize=tools.cm2inch(0.9*one_column_width_cm, 10.5), sharex=True, sharey=True)

y_lim = 0.2
for n, m in enumerate(['1_holes', '3_holes']):
    axes[0][n].imshow(xi[m], extent=[phi[0], phi[-1], theta[0], theta[-1]],
                     aspect='auto', cmap=cmap_xi, norm=norm_xi)
    axes[0][n].set_xticks([0, 90, 180, 270, 360])
    axes[0][n].set_title(rf'$\xi_{{\mathrm{{{m[0]}h}}}}$')

    axes[1][n].imshow(plunger_eff[m], extent=[phi[0], phi[-1], theta[0], theta[-1]],
                   aspect='auto', cmap=cmap_plunger_eff, norm=norm_plunger_eff)
    axes[1][n].set_xticks([0, 90, 180, 270, 360])
    axes[1][n].set_title(rf'$f^R_{{\mathrm{{{m[0]}h,P}}}}$')

    axes[2][n].imshow(barrier_eff[m], extent=[phi[0], phi[-1], theta[0], theta[-1]],
                   aspect='auto', cmap=cmap_plunger_eff, norm=norm_plunger_eff)
    axes[2][n].set_xlabel(r'$\phi$ (deg)')
    axes[2][n].set_xticks([0, 90, 180, 270, 360])
    axes[2][n].set_title(rf'$f^R_{{\mathrm{{{m[0]}h,B}}}}$')

for n in range(3):
    axes[n][0].set_ylabel(r'$\theta$ (deg)')

for m in range(3):
    for n in range(0, 2):
        axes[m][n].hlines(92, 0, 360, color='white', ls=':', lw=0.5)
        axes[m][n].hlines(93, 0, 360, color='white', ls='--', lw=0.5)
        axes[m][n].hlines(88, 0, 360, color='white', ls=':', lw=0.5, label='2 deg out-of-plane')
        axes[m][n].hlines(87, 0, 360, color='white', ls='--', lw=0.5, label='3 deg out-of-plane')
# axes[1][2].legend()

plt.tight_layout()
plt.savefig(fig_path / 'drive_eff_field_xi_f_Rabi_pl_bar.pdf', dpi=300, transparent=True)
plt.show()

# %% plot f_Rabi ratio between 3_holes and 1_holes

fig, axes = plt.subplots(1, 2, figsize=tools.cm2inch(8, 4), sharey=True)

norm = plt.Normalize(0,  15)

axes[0].imshow(plunger_eff['3_holes']/plunger_eff['1_holes'], extent=[phi[0], phi[-1], theta[0], theta[-1]],
               aspect='auto', cmap=cmap_plunger_eff, norm=norm)
axes[0].set_xticks([0, 90, 180, 270, 360])
axes[0].set_title(rf'$f^R_{{\mathrm{{3h,P}}}}/f^R_{{\mathrm{{1h,P}}}}$')
axes[1].imshow(barrier_eff['3_holes']/barrier_eff['1_holes'], extent=[phi[0], phi[-1], theta[0], theta[-1]],
               aspect='auto', cmap=cmap_plunger_eff, norm=norm)
axes[1].set_xlabel(r'$\phi$ (deg)')
axes[1].set_xticks([0, 90, 180, 270, 360])
axes[1].set_title(rf'$f^R_{{\mathrm{{3h,B}}}}/f^R_{{\mathrm{{1h,B}}}}$')
for n in range(2):
    axes[n].set_ylabel(r'$\theta$ (deg)')
    axes[n].hlines(92, 0, 360, color='white', ls=':', lw=0.5)
    axes[n].hlines(93, 0, 360, color='white', ls='--', lw=0.5)
    axes[n].hlines(88, 0, 360, color='white', ls=':', lw=0.5, label='2 deg out-of-plane')
    axes[n].hlines(87, 0, 360, color='white', ls='--', lw=0.5, label='3 deg out-of-plane')
# axes[1][2].legend()

# add colorbar to plot
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap_plunger_eff), cax=cbar_ax, location='left')
# cbar.set_label(r'$f^R_{\mathrm{3h}}/f^R_{\mathrm{1h}}$', rotation=270, labelpad=10)
cbar.ax.yaxis.set_ticks_position('left')
cbar.ax.yaxis.set_tick_params(labelsize=8)

# adjust layout to accommodate colorbar
plt.subplots_adjust(left=0.15, right=0.8, bottom=0.25, top=0.85, wspace=0.3, hspace=0.3)

# plt.tight_layout()
# plt.savefig(fig_path / 'drive_eff_field_f_Rabi_ratio.pdf', dpi=300, transparent=True)
plt.show()

# plot linecut at 92.5 deg
theta_linecut = 92.5
linecut_plunger_eff_1h = plunger_eff['1_holes'][np.abs(theta - theta_linecut).argmin(), :]
linecut_plunger_eff_3h = plunger_eff['3_holes'][np.abs(theta - theta_linecut).argmin(), :]
linecut_barrier_eff_1h = barrier_eff['1_holes'][np.abs(theta - theta_linecut).argmin(), :]
linecut_barrier_eff_3h = barrier_eff['3_holes'][np.abs(theta - theta_linecut).argmin(), :]

fig, axes = plt.subplots(1, 2, figsize=tools.cm2inch(8, 4))

axes[0].plot(phi, linecut_plunger_eff_3h / linecut_plunger_eff_1h, label=r'$f^R_{\mathrm{3h,P}}/f^R_{\mathrm{1h,P}}$')
axes[0].set_xlabel(r'$\phi$ (deg)')
axes[0].set_ylabel(r'$f^R_{\mathrm{3h,P}}/f^R_{\mathrm{1h,P}}$')
axes[0].set_title(rf'Linecut at $\theta = {theta_linecut:.1f}$ deg')

axes[1].plot(phi, linecut_barrier_eff_3h / linecut_barrier_eff_1h, label=r'$f^R_{\mathrm{3h,B}}/f^R_{\mathrm{1h,B}}$')
axes[1].set_xlabel(r'$\phi$ (deg)')
axes[1].set_ylabel(r'$f^R_{\mathrm{3h,B}}/f^R_{\mathrm{1h,B}}$')
axes[1].set_title(rf'Linecut at $\theta = {theta_linecut:.1f}$ deg')

plt.tight_layout()
plt.show()
