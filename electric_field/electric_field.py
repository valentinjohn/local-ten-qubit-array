import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from drive_locality import utils_nearest_neighbours as unn
from pathlib import Path
import os
from utils.analysis_tools import cm2inch

from config import DATA_DIR, PROJECT_DIR

script_dir = PROJECT_DIR / 'electric_field'
fig_path = script_dir / "images"
data_path = DATA_DIR

# %%
# Constants
epsilon_0 = 8.854187817e-12  # vacuum permittivity
sigma_0 = 10e-6  # surface charge density amplitude [C/m^2]

def get_e_field_from_disk(x, z, R_disk=130e-9):
    x = np.asarray(x, dtype=float)  # ensure x is a numpy array
    z = np.asarray(z, dtype=float)  # ensure z is a numpy array

    N_phi = 100  # angular discretization of disk
    N_r = 100  # radial discretization of disk

    X, Z = np.meshgrid(x, z)
    # Initialize field arrays
    Ex = np.zeros_like(X)
    Ez = np.zeros_like(Z)

    # Discretize using midpoint rule
    dr = R_disk / N_r
    dphi = 2 * np.pi / N_phi
    r_vals = np.linspace(dr/2, R_disk - dr/2, N_r)
    phi_vals = np.linspace(0, 2 * np.pi, N_phi, endpoint=False)

    # Loop over elements on the disk
    for r in r_vals:
        for phi in phi_vals:
            # Disk element position
            xq = r * np.cos(phi)
            yq = r * np.sin(phi)
            zq = 0.0  # disk lies in z = 0

            # Position vector from source to field point
            Rx = X - xq
            Ry = - yq  # since we're evaluating at y = 0
            Rz = Z - zq
            R = np.sqrt(Rx**2 + Ry**2 + Rz**2)

            # Avoid singularity at source
            R[R == 0] = np.finfo(float).eps

            # Differential area element in polar coords
            dA = r * dr * dphi

            # Coulomb’s Law for each element
            dE = (1 / (4 * np.pi * epsilon_0)) * sigma_0 * dA / R**3
            Ex += dE * Rx
            Ez += dE * Rz

    E_total = np.sqrt(Ex**2 + Ez**2)

    return E_total, Ex, Ez

def test_axis_field_from_disk():
    import numpy as np

    # Constants
    R_disk = 130e-9  # 130 nm

    # Evaluation height
    z = 72e-9  # 50 nm
    rho = np.array([0.0])  # on the axis

    # Compute numerical field
    E_total, E_rho, E_z = get_e_field_from_disk(rho, np.array([z]), R_disk=R_disk)

    # Analytical field on axis
    E_z_analytical = (sigma_0 / (2 * epsilon_0)) * (1 - z / np.sqrt(z**2 + R_disk**2))

    # Compare numerical and analytical
    rel_error = np.abs(E_z[0, 0] - E_z_analytical) / np.abs(E_z_analytical)

    print('Testing numerical function for electric field from disk:')
    print(f"Numerical E_z:  {E_z[0, 0]:.6e} V/m")
    print(f"Analytical E_z: {E_z_analytical:.6e} V/m")
    print(f"Relative error: {rel_error:.2e}")

    assert rel_error < 0.01, "Relative error exceeds 1%!"

# Run the test
test_axis_field_from_disk()


def get_drive_from_disk(x, z, drive_eff_x, drive_eff_z, R_disk=130e-9):
    """
    Calculate the drive field from a disk with given effective drive fields in x and z directions.
    """
    E_total, Ex, Ez = get_e_field_from_disk(x, z, R_disk)
    # Scale the electric field by the drive efficiency
    drive_x = Ex*drive_eff_x
    drive_z = Ez*drive_eff_z
    drive_constructive = np.sqrt(drive_x**2 + drive_z**2)
    drive_deconstructive = np.sqrt(np.abs(drive_x**2 - drive_z**2))
    return drive_constructive, drive_deconstructive

# calculate all distances in 2d grid
spacing = 195e-9  # distance between qubits in m
qubit_distances = []
for n in range(10):
    for m in range(n, 10):
        d = np.sqrt((n*spacing)**2 + (m*spacing)**2)
        qubit_distances.append(d)
qubit_distances = np.sort(np.unique(qubit_distances)*1e6)

# %%
# Observation grid
grid_size = (300, 50)  # number of grid points
x_e_field_from_disk = np.linspace(-900e-9, 900e-9, grid_size[0])
z_e_field_from_disk = np.linspace(-150e-9, -20e-9, grid_size[1])
E_total, Ex, Ez = get_e_field_from_disk(x_e_field_from_disk, z_e_field_from_disk, R_disk=130e-9)

# %%

grid_size = 300  # number of grid points
x = x_e_field_from_disk
z = z_e_field_from_disk

# Plotting
fig, axes = plt.subplots(2, 1, figsize=cm2inch(8, 6), sharex=True)
# axes = [ax]

qubit_sites = -1*qubit_distances[:10][::-1][:-1]
qubit_sites = np.append(qubit_sites, qubit_distances[:10])
qubit_sites = qubit_sites*1e3# add the center qubit at 0 nm

# plot linecut at z = -60
# find z index where z=-72
z_qw = 72
z_index = np.argmin(np.abs(z + z_qw*1e-9))
linecut = np.abs(E_total[z_index, :])
linecut_ex = np.abs(Ex[z_index, :])
linecut_ez = np.abs(Ez[z_index, :])

im = axes[0].imshow(E_total, extent=(x.min() * 1e9, x.max() * 1e9, z.min() * 1e9, z.max() * 1e9),
           origin='lower', aspect='auto', cmap='jet', norm=Normalize())
axes[0].scatter(qubit_sites, [-60]*len(qubit_sites), color='black', s=5, label='Qubit positions')
axes[0].set_xlabel('r (nm)')
axes[0].set_ylabel('z (nm)')
cbar = plt.colorbar(im, ax=axes[0], orientation='horizontal', shrink=0.8, pad=0.15, location='top')
cbar.set_label('|E| (a.u.)')

axes[1].plot(x * 1e9, linecut/max(linecut), color='black', linestyle='-', label='Linecut at z = -60 nm')
axes[1].set_xlabel('x [nm]')
axes[1].set_ylabel('|E| [V/m]')
for site in qubit_sites:
    axes[1].vlines(site, ymin=0, ymax=1, color='black', linestyle='--', linewidth=0.5)

for ax in axes:
    ax.set_xlim(x.min() * 1e9, x.max() * 1e9)

plt.tight_layout()
plt.savefig(os.path.join(fig_path, 'e_field_underneath_plunger.png'), dpi=300, transparent=True)
plt.show()

# %% fit linecut

from scipy.optimize import curve_fit
from scipy.optimize import least_squares
from sklearn.metrics import r2_score

def potential_disk(r, z, d, q=1e3):
    r_gate = d/2
    phi = q*np.arcsin(d/(np.sqrt((r-r_gate)**2+ z**2)+np.sqrt((r+r_gate)**2+ z**2)))
    return phi

def get_ex_from_potential_disk(r, z, d, q=1e3):
    phi = potential_disk(r, z, d, q)
    ex = -np.gradient(phi, r)
    return ex

def get_ey_from_potential_disk(r, z, d):
    phi = potential_disk(r, z, d)
    ey = -np.gradient(phi, z)
    return ey

def e_field_x(x, r0, exponent, A):
    z0 = 72
    # A = (z0**2+r0**2)**(exponent+1) / (2*exponent*z0)
    return np.abs(2 * exponent * A * x / ((x**2 + z0**2 + r0**2)**(exponent + 1)))

def get_amplitude(r0, exponent):
    z0 = 72
    A = (z0 ** 2 + r0 ** 2) ** (exponent + 1) / (2 * exponent * z0)
    return A

def e_field_z(x, r0, exponent):
    z0 = 72
    A = get_amplitude(r0, exponent)
    return 2 * exponent * A * z0 / ((x**2 + z0**2 + r0**2)**(exponent + 1))

def e_field_total(x, A, r0, exponent):
    return np.sqrt(e_field_x(x, A, r0, exponent)**2 + e_field_z(x, A, r0, exponent)**2)

def residuals(params, x1, y1, x2, y2):
    # if len(params) == 3:
    r1, r2, exponent = params
    # elif len(params) == 2:
    #     A, exponent = params
    #     r0 = 130
    amp = get_amplitude(r2, exponent)

    res1 = e_field_x(x1, r1, exponent, amp) - y1
    res2 = e_field_z(x2, r2, exponent) - y2
    return np.concatenate([res1, res2])

x_nm = x*1e9
E_total_normalized = E_total / max(linecut)
linecut_normalized = linecut / max(linecut)
linecut_ex_normalized = linecut_ex / max(linecut)
linecut_ez_normalized = linecut_ez / max(linecut)

r0_guess = 360
exp_guess = 0.8
a_guess = (r0_guess**2 + z_qw**2) ** (exp_guess+1) / (2*exp_guess*r0_guess)
result = least_squares(residuals, [r0_guess, r0_guess, exp_guess],
                       args=(x_nm, linecut_ex_normalized, x_nm, linecut_ez_normalized)
                       , bounds=(0.1, [2*r0_guess, 2*r0_guess, 3]))
popt_etot = result.x

# result_70nm = least_squares(residuals, [a_guess, exp_guess],
#                        args=(x_nm, linecut_ex_normalized, x_nm, linecut_ez_normalized))
# popt_etot_70nm = result_70nm.x
# popt_etot_70nm = np.insert(popt_etot_70nm, 1, r0_guess)
# Fit the linecut data

# popt, pcov = curve_fit(coulomb_like, x_nm, linecut_normalized,
#                        p0=[a_guess, x0_guess, 1.01],
#                        bounds=(0.1, [2*a_guess, 2*x0_guess, 3]))

# x0_guess = 240.0  # initial guess for x0
# exp_guess = 1.4  # initial guess for exponent
# a_guess = (x0_guess**2)**(exp_guess+1) / (2*exp_guess*x0_guess)
# upper_bounds_etot = np.array([1E3*a_guess, 2*x0_guess, 3])
# popt_etot, pcov_etot = curve_fit(e_field_total, x_nm, linecut_normalized,
#                        p0=[a_guess, x0_guess, exp_guess],
#                        bounds=(0.1, upper_bounds_etot))
# print(popt_etot/upper_bounds_etot)

r0_guess = 240.0  # initial guess for x0

# from functools import partial
# e_field_z_partial = partial(e_field_z, A=popt_ex[0], z0=popt_ex[1])
# def e_field_z_partial(x, exp, z0):
#     return e_field_z(x, popt_ex[0], z0, exp)
#
# popt_ez, pcov_ez = curve_fit(e_field_z_partial, x_nm, linecut_ez_normalized,
#                        p0=[exp_guess, popt_ex[1]],
#                        bounds=([0.5, 0.9*popt_ex[1]], [2, 1.1*popt_ex[1]]))

# r0_guess = popt_ex[1]
exp_guess = 0.8  # initial guess for exponent
upper_bounds= np.array([400, 2.5])
popt_ez, pcov_ez = curve_fit(e_field_z, x_nm, linecut_ez_normalized,
                             p0=[r0_guess, exp_guess],
                             bounds=(0.4, upper_bounds))
print(popt_ez/upper_bounds)

exp_guess = 1.4  # initial guess for exponent
a_guess = get_amplitude(*popt_ez)
lower_bounds_ex = np.array([10, 0.2, 0.99*a_guess])
upper_bounds_ex = np.array([400, 3, 1.01*a_guess])
popt_ex, pcov_ex = curve_fit(e_field_x, x_nm, linecut_ex_normalized,
                             p0=[r0_guess, exp_guess, a_guess],
                             bounds=(0.4, upper_bounds_ex))
print(popt_ex/upper_bounds_ex)

x_fit = np.linspace(x_nm.min(), x_nm.max(), 1000)

y_fit_ez = {}
y_fit_ex = {}
y_fit = {}

y_fit_ez['popt_ez'] = e_field_z(x_fit, *popt_ez)
y_fit_ex['popt_ex'] = e_field_x(x_fit, *popt_ex)
y_fit['popt_ex_ez'] = np.sqrt(y_fit_ez['popt_ez']**2 + y_fit_ex['popt_ex']**2)

y_fit_ez['popt_etot'] = e_field_z(x_fit, popt_etot[1], popt_etot[2])
y_fit_ex['popt_etot'] = e_field_x(x_fit, popt_etot[0], popt_etot[2], get_amplitude(popt_etot[1], popt_etot[2]))
y_fit['popt_etot'] = np.sqrt(y_fit_ez['popt_etot'] **2 + y_fit_ex['popt_etot'] **2)

# y_fit_ez['popt_etot_70nm'] = e_field_z(x_fit, *popt_etot_70nm)
# y_fit_ex['popt_etot_70nm'] = e_field_x(x_fit, *popt_etot_70nm)
# y_fit['popt_etot_70nm'] = np.sqrt(y_fit_ez['popt_etot_70nm'] **2 + y_fit_ex['popt_etot_70nm'] **2)

# y_fit = lorentzian(x_fit, *popt)
# Plot the original linecut and the fitted curve
fig, axes = plt.subplots(2, 2, figsize=cm2inch(10, 7.2), sharex=True)
axes = axes.flatten()

norm = Normalize(vmin=0, vmax=np.max(E_total_normalized))
im = axes[0].imshow(E_total_normalized, extent=(x.min() * 1e9, x.max() * 1e9, z.min() * 1e9, z.max() * 1e9),
           origin='lower', aspect='auto', cmap='jet', norm=norm)
axes[0].scatter(qubit_sites, [-60]*len(qubit_sites), facecolors='none', color='black', s=5, label='Qubit positions')
axes[0].set_xlabel('r (nm)')
axes[0].set_ylabel('z (nm)')
cbar = plt.colorbar(im, ax=axes[0], orientation='horizontal', shrink=0.8, pad=0.15, location='top')
cbar.set_label('$|E| / |E_\mathrm{qw}^\mathrm{max}|$')
cbar.set_ticks([0, 0.5, 1, 1.5])

axes[1].plot(x_nm, linecut_normalized, '-', label=f'Linecut at $z = -{z_qw:.0f}$ nm')
# axes[1].plot(x_fit, y_fit['popt_etot'], color='tab:orange', ls='--', lw=1, label='Fit shared popt')
# axes[1].plot(x_fit, y_fit['popt_ex_ez'], color='tab:green', ls='--', lw=1, label='Fit ind popt')
# axes[1].plot(x_fit, y_fit['popt_etot_70nm'], color='tab:red', ls='--', lw=1)
axes[1].set_ylim(0, 1.1)
axes[1].set_ylabel('$|E_\mathrm{qw}| / |E_\mathrm{qw}^\mathrm{max}|$')
# make legend outside plot
axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.4), frameon=False)
# axes[1].set_ylabel('$\sqrt{E_{tot}^2 / |E_{tot, max}^2}$')
# axes[1].set_title('$E_{tot}(z=-60 \mathrm{nm})$')

axes[2].plot(x_nm, linecut_ex_normalized, '-', label='Linecut data')
# axes[2].plot(x_fit, y_fit_ex['popt_etot'], color='tab:orange', ls='--', lw=1, label='Fitted curve')
# axes[2].plot(x_fit, y_fit_ex['popt_ex'], color='tab:green', ls='--', lw=1, label='Fitted curve')
# axes[2].plot(x_fit, y_fit_ex['popt_etot_70nm'], color='tab:red', ls='--', lw=1)
axes[2].set_xlabel('r (nm)')
axes[2].set_ylabel('$|E_\mathrm{r,qw}| / |E_\mathrm{qw}^\mathrm{max}|$')
# axes[2].set_title('$E_{x}(z=-60 \mathrm{nm}))$')

axes[3].plot(x_nm, linecut_ez_normalized, '-', label='Linecut data')
# axes[3].plot(x_fit, y_fit_ez['popt_etot'], color='tab:orange', ls='--', lw=1, label='Fitted curve')
# axes[3].plot(x_fit, y_fit_ez['popt_ez'], color='tab:green', ls='--', lw=1, label='Fitted curve')
# axes[3].plot(x_fit, y_fit['popt_etot_70nm'], color='tab:red', ls='--', lw=1)
axes[3].set_xlabel('r (nm)')
axes[3].set_ylabel('$|E_\mathrm{z,qw}| / |E_\mathrm{qw}^\mathrm{max}|$')
# axes[3].set_title('$E_{z}(z=-60 \mathrm{nm}))$')

for ax in axes[1:]:
    for site in qubit_sites:
        ax.set_ylim(0, 1)
        ax.set_xlim(x_nm.min(), x_nm.max())
        ax.vlines(site, ymin=0, ymax=1, color='black', linestyle='--', linewidth=0.5)

# plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(fig_path, 'e_field_along_x.png'), dpi=300, transparent=True)
plt.savefig(os.path.join(fig_path, 'e_field_along_x.pdf'), dpi=300, transparent=True)
plt.show()


# %%
# b = 130
# z0 = 72
# r = np.linspace(-500, 500, 1000)  # radial distance in nm
#
# V_smythe = 54*np.arctan(np.sqrt(2)*b / np.sqrt(r**2+z0**2-b**2 + np.sqrt((r**2+z0**2-b**2)**2 + 4*b**2*r**2)))
#
# er_smythe = np.abs(-np.gradient(V_smythe, r))
# ez_smythe = -np.gradient(V_smythe, z0)
#
# plt.figure()
# plt.plot(r, er_smythe, label='E_r (Smythe)', color='tab:blue')
# plt.plot(r, ez_smythe, label='E_z (Smythe)', color='tab:orange')
# # plt.plot(x_nm, linecut_ex_normalized, label='E_r (data)', color='tab:green', linestyle='--')
# plt.tight_layout()
# plt.show()

# %% plot decay constant

# fig, axes = plt.subplots(3, 1, figsize=(3, 4), sharex=True)
#
# # mask = np.array(rsquare_values) > 0.8
# # # color background as a function of the mask and x
# # regions = np.split(x_nm[::step_size][mask], 2)
# # plt.axvspan(regions[0][0], regions[0][-1], color='green', alpha=0.5)
# # plt.axvspan(regions[1][0], regions[1][-1], color='green', alpha=0.5)
#
# axes[0].plot(x_nm[::step_size], np.array(decay_As), marker='o', label='Decay exponent')
# axes[0].set_xlabel('x [nm]')
# axes[0].set_ylabel('A')
#
# axes[1].plot(x_nm[::step_size], np.array(decay_x0s), marker='o', label='Decay exponent')
# axes[1].set_xlabel('x [nm]')
# axes[1].set_ylabel('x0')
#
# axes[2].plot(x_nm[::step_size], 2*np.array(decay_exponents), marker='o', label='Decay exponent')
# axes[2].set_xlabel('x [nm]')
# axes[2].set_ylabel('Decay exponent')
#
# plt.tight_layout()
# plt.savefig(os.path.join(fig_path, 'decay_exponent.png'), dpi=300, transparent=True)
# plt.show()

# %%

# boxplot locality data
df_drive_eff, drive_eff = unn.get_locality_data()
mean_value_dict = {'1h': {}, '3h': {}}
std_value_dict = {'1h': {}, '3h': {}}
for hole in ['1h', '3h']:
    for gate_type in ['plunger', 'barrier']:
        mean_value_list = []
        sigma_list = []
        mean_value_norm_list = []
        for step in df_drive_eff[hole][gate_type].keys():
            values = df_drive_eff[hole][gate_type][step].stack()
            sigma_list.append(values.std())
            mean_value_list.append(np.mean(values))
        mean_value_dict[hole][gate_type] = mean_value_list
        std_value_dict[hole][gate_type] = sigma_list


grid_size = 100
x = np.linspace(0, 800e-9, grid_size)
z = [-60e-9]

ratio_drive_x_z = 1

drives = {}
drives_deconstructive = {}
drives_x_z = [[0.6, 0.32],
              [0.8, 1]]
for drive_eff_x, drive_eff_z in drives_x_z:
    # drive_eff_z = drive_x_z
    # drive_eff_x = 1-drive_x_z
    ratio_drive_x_z = drive_eff_x / drive_eff_z
    drives[f'({drive_eff_x}, {drive_eff_z})'] = get_drive_from_disk(x, z, drive_eff_x=drive_eff_x, drive_eff_z=drive_eff_z, R_disk=130e-9)[0][0]
    drives_deconstructive[f'({drive_eff_x}, {drive_eff_z})'] = get_drive_from_disk(x, z, drive_eff_x=drive_eff_x, drive_eff_z=drive_eff_z, R_disk=130e-9)[1][0]
    # Calculate the effective drive fields

# Plot the drive field
fig, ax = plt.subplots(figsize=cm2inch(8, 6.3))

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
max_efield = max(np.array(list(drives.values())).flatten())
max_drive = max(mean_value_dict['3h']['plunger'])

for n, key in enumerate(drives.keys()):
    drive = drives[key]
    drive_dec = drives_deconstructive[key]
    color = colors[n]
    num_holes = ['1h', '3h'][n]
    ax.plot(x * 1e9, drive/max_efield*max_drive, label=f'$(w_r^{{{num_holes}}}, w_z^{{{num_holes}}}, \phi)$ = {key[:-1]}, 0)', color=color)
    ax.plot(x * 1e9, drive_dec / max_efield * max_drive, label=f'$(w_r^{{{num_holes}}}, w_z^{{{num_holes}}}, \phi) = {key[:-1]}, \pi/2)$', linestyle='--', color=color)

# for color , (key, drive) in zip(colors, drives_deconstructive.items()):
#     ax.plot(x * 1e9, drive/max_efield*max_drive, label=f'(w_x, w_y) = {key}, $\phi=\pi/2$', linestyle='--', color=color)

for qubit_distance in qubit_distances[:10]:
    ax.vlines(qubit_distance*1e3, ymin=0, ymax=390, color='black', linestyle='--', linewidth=0.5)


ax.errorbar(qubit_distances[:6]*1e3-5, mean_value_dict['1h']['plunger'],
             std_value_dict['1h']['plunger'], label='measurement data for $f_{1h,P}^R$', color='blue', linestyle='')
ax.errorbar(qubit_distances[:6]*1e3+5, mean_value_dict['3h']['plunger'],
             std_value_dict['3h']['plunger'], label='measurement data for $f_{3h,P}^R$', color='orange', linestyle='')
ax.set_ylabel('$f_R/A$ (MHz/mV)')
ax.legend(loc='upper right')

scale_factor = 1.4
max_value_ax = scale_factor*np.array(list(drives.values())).flatten().max()*1e-3
max_value_ax2 = scale_factor*max(mean_value_dict['3h']['plunger'])
ax.set_ylim(0, max_value_ax2)

ax.set_xlabel('r (nm)')
# ax.set_title('Drive field beneath an oscillating charged disk')
ax.legend(loc='upper right', frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(fig_path, 'x_z_drive_weights.png'), dpi=300, transparent=True)
plt.savefig(os.path.join(fig_path, 'x_z_drive_weights.pdf'), dpi=300, transparent=True)
plt.show()

# %%

qubit_nearest_neighbour_distances = []
qubit_nearest_neighbour_distances_e_total = []
qubit_nearest_neighbour_distances_ex = []
qubit_nearest_neighbour_distances_ey = []
qubit_nearest_neighbour_distances_number = []

for qubit_distances in unn.qubit_distances_drive['plunger'].values():
    qubit_nearest_neighbour_distances.append(qubit_distances[0])
    qubit_nearest_neighbour_distances_number.append(len(qubit_distances))

e_field, ex_field, ez_field = get_e_field_from_disk(np.array(qubit_nearest_neighbour_distances)*1e-6, [-60e-9],
                                                    R_disk=130e-9)
# normalize the electric field
e_norm_field = e_field / np.max(e_field)
ex_norm_field = np.abs(ex_field / np.max(e_field))
ez_norm_field = np.abs(ez_field / np.max(e_field))

# fit e_norm_field to a power lay
def power_law(x, a, b):
    if np.isscalar(x):
        x = np.array([x])
    return 1/(1+a*x**b)

# popt, _ = curve_fit(power_law,
#                     np.array(qubit_nearest_neighbour_distances),
#                     e_norm_field, p0=[200, 3],
#                     bounds=([10, 1], [1000, 4]))

# Generate fitted curve
x_fit = np.linspace(min(qubit_nearest_neighbour_distances), max(qubit_nearest_neighbour_distances), 1000)
popt = [200, 2.9]  # Example parameters for the power law fit
y_fit = power_law(x_fit, *popt)

# plot electric field for each qubit and label number of nearest neighbours
fig, ax = plt.subplots(figsize=(4, 3))
ax.scatter(qubit_nearest_neighbour_distances, np.abs(e_norm_field), label='E_total')
ax.plot(x_fit, y_fit, 'r--', label='Power law fit')
ax.scatter(qubit_nearest_neighbour_distances, np.abs(ex_norm_field), label='E_r', marker='x')
ax.scatter(qubit_nearest_neighbour_distances, np.abs(ez_norm_field), label='E_z', marker='^')
ax.set_xlabel('x [nm]')
ax.set_ylabel('|E| [V/m]')
ax.set_title('Electric field linecuts for different qubits')
ax.legend()
plt.tight_layout()
plt.show()

