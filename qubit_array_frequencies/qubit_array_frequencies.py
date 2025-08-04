# %% imports

from drive_locality import utils_nearest_neighbours as unn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.optimize import differential_evolution
from tqdm import tqdm

# %% definitions

def compute_distance_matrix(coords1, coords2):
    """
    Computes a matrix of Euclidean distances between points in coords1 and coords2.

    Parameters:
    - coords1: List of (x, y) tuples or 2D NumPy array of shape (n1, 2)
    - coords2: List of (x, y) tuples or 2D NumPy array of shape (n2, 2)

    Returns:
    - distance_matrix: 2D NumPy array of shape (n1, n2)
    """
    coords1 = np.array(coords1)
    coords2 = np.array(coords2)

    diff = coords1[:, np.newaxis, :] - coords2[np.newaxis, :, :]
    distance_matrix = np.linalg.norm(diff, axis=2)

    return distance_matrix

def matrix_rank_by_value(matrix, digits=None):
    """
    Converts a matrix into a rank matrix based on descending order of values.
    Equal values (after optional rounding) get the same rank.

    Parameters:
    - matrix: 2D array-like (e.g., list of lists or np.ndarray)
    - digits: int or None. If set, round each value to that number of digits before ranking.

    Returns:
    - rank_matrix: np.ndarray of same shape as input, with ranks (1 = largest).
    """
    matrix = np.array(matrix, dtype=float)

    if digits is not None:
        matrix = np.round(matrix, digits)

    flat = matrix.flatten()
    # Get unique sorted values
    unique_values = np.unique(flat)

    # Create a mapping from value to rank
    value_to_rank = {val: rank+1 for rank, val in enumerate(unique_values)}

    # Apply the mapping to each entry
    rank_matrix = np.vectorize(value_to_rank.get)(matrix)

    return rank_matrix

def fill_by_rank(rank_matrix, values):
    """
    Fills the input rank_matrix with values according to their rank.
    Ranks start at 1. Values beyond the provided list are filled with np.nan.

    Parameters:
    - rank_matrix: 2D array-like with integer ranks (starting at 1)
    - values: list of values corresponding to rank 1, 2, ...

    Returns:
    - filled_matrix: np.ndarray with same shape as rank_matrix
    """
    rank_matrix = np.array(rank_matrix, dtype=int)
    filled = np.full(rank_matrix.shape, np.nan)

    max_rank = np.max(rank_matrix)
    for r in range(1, max_rank + 1):
        mask = rank_matrix == r
        if r <= len(values):
            filled[mask] = values[r - 1]
        else:
            filled[mask] = np.nan

    return filled

# %% drive efficiency nearest neighbours

df_drive_eff, drive_eff = unn.get_locality_data()

drive_eff_pl_3h = pd.DataFrame(drive_eff['3h']['plunger']).loc['mean'].to_list()

# %% 4x4 unit cell nearest neighbour matrix

nearest_neighbour_distances = [0.0, 0.2, 0.28, 0.39, 0.44, 0.55]

coordinates = {}

gates = []
positions = []

array_size = 12
for n in range(array_size):
    for m in range(array_size):
        gate = f"P{n+1}-{m+1}"
        gates.append(gate)
        positions.append([0.2*n, 0.2*m])
        coordinates[gate] = [0.2*n, 0.2*m]

distances = compute_distance_matrix(positions, positions)
distances_unique_sorted = sorted(set(np.round(distances.flatten(), 4)))

ranked_distances = matrix_rank_by_value(distances, digits=4)

# %% fill by rank

drive_eff_pl_3h_rounded = np.round(drive_eff_pl_3h, 4)
drive_eff_matrix = fill_by_rank(ranked_distances, drive_eff_pl_3h_rounded)

f_rabi_target = 2 # MHz
amplitude = f_rabi_target / drive_eff_pl_3h_rounded[0]

f_rabi_matrix = drive_eff_matrix * amplitude
# replace nan values with 0
f_rabi_matrix = np.nan_to_num(f_rabi_matrix, nan=0.0)

qubits = [f'Q{n+1}' for n in range(array_size**2)]
df_larmor_matrix = pd.DataFrame(f_rabi_matrix, index=gates, columns=qubits)

# effective_f_rabi_matrix = fold_back_tile(f_rabi_matrix, num_tiles=3)

# %% Larmor frequencies

# define 16 Larmor frequencies for 4x4 unit cell
larmor_fquencies = [200, 220, 240, 260,
                     280, 300, 320, 340,
                     360, 380, 400, 420,
                     440, 460, 480, 500] # MHz

# %% central tile

plunger_names_all = [f"P{n+1}-{m+1}" for n in range(12) for m in range(12)]
plunger_names_central = [f"P{n+1}-{m+1}" for n in range(4, 8) for m in range(4, 8)]
plunger_indices_central = [np.where(np.array(plunger_names_all) == name)[0][0] for name in plunger_names_central]

qubit_names = [f"Q{n+1}" for n in range(12**2)]
# assign tile e.g. P1-1 up to P1-4 and P4-1 to tile 0, P4-1 up to P4-8 and P8-4 to tile 1,
tile_assignment = {f"P{n+1}-{m+1}": (n // 4) * 3 + (m // 4) for n in range(12) for m in range(12)}
# invert dictionary and save keys of the same tile in a list
tile_dict = {}
for key, value in tile_assignment.items():
    if value not in tile_dict:
        tile_dict[value] = []
    tile_dict[value].append(key)

# dataframe
df = pd.DataFrame(plunger_names_all, qubit_names, columns=['plunger_name'])
df['tile'] = [tile_assignment[name] for name in plunger_names_all]
df['tile'] = df['tile'].astype(int)

# assign for each tile subnumber for each identical qubits
df['subnumber'] = df.groupby('tile').cumcount() + 1
df['subnumber'] = df['subnumber'].astype(int)

df['x'] = [np.round(coordinates[name][0], 2) for name in plunger_names_all]
df['y'] = [np.round(coordinates[name][1], 2) for name in plunger_names_all]

df_central = df.where(df.tile == 4).dropna()

# %% plot at each position x,y a circle with color value subnumber

color_prop = 'tile'  # color by subnumber
text_prop = 'subnumber'  # text label by plunger name

plt.figure(figsize=(4, 4))
for name, row in df.iterrows():
    color = plt.cm.viridis(row[color_prop] / df[color_prop].max())  # normalize subnumber to [0, 1]
    circle = plt.Circle((row['x'], row['y']), 0.06, color=color, label=row['plunger_name'])
    plt.gca().add_artist(circle)
    plt.text(row['x'], row['y'], row[text_prop], fontsize=8, ha='center', va='center',
             color='white')
plt.xlim(-0.1, 2.3)
plt.ylim(-0.1, 2.3)
plt.axis('off')
plt.gca().set_aspect('equal', adjustable='box')
plt.title('Plunger Positions with Subnumber Color Coding')
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.show()

# %% make distance dataframe

# compute distances between qubit and plunger gates
distances_from_dfxy = compute_distance_matrix(df[['x', 'y']].to_numpy(), df[['x', 'y']].to_numpy())
df_distances = pd.DataFrame(distances_from_dfxy, index=df.index, columns=df['plunger_name'])

# rank distances
ranked_distances_from_dfxy = matrix_rank_by_value(distances_from_dfxy, digits=4)
df_ranked_distances = pd.DataFrame(ranked_distances_from_dfxy, index=df.index, columns=df['plunger_name'])

# fill by rank

drive_eff_pl_3h_rounded = np.round(drive_eff_pl_3h, 4)
drive_eff_matrix_from_dfxy = fill_by_rank(ranked_distances_from_dfxy, drive_eff_pl_3h_rounded)

f_rabi_target = 5 # MHz
amplitude = f_rabi_target / drive_eff_pl_3h_rounded[0]

f_rabi_matrix_from_dfxy = drive_eff_matrix_from_dfxy * amplitude
f_rabi_matrix_from_dfxy = np.nan_to_num(f_rabi_matrix, nan=0.0)

df_rabi = pd.DataFrame(f_rabi_matrix_from_dfxy, index=df.index, columns=df['plunger_name'])

# %% compare distances, ranked distances and f_rabi by plotting them

fig, axes = plt.subplots(3, 1, figsize=(6, 12))
vmaxes = [0.5, 6, df_rabi.values.max()]
for ax, vmax, data, title in zip(axes, vmaxes, [df_distances, df_ranked_distances, df_rabi],
                            ['Distances', 'Ranked Distances', 'f_Rabi']):
    x = data.index
    y = data.columns
    z = data.values
    c = ax.pcolor(x, y, z.T, cmap='viridis', shading='auto', vmax=vmax)
    ax.set_title(title)
    ax.axis('off')

plt.tight_layout()
plt.show()

df_subset = df_rabi[df['tile'] == 4].dropna()  # central tile
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
x = df_subset.index
y = df_subset.columns
z = df_subset.values
c = ax.pcolor(x, y, z.T, cmap='viridis', shading='auto')
ax.set_title('f_Rabi')
ax.axis('off')
plt.tight_layout()
plt.show()

# %% assign Larmor frequencies to the qubits (one for each subnumber in the tile)

def f_rabi_function(f_rabi, f_rabi_target, f_larmor, f_drive):
    if f_rabi == 0:
        return 0.0
    t_rabi = 1 / (2*f_rabi_target)
    return f_rabi ** 2 / (2 * (f_rabi ** 2 + (f_larmor - f_drive) ** 2)) * (1 - np.cos(2*np.pi*np.sqrt(f_rabi**2 + (f_larmor - f_drive)**2) * t_rabi))

def f_rabi_function_envelope(f_rabi, f_larmor, f_drive):
    if f_rabi == 0:
        return 0.0
    return f_rabi ** 2 / (f_rabi ** 2 + (f_larmor - f_drive) ** 2)

f_rabi = 5  # MHz
f_rabi_target = 5  # MHz
f_larmor = 300  # MHz
f_drive = np.linspace(250, 350, 101)

plt.figure(figsize=(4, 3))
plt.plot(f_drive, f_rabi_function(f_rabi, f_rabi_target, f_larmor, f_drive), label='f_rabi_function', color='blue')
plt.plot(f_drive, f_rabi_function_envelope(f_rabi, f_larmor, f_drive), label='f_rabi_gaussian', color='orange')

plt.xlabel('Drive Frequency (MHz)')
plt.ylabel('Spin Probability')
plt.tight_layout()
plt.show()

# %%

def get_crosstalk_df(df, df_rabi, larmor_frequencies=[300]*16):
    df_larmor_const = df.copy()
    for subnumber in sorted(df_larmor_const['subnumber'].unique()):
        # assign the same Larmor frequency to all qubits with the same subnumber
        larmor_freq = larmor_frequencies[subnumber - 1]  # subnumber starts at 1
        df_larmor_const.loc[df_larmor_const['subnumber'] == subnumber, 'larmor_frequency'] = larmor_freq

    qubit_to_larmor = df_larmor_const['larmor_frequency'].to_dict()
    gate_to_larmor = (
        df_larmor_const.set_index('plunger_name')['larmor_frequency']
        .dropna()
        .to_dict()
    )
    result = np.zeros_like(df_rabi.values, dtype=float)
    for i, qubit in enumerate(df_rabi.index):
        f_larmor = qubit_to_larmor[qubit]
        for j, gate in enumerate(df_rabi.columns):
            f_rabi = df_rabi.iat[i, j]
            if i != j:
                if f_rabi != 0:
                    f_drive = gate_to_larmor.get(gate, np.nan)
                    if not np.isnan(f_drive):
                        result[i, j] = f_rabi_function_envelope(f_rabi, f_larmor, f_drive)
                        if np.isnan(result[i, j]):
                            print(
                                f"NaN detected for i={i}, j={j}, f_rabi={f_rabi}, f_larmor={f_larmor}, f_drive={f_drive}")
            else:
                result[i, j] = 0.0

    df_crosstalk_matrix = pd.DataFrame(result, index=df_rabi.index, columns=df_rabi.columns)
    
    return df_crosstalk_matrix

# df_crosstalk = get_crosstalk_df(df, df_rabi, larmor_frequencies=[300]*16)
#
# summed_qubit_crosstalk = df_crosstalk.sum(axis=1) - 1
#
# # add to df
# df['crosstalk'] = summed_qubit_crosstalk
# df_crosstalk_central = df_crosstalk[df['tile'] == 4].dropna().sum(axis=1) - 1

# %% minimize crosstalk metric

df_local = df.copy()
df_rabi_local = df_rabi.copy()

def crosstalk_metric(larmor_frequencies, mode='max'):
    df_crosstalk = get_crosstalk_df(df_local, df_rabi_local, larmor_frequencies=larmor_frequencies)
    # summed_qubit_crosstalk = df_crosstalk.sum(axis=1) - 1
    # df['crosstalk'] = summed_qubit_crosstalk
    # df_crosstalk_central = df_crosstalk[df['tile'] == 4].dropna().sum(axis=1)
    # print("Evaluating:", larmor_frequencies)
    # print("Crosstalk:", df_crosstalk_central.max())
    if mode == 'max':
        return df_crosstalk[df['tile'] == 4].dropna().max(axis=1).max()
    elif mode == 'sum':
        return df_crosstalk[df['tile'] == 4].dropna().sum(axis=1).max()

def crosstalk_metric_reduced_larmors(reduced_larmor_frequencies, mode='max'):
    reduced_larmor_frequencies = list(reduced_larmor_frequencies)
    reduced_larmor_frequencies = [400, 200] + reduced_larmor_frequencies
    larmor_frequencies = reduced_larmor_frequencies + (reduced_larmor_frequencies[2:4] +
                                                       reduced_larmor_frequencies[0:2] +
                                                       reduced_larmor_frequencies[6:8] +
                                                       reduced_larmor_frequencies[4:6])
    crosstalk = crosstalk_metric(larmor_frequencies, mode=mode)
    # print('--------------------------------------------')
    # print("Evaluating reduced Larmor frequencies:", reduced_larmor_frequencies)
    print(crosstalk)
    return crosstalk


# %%

reduced_larmor_frequencies_guess = [380, 220, 250, 330, 280, 350]
reduced_larmor_frequencies_bounds = [(350, 400), (200, 250),
                                     (250, 300), (300, 350),
                                     (250, 300), (300, 350)]

# Define how many generations you expect
maxiter = 20
pbar = tqdm(total=maxiter, desc="Optimizing")
def callback_fn(xk, convergence):
    pbar.update(1)

fitness_history = []
def track_evolution(xk, convergence):
    current_best = crosstalk_metric_reduced_larmors(xk, mode='max')
    fitness_history.append(current_best)

results_mode = {'max': None, 'sum': None}

# for mode in ['max', 'sum']:
mode = 'max'
# result = differential_evolution(crosstalk_metric_reduced_larmors, reduced_larmor_frequencies_bounds,
#                                 x0=reduced_larmor_frequencies_guess,
#                                 args=(mode,),
#                                 popsize=20, maxiter=maxiter, callback=callback_fn,
#                                 strategy='rand1bin', polish=False)

result = differential_evolution(crosstalk_metric_reduced_larmors, reduced_larmor_frequencies_bounds,
                                x0=reduced_larmor_frequencies_guess,
                                args=(mode,),
                                strategy='randtobest1bin',
                                maxiter=20, popsize=15,
                                callback=track_evolution,
                                # seed=42
                                )

results_mode[mode] = result

# %% plot evolution

plt.figure(figsize=(8, 4))
plt.plot(fitness_history)
plt.xlabel("Generation")
plt.ylabel("Best objective (crosstalk)")
plt.title("Differential Evolution Progress")

plt.show()

# %%

# mode = 'max'  # or 'max'
# result = results_mode[mode]
best_frequencies_semi_tile = [400.00, 200.00, 381.04, 216.91, 253.21, 331.23, 268.83, 347.48]

# best_frequencies_minimizer = list(result.x)
# best_frequencies_semi_tile = list(np.array([400, 200] + list(result.x)))
best_frequencies_tile = best_frequencies_semi_tile + (best_frequencies_semi_tile[2:4] +
                                                   best_frequencies_semi_tile[0:2] +
                                                   best_frequencies_semi_tile[6:8] +
                                                   best_frequencies_semi_tile[4:6])
# add best frequencies to df
for subnumber, larmor_freq in zip(sorted(df['subnumber'].unique()), best_frequencies_tile):
    # assign the same Larmor frequency to all qubits with the same subnumber
    larmor_freq = best_frequencies_tile[subnumber - 1]  # subnumber starts at 1
    df.loc[df['subnumber'] == subnumber, 'larmor_frequency'] = larmor_freq

df_cross_talk_matrix_best = get_crosstalk_df(df, df_rabi, larmor_frequencies=best_frequencies_tile)
cross_talk_matrix_best = df_cross_talk_matrix_best.values
index_best = np.unravel_index(np.argmax(cross_talk_matrix_best, axis=None), cross_talk_matrix_best.shape)

cross_talk_central_dict = {}
cross_talk_central = []
for name, index in zip(plunger_names_central, plunger_indices_central):
    cross_talk_central.append(cross_talk_matrix_best[index, :].sum() + cross_talk_matrix_best[:, index].sum())
    cross_talk_central_dict[name] = cross_talk_matrix_best[index, :].sum() + cross_talk_matrix_best[:, index].sum()

# print("Optimal parameters:", result.x)
# print("Minimum value:", result.fun)

# weakest link
larmor_nearest_neighbour_diff_axis_one = np.diff(np.array(best_frequencies_tile).reshape(-1, 4), axis=1)
larmor_nearest_neighbour_diff_axis_zero = np.diff(np.array(best_frequencies_tile).reshape(-1, 4), axis=0)

summed_qubit_crosstalk = df_cross_talk_matrix_best.sum(axis=1)
max_qubit_crosstalk = df_cross_talk_matrix_best.max(axis=1)

df['crosstalk_sum'] = summed_qubit_crosstalk
df['crosstalk_max'] = max_qubit_crosstalk

df_central = df[df['tile'] == 4].dropna()
df_cross_talk_matrix_best_central = df_cross_talk_matrix_best[df['tile'] == 4].dropna()

df_crosstalk_best_central_summary = df_central[['subnumber', 'larmor_frequency', 'crosstalk_max', 'crosstalk_sum']]
df_crosstalk_best_central_summary = df_crosstalk_best_central_summary.set_index('subnumber')
df_crosstalk_best_central_summary['larmor_frequency'] = df_crosstalk_best_central_summary['larmor_frequency'].round(2)
df_crosstalk_best_central_summary['crosstalk_sum'] = df_crosstalk_best_central_summary['crosstalk_sum'].round(5)

# write df_crosstalk_best_central_summary into a latex table
df_crosstalk_best_central_summary_latex = df_crosstalk_best_central_summary.copy()
df_crosstalk_best_central_summary_latex = df_crosstalk_best_central_summary_latex.T
df_crosstalk_best_central_summary_latex.index = ['$f_{Larmor}$', 'max Q2Q crosstalk ($10^{-3}$)', 'sum Q2Q crosstalk ($10^{-3}$)']
df_crosstalk_best_central_summary_latex.loc['max Q2Q crosstalk ($10^{-3}$)'] = df_crosstalk_best_central_summary_latex.loc['max Q2Q crosstalk ($10^{-3}$)'] * 1e3
df_crosstalk_best_central_summary_latex.loc['sum Q2Q crosstalk ($10^{-3}$)'] = df_crosstalk_best_central_summary_latex.loc['sum Q2Q crosstalk ($10^{-3}$)'] * 1e3
df_crosstalk_best_central_summary_latex.T.to_latex('df_crosstalk_best_central_summary.tex',
                                            column_format='|c|c|c|c|', float_format="%.2f",
                                            header=df_crosstalk_best_central_summary_latex.index,
                                            index=True, escape=False,
                                            caption='Crosstalk Summary for Central Tile',
                                            label='tab:crosstalk_summary_central_tile')

# %% plot cross talk matrix

fig, ax = plt.subplots(1, 1, figsize=(4, 4))
x = df_cross_talk_matrix_best_central.index
y = df_cross_talk_matrix_best_central.columns
z = df_cross_talk_matrix_best_central.values
c = ax.pcolor(x, y, z.T, cmap='viridis', shading='auto')

# add colorbar
cbar = plt.colorbar(c, ax=ax, location='bottom', pad=0.05, aspect=50)
cbar.set_label('Crosstalk Worst qubit', labelpad=5)

ax.set_title('Crosstalk Matrix')
ax.axis('off')
plt.tight_layout()
plt.show()

# %% plot at each position x,y a circle with color value subnumber

fig, axes = plt.subplots(1, 2, figsize=(6, 4))

color_prop = 'larmor_frequency'  # color by subnumber
text_prop = 'subnumber'  # text label by plunger name
norm = plt.Normalize(df_central[color_prop].min(), df_central[color_prop].max())
for name, row in df_central.iterrows():
    color = plt.cm.viridis(norm(row[color_prop]))  # normalize larmor frequency to [0, 1]
    circle = plt.Circle((row['x'], row['y']), 0.06, color=color, label=row['plunger_name'])
    ax.add_artist(circle)
    ax.text(row['x'], row['y'], row[text_prop], fontsize=8, ha='center', va='center',
             color='white')
ax.set_xlim(0.7, 1.5)
ax.set_ylim(0.7, 1.5)
# ax.set_xlim(0., 2.2)
# ax.set_ylim(0., 2.2)
ax.axis('off')
ax.set_aspect('equal', adjustable='box')
# ax.title('Plunger Positions with Subnumber Color Coding')
ax.set_xlabel('X Position (m)')
ax.set_ylabel('Y Position (m)')

# add colorbar with non normalized larmor frequency
sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, pad=0.02, ax=ax, location='bottom', aspect=50, shrink=0.8)
cbar.set_label('Larmor Frequency (MHz)', labelpad=5)


color_prop = 'crosstalk'  # color by subnumber
text_prop = 'subnumber'  # text label by plunger name
norm = plt.Normalize(0, df_central[color_prop].max())
for name, row in df_central.iterrows():
    color = plt.cm.viridis(norm(row[color_prop]))  # normalize larmor frequency to [0, 1]
    circle = plt.Circle((row['x'], row['y']), 0.06, color=color, label=row['plunger_name'])
    axes[1].add_artist(circle)
    axes[1].text(row['x'], row['y'], row[text_prop], fontsize=8, ha='center', va='center',
             color='white')
axes[1].set_xlim(0.7, 1.5)
axes[1].set_ylim(0.7, 1.5)
axes[1].axis('off')
axes[1].set_aspect('equal', adjustable='box')
# ax.title('Plunger Positions with Subnumber Color Coding')
axes[1].set_xlabel('X Position (m)')
axes[1].set_ylabel('Y Position (m)')

# add colorbar with non normalized larmor frequency
sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, pad=0.02, ax=axes[1], location='bottom', aspect=50, shrink=0.8)
if mode == 'sum':
    cbar.set_label(r'Summed qubit-qubit crosstalk', labelpad=5)
else:
    cbar.set_label(r'Maximum qubit-qubit crosstalk', labelpad=5)

# plt.suptitle(r'Larmor frequencies with minimal crosstalk for $f_{\mathrm{Rabi}}=2$ MHz')

# plt.tight_layout()
plt.show()

# %% plot at each position x,y a circle with color value subnumber

fig, ax = plt.subplots(1, 1, figsize=(4, 4))

color_prop = 'larmor_frequency'  # color by subnumber
text_prop = 'subnumber'  # text label by plunger name
norm = plt.Normalize(df_central[color_prop].min(), df_central[color_prop].max())
for name, row in df.iterrows():
    color = plt.cm.viridis(norm(row[color_prop]))  # normalize larmor frequency to [0, 1]
    circle = plt.Circle((row['x'], row['y']), 0.06, color=color, label=row['plunger_name'])
    ax.add_artist(circle)
    # ax.text(row['x'], row['y'], row[text_prop], fontsize=8, ha='center', va='center',
    #          color='white')
ax.set_xlim(-0.1, 2.3)
ax.set_ylim(-0.1, 2.3)
ax.axis('off')
ax.set_aspect('equal', adjustable='box')
# ax.title('Plunger Positions with Subnumber Color Coding')
ax.set_xlabel('X Position (m)')
ax.set_ylabel('Y Position (m)')

# add colorbar with non normalized larmor frequency
sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, pad=0.02, ax=ax, location='bottom', aspect=50, shrink=0.8)
cbar.set_label('Larmor Frequency (MHz)', labelpad=5)

# plt.tight_layout()
plt.savefig('qubit_array_frequencies.pdf', dpi=300, transparent=True)
plt.show()

# %%
import numpy as np

best_frequencies_tile = [400.00, 200.00, 381.04, 216.91,
                         253.21, 331.23, 268.83, 347.48,
                         381.04, 216.91, 400.00, 200.00,
                         268.83, 347.48, 253.21, 331.23]

# weakest link
larmor_nearest_neighbour_diff_axis_one = np.diff(np.array(best_frequencies_tile).reshape(-1, 4), axis=1)
larmor_nearest_neighbour_diff_axis_zero = np.diff(np.array(best_frequencies_tile).reshape(-1, 4), axis=0)