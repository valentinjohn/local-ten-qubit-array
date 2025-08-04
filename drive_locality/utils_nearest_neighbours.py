# %% Path management

from pathlib import Path
from config import DATA_DIR, PROJECT_DIR

# %% imports

from utils import *

# %%

def generate_centered_grid(rows, cols, spacing, flatten=True):
    # Calculate the total width and height of the grid
    total_width = (cols - 1) * spacing
    total_height = (rows - 1) * spacing

    # Generate the grid
    grid = []
    for row in range(rows):
        grid_row = []
        for col in range(cols):
            x = col * spacing - total_width / 2
            y = row * spacing - total_height / 2
            grid_row.append((x, y))
        grid.append(grid_row)

    if flatten:
        grid = np.reshape(grid, (rows * cols, 2))

    return grid

def create_empty_mask_dataframe():
    """
    Creates a DataFrame with columns ['P1', 'P2', ..., 'P10', 'B1', 'B2', ..., 'B12'] and
    rows ['P1', 'P2', ..., 'P10'], filled with False values.

    Returns:
    - A DataFrame filled with False values.
    """
    # Define row and column labels
    row_labels = [f'P{i}' for i in range(1, 11)]
    col_labels = [f'P{i}' for i in range(1, 11)] + [f'B{i}' for i in range(1, 13)]

    # Create DataFrame filled with False values
    mask_df = pd.DataFrame(0.0, index=row_labels, columns=col_labels)
    return mask_df

coordinates_barriers = []
n_col = 3
spacing = 0.195 # dot-to-dot distance
spacing_hv = 2**0.5*spacing
bar_offset = -0.002
for i in range(n_col):
    (x,y) = ((i-3/4)*spacing_hv+bar_offset, spacing_hv/4+bar_offset)
    coordinates_barriers.append([-x,y])
    coordinates_barriers.append([x,y])
    coordinates_barriers.append([-x,-y])
    coordinates_barriers.append([x,-y])

coordinates_barriers = np.sort(coordinates_barriers, axis=0)
coordinates_barriers = np.sort(coordinates_barriers, axis=1)

# %%

qubit_spacing = 0.195

plunger_polygon = np.array([[-0.019, -0.065],
                           [-0.024, -0.064],
                           [-0.029, -0.062],
                           [-0.033, -0.059],
                           [-0.059, -0.033],
                           [-0.062, -0.029],
                           [-0.064, -0.024],
                           [-0.065, -0.019],
                           [-0.065,  0.019],
                           [-0.064,  0.024],
                           [-0.062,  0.029],
                           [-0.059,  0.033],
                           [-0.033,  0.059],
                           [-0.029,  0.062],
                           [-0.024,  0.064],
                           [-0.019,  0.065],
                           [ 0.019,  0.065],
                           [ 0.024,  0.064],
                           [ 0.029,  0.062],
                           [ 0.033,  0.059],
                           [ 0.059,  0.033],
                           [ 0.062,  0.029],
                           [ 0.064,  0.024],
                           [ 0.065,  0.019],
                           [ 0.065, -0.019],
                           [ 0.064, -0.024],
                           [ 0.062, -0.029],
                           [ 0.059, -0.033],
                           [ 0.033, -0.059],
                           [ 0.029, -0.062],
                           [ 0.024, -0.064],
                           [ 0.019, -0.065]])

barrier_polygon = {}
barrier_polygon['vertical'] = np.array([[-0.025, -0.045],
                                        [-0.025, 0.045],
                                        [0.025, 0.045],
                                        [0.025, -0.045]])

barrier_polygon['horizontal'] = np.array([[-0.045, -0.025],
                                          [-0.045, 0.025],
                                          [0.045, 0.025],
                                          [0.045, -0.025]])

barrier_coordinates = {}
barrier_coordinates['plunger_drive'] = {}
barrier_coordinates['barrier_drive'] = {}
barrier_coordinates['plunger_drive']['vertical'] = generate_centered_grid(5, 4, qubit_spacing)
barrier_coordinates['plunger_drive']['horizontal'] = generate_centered_grid(4, 5, qubit_spacing)
barrier_coordinates['barrier_drive']['vertical'] = generate_centered_grid(5, 3, qubit_spacing)
barrier_coordinates['barrier_drive']['horizontal'] = generate_centered_grid(4, 4, qubit_spacing)

coordinates = {'P1': np.array([-2, 1]),
               'P2': np.array([0, 1]),
               'P3': np.array([2, 1]),
               'P4': np.array([-3,0]),
               'P5': np.array([-1,0]),
               'P6': np.array([1,0]),
               'P7': np.array([3,0]),
               'P8': np.array([-2,-1]),
               'P9': np.array([0,-1]),
               'P10': np.array([2,-1]),
               'B1': np.array([-2.5,0.5]),
               'B2': np.array([-1.5,0.5]),
               'B3': np.array([-0.5,0.5]),
               'B4': np.array([0.5,0.5]),
               'B5': np.array([1.5,0.5]),
               'B6': np.array([2.5,0.5]),
               'B7': np.array([-2.5,-0.5]),
               'B8': np.array([-1.5,-0.5]),
               'B9': np.array([-0.5,-0.5]),
               'B10': np.array([0.5,-0.5]),
               'B11': np.array([1.5,-0.5]),
               'B12': np.array([2.5,-0.5])}


qubit_steps_drive = {}
qubit_steps_drive['plunger'] = {1: [[0,0]],
                                2: [[1,0], [-1,0], [0,1], [0,-1]],
                                3: [[1,1],[-1,1],[1,-1],[-1,-1]],
                                4: [[2,0],[-2,0],[0,2],[0,-2]],
                                5: [[1,2],[-1,2],[1,-2],[-1,-2], [2,1],[-2,1],[2,-1],[-2,-1]],
                                6: [[2,2],[-2,2],[2,-2],[-2,-2]]
                                }

qubit_steps_drive['barrier'] = {1: [[0.5,0], [-0.5,0]],
                                2: [[0.5,1], [-0.5,1], [0.5,-1], [-0.5,-1]],
                                3: [[1.5,0], [-1.5,0]],
                                4: [[1.5,1], [-1.5,1], [1.5,-1], [-1.5,-1]],
                                5: [[0.5,2], [-0.5,2], [0.5,-2], [-0.5,-2]],
                                6: [[1.5,2], [-1.5,2], [1.5,-2], [-1.5,-2]],
                                }

qubit_distances_drive = {}
qubit_distances_drive['plunger'] = {}
qubit_distances_drive['barrier'] = {}

qubit_coordinates_drive = {}
qubit_coordinates_drive['plunger'] = {}
qubit_coordinates_drive['barrier'] = {}

qubit_drive_polygons = {}
qubit_drive_polygons['plunger'] = {}
qubit_drive_polygons['barrier'] = {}
# calculate distances to origin

for gate_type in ['plunger', 'barrier']:
    for step, direction in qubit_steps_drive[gate_type].items():
        qubit_distances_drive[gate_type][step] = []
        qubit_coordinates_drive[gate_type][step] = qubit_spacing*np.array(direction)
        for coord in direction:
            distance = qubit_spacing*np.linalg.norm(coord)
            qubit_distances_drive[gate_type][step].append(np.round(distance,2))

        qubit_drive_polygons[gate_type][step] = []
        for coord in direction:
            qubit_drive_polygons[gate_type][step].append(plunger_polygon + np.array(coord)*qubit_spacing)

# %%

df_distances = create_empty_mask_dataframe()

for pl_gate in all_plungers:
    for gate in all_gates:
        dist = 0.276/2*np.linalg.norm(coordinates[pl_gate] - coordinates[gate])
        df_distances.loc[pl_gate, gate] = np.round(dist, 2)

unique_plunger_distances = np.unique(df_distances[all_plungers].to_numpy())
unique_barrier_distances = np.unique(df_distances[all_barriers].to_numpy())

def rank_replace(df):
    # ranks and replaces values in a DataFrame
    flattened = df.values.flatten()
    ranked = pd.Series(flattened).rank(method='dense').astype(int)
    ranked_df = pd.DataFrame(ranked.values.reshape(df.shape), index=df.index, columns=df.columns)
    return ranked_df

df_distances_ranked = df_distances.copy()
df_distances_ranked[all_plungers] = rank_replace(df_distances_ranked[all_plungers])
df_distances_ranked[all_barriers] = rank_replace(df_distances_ranked[all_barriers])


# %% definitions

def get_locality_data(larmor_norm=False):
    results = pd.read_csv(DATA_DIR / 'final_result_summary.csv')
    results = results.set_index('label')

    df_distances_ranked = pd.read_csv(DATA_DIR / 'distances_ranked.csv')

    df_all = {}
    df_all['1h'] = pd.read_csv(DATA_DIR / 'driving_strength_1hole.csv',
                               header=[0], index_col=[0])
    df_all['3h'] = pd.read_csv(DATA_DIR / 'driving_strength_3hole.csv',
                               header=[0], index_col=[0])

    df_drive_eff = {'1h': {}, '3h': {}}
    drive_eff = {'1h': {}, '3h': {}}

    for hole in ['1h', '3h']:
        df = df_all[hole]

        df_distances_ranked = df_distances_ranked.set_index(df.index)

        df_drive_eff_all_gates = df.copy()
        for qubit in df_drive_eff_all_gates.index:
            f_larmor = results.loc['Larmor frequency (MHz)'][qubit]
            df_drive_eff_all_gates.loc[qubit] = df_drive_eff_all_gates.loc[qubit]
            if larmor_norm:
                df_drive_eff_all_gates.loc[qubit] = df_drive_eff_all_gates.loc[qubit] / (f_larmor)

        df_drive_eff[hole]['plunger'] = {}
        df_drive_eff[hole]['barrier'] = {}

        drive_eff[hole]['plunger'] = {}
        drive_eff[hole]['barrier'] = {}

        for rank in qubit_steps_drive['plunger'].keys():
            df_drive_eff[hole]['plunger'][rank] = df_drive_eff_all_gates[all_plungers][
                df_distances_ranked[all_plungers] == rank]
            drive_eff[hole]['plunger'][rank] = {}
            drive_eff[hole]['plunger'][rank]['mean'] = df_drive_eff[hole]['plunger'][rank].mean().mean()
            drive_eff[hole]['plunger'][rank]['std'] = df_drive_eff[hole]['plunger'][rank].mean().std()

        for rank in qubit_steps_drive['barrier'].keys():
            df_drive_eff[hole]['barrier'][rank] = df_drive_eff_all_gates[all_barriers][
                df_distances_ranked[all_barriers] == rank]
            drive_eff[hole]['barrier'][rank] = {}
            drive_eff[hole]['barrier'][rank]['mean'] = df_drive_eff[hole]['barrier'][rank].mean().mean()
            drive_eff[hole]['barrier'][rank]['std'] = df_drive_eff[hole]['barrier'][rank].mean().std()

    return df_drive_eff, drive_eff