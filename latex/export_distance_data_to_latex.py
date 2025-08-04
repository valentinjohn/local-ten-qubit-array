# %% Path management

from pathlib import Path
from config import DATA_DIR, all_qubits

try:
    script_dir = Path(__file__).resolve().parent
except NameError:
    script_dir = Path().resolve() / 'latex'
fig_path = script_dir / "images"

# %% imports

import pandas as pd

# %% definitions

def export_df_to_latex(df, output_path, cell_comand=None):
    # Prepare LaTeX output
    def format_row(row, header=False):
        if header:
            return " & ".join([f"\\textbf{{{col}}}" for col in row]) + r" \\ \hline"
        else:
            return " & ".join(row) + r" \\ \hline"

    # Create LaTeX lines
    latex_lines = []

    # Header row
    latex_lines.append(format_row([""] + df.columns.tolist(), header=True))

    # Data rows
    if cell_comand is not None:
        for index, row in df.iterrows():
            latex_lines.append(
                format_row([f"\\textbf{{{index}}}"] + [f"\\{cell_comand}{{{val}}}" for val in row.astype(str)]))
    else:
        for index, row in df.iterrows():
            latex_lines.append(format_row([f"\\textbf{{{index}}}"] + [str(val) for val in row.astype(str)]))

    latex_table = "\n".join(latex_lines)
    latex_table[:1000]  # Preview the beginning of the generated LaTeX table

    # save to a .tex file
    with open(output_path, "w") as f:
        f.write(latex_table)

# %% Load the CSV file and export to LaTeX

# Load the CSV file
data_path = DATA_DIR / "distance.csv"

transpose = True
df = pd.read_csv(data_path, index_col=0)
if transpose:
    df = df.transpose()
    output_path = script_dir / "distance_transpose.tex"
    df.columns = all_qubits
else:
    output_path = script_dir / "distance.tex"
    df.index = all_qubits

export_df_to_latex(df, output_path, cell_comand='colorcelldis')


# Load the CSV file
data_path = DATA_DIR / "distances_ranked.csv"

transpose = True
df = pd.read_csv(data_path, index_col=0)
if transpose:
    df = df.transpose()
    output_path = script_dir / "distances_ranked_transpose.tex"
    df.columns = all_qubits
else:
    output_path = script_dir / "distances_ranked.tex"
    df.index = all_qubits

export_df_to_latex(df, output_path, cell_comand='colorcell')