# %%

def convert_table_to_colored_cells(table_string, cellcolor_name):
    # Split lines of the table
    lines = table_string.strip().split("\n")
    converted_lines = []

    for line in lines:
        # Split each line into columns
        columns = line.split("&")
        # Apply \cellcolor to each value, ignoring the first column (row headers)
        colored_columns = [columns[0].strip()] + [
            f"\\{cellcolor_name}"+"{"+f"{value.strip()}"+'}' for value in columns[1:]
        ]
        # Rejoin the line with LaTeX column separators
        converted_lines.append(" & ".join(colored_columns) + " \\\\")

    # Join all lines into a single string
    return "\n".join(converted_lines)

# %%

input_table = """
\textbf{Q1}  & 0.0         & 0.28        & 0.55        & 0.2         & 0.2         & 0.44        & 0.7         & 0.28        & 0.39        & 0.62         & 0.1         & 0.1         & 0.22        & 0.35        & 0.49        & 0.62        & 0.22        & 0.22        & 0.29        & 0.4          & 0.53         & 0.65         \\ \hline
\textbf{Q2}  & 0.28        & 0.0         & 0.28        & 0.44        & 0.2         & 0.2         & 0.44        & 0.39        & 0.28        & 0.39         & 0.35        & 0.22        & 0.1         & 0.1         & 0.22        & 0.35        & 0.4         & 0.29        & 0.22        & 0.22         & 0.29         & 0.4          \\ \hline
\textbf{Q3}  & 0.55        & 0.28        & 0.0         & 0.7         & 0.44        & 0.2         & 0.2         & 0.62        & 0.39        & 0.28         & 0.62        & 0.49        & 0.35        & 0.22        & 0.1         & 0.1         & 0.65        & 0.53        & 0.4         & 0.29         & 0.22         & 0.22         \\ \hline
\textbf{Q4}  & 0.2         & 0.44        & 0.7         & 0.0         & 0.28        & 0.55        & 0.83        & 0.2         & 0.44        & 0.7          & 0.1         & 0.22        & 0.35        & 0.49        & 0.62        & 0.76        & 0.1         & 0.22        & 0.35        & 0.49         & 0.62         & 0.76         \\ \hline
\textbf{Q5}  & 0.2         & 0.2         & 0.44        & 0.28        & 0.0         & 0.28        & 0.55        & 0.2         & 0.2         & 0.44         & 0.22        & 0.1         & 0.1         & 0.22        & 0.35        & 0.49        & 0.22        & 0.1         & 0.1         & 0.22         & 0.35         & 0.49         \\ \hline
\textbf{Q6}  & 0.44        & 0.2         & 0.2         & 0.55        & 0.28        & 0.0         & 0.28        & 0.44        & 0.2         & 0.2          & 0.49        & 0.35        & 0.22        & 0.1         & 0.1         & 0.22        & 0.49        & 0.35        & 0.22        & 0.1          & 0.1          & 0.22         \\ \hline
\textbf{Q7}  & 0.7         & 0.44        & 0.2         & 0.83        & 0.55        & 0.28        & 0.0         & 0.7         & 0.44        & 0.2          & 0.76        & 0.62        & 0.49        & 0.35        & 0.22        & 0.1         & 0.76        & 0.62        & 0.49        & 0.35         & 0.22         & 0.1          \\ \hline
\textbf{Q8}  & 0.28        & 0.39        & 0.62        & 0.2         & 0.2         & 0.44        & 0.7         & 0.0         & 0.28        & 0.55         & 0.22        & 0.22        & 0.29        & 0.4         & 0.53        & 0.65        & 0.1         & 0.1         & 0.22        & 0.35         & 0.49         & 0.62         \\ \hline
\textbf{Q9}  & 0.39        & 0.28        & 0.39        & 0.44        & 0.2         & 0.2         & 0.44        & 0.28        & 0.0         & 0.28         & 0.4         & 0.29        & 0.22        & 0.22        & 0.29        & 0.4         & 0.35        & 0.22        & 0.1         & 0.1          & 0.22         & 0.35         \\ \hline
\textbf{Q10} & 0.62        & 0.39        & 0.28        & 0.7         & 0.44        & 0.2         & 0.2         & 0.55        & 0.28        & 0.0          & 0.65        & 0.53        & 0.4         & 0.29        & 0.22        & 0.22        & 0.62        & 0.49        & 0.35        & 0.22         & 0.1          & 0.1          \\ \hline
"""

result = convert_table_to_colored_cells(input_table, cellcolor_name='colorcelldis')
print(result)

# %%



