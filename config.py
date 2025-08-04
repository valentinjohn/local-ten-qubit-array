from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / 'data'

all_qubits = [f'Q{i}' for i in range(1, 11)]
all_plungers = [f'P{i}' for i in range(1, 11)]
all_barriers = [f'B{i}' for i in range(1, 13)]
