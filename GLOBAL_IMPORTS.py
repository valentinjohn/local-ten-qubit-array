from pathlib import Path

file_path = Path(__file__)

folder = file_path.parent
data_folder = file_path.parent / 'data'
