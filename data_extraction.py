import sys
import subprocess

PYTHON = sys.executable

def main():
    print("Extract drive efficiency data for all qubits, hole occupations and gates. This can take quite a while.")
    subprocess.run([PYTHON, "drive_efficiency/extract_statistics_driving_efficiency_multiple_holes.py"], check=True)
    print("Done extracting drive efficiency data.")

    print("Extract g-factor tunability data for all qubits. This can take quite a while.")
    subprocess.run([PYTHON, "gfactor_tunability/extract_statistics_gfactor_tunability_multiple_holes.py"], check=True)
    print("Done extracting g-factor tunability data.")

if __name__ == '__main__':
    main()
