import sys
import subprocess

PYTHON = sys.executable

extract_data = False # extracts g-factor tunability and drive efficiency data, AND plots them. Takes a long time.

def main():
    print("Supplementary Note 1:")
    print("Read-out data")
    subprocess.run([PYTHON, "read_out/read_out.py"], check=True)

    print("Supplementary Note 2:")
    print("Randomised benchmarking data")
    subprocess.run([PYTHON, "single_qubit_RB/plot_rb_1Q.py"], check=True)
    print("Visibilities and offsets")
    subprocess.run([PYTHON, "visibility/visibilities.py"], check=True)

    print("Supplementary Note 3:")
    print("in progress ...")

    print("Supplementary Note 4:")
    print("Exchange Interaction data")
    subprocess.run([PYTHON, "exchange_splitting/plot_exchange_splitting.py"], check=True)

    print("Supplementary Note 7:")
    print("EDSR efficiency data")
    print("in progress ...")

    print("Supplementary Note 8:")
    print("g-factor tunability data")
    print("in progress ...")

    if extract_data:
        print("Supplementary Note 9:")
        print("Extract g-factor tunability data for all qubits. This can take quite a while.")
        subprocess.run([PYTHON, "gfactor_tunability/extract_statistics_gfactor_tunability_multiple_holes.py"], check=True)
        print("Done extracting g-factor tunability data.")

        print("Supplementary Note 10:")
        print("Extract drive efficiency data for all qubits, hole occupations and gates. This can take quite a while.")
        subprocess.run([PYTHON, "drive_efficiency/extract_statistics_driving_efficiency_multiple_holes.py"], check=True)
        print("Done extracting drive efficiency data.")

    print("Supplementary Note 11:")
    print("Physical distance from gates to qubits")
    subprocess.run([PYTHON, "drive_locality/nearest_neighbours.py"], check=True)

    print("Supplementary Note 12:")
    print("Spatial locality of electric field and qubit drive")
    subprocess.run([PYTHON, "electric_field/electric_field.py"], check=True)

    print("Supplementary Note 13:")
    print("Frequency cross-talk")
    subprocess.run([PYTHON, "qubit_array_frequencies/qubit_array_frequencies.py"], check=True)

if __name__ == '__main__':
    main()
