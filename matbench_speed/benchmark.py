import ase
from ase.build import bulk
import time
import csv
import os
import sys
import numpy as np

def write_to_csv(model_name, atoms, time_ms, time_std, precision, gpu_name, csv_filename="./data/timing_data.csv"):
    """Append timing data to CSV file"""
    file_exists = os.path.isfile(csv_filename)

    with open(csv_filename, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["model", "atoms", "time", "std", "precision", "gpu"])
        writer.writerow([model_name, atoms, time_ms, time_std, precision, gpu_name])

def benchmark_size(size, calc):
    atoms = ase.build.bulk("Si", "diamond", a=5.43, cubic=True)
    atoms = atoms.repeat((size, size, size))
    num_atoms = len(atoms)
    print("Number of atoms: ", num_atoms)
    atoms.calc = calc

    # warmup
    i_warmup = 0
    t_warmup = 0

    while i_warmup < 20 and t_warmup < 1:

        t_start = time.time()

        E = atoms.get_potential_energy()
        F = atoms.get_forces()
        S = atoms.get_stress()
        atoms.rattle()

        t_end = time.time()
        t_warmup += (t_end - t_start)
        i_warmup += 1
        

    i_timing = 0
    t_timing = 0
    times = []

    while i_timing < 100 and t_timing  < 5:

        t_start = time.time()

        E = atoms.get_potential_energy()
        F = atoms.get_forces()
        S = atoms.get_stress()
        atoms.rattle()

        t_end = time.time()
        t_timing += (t_end - t_start)
        i_timing += 1
        times.append(t_end - t_start)

    return np.median(times), np.std(times), num_atoms

def get_gpu_name():
    if 'torch' in sys.modules:
        import torch
        return torch.cuda.get_device_name(0)
    elif 'jax' in sys.modules:
        import jax
        return jax.devices()[0].device_kind
    else:
        raise ValueError("No GPU found")

def benchmark(calculators, precision="float32"):

    gpu_name = get_gpu_name()

    for model, calculator in calculators.items():
        for i, size in enumerate([1,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]):
        # for i, size in enumerate([1,6,]):
            time_s, time_std, num_atoms = benchmark_size(size, calculator)
            time_ms, time_std_ms = time_s * 1000, time_std * 1000
            if i == 0:
                # extra warmup to be safe
                continue
            print(model, ": ", "atoms: ", num_atoms, "time: ", time_ms, " +/- ", time_std_ms, " ms")
            write_to_csv(model, num_atoms, time_ms, time_std_ms, precision, gpu_name)
