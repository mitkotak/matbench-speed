import ase
from ase.build import bulk
from ase.md.langevin import Langevin
from ase.md.verlet import VelocityVerlet
from ase import units
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

    atoms = ase.Atoms(
        "OHH"*8,
        positions = [
            [1, 1, 1],
            [2, 1, 1],
            [1, 2, 1],
            [4, 1, 1],
            [5, 1, 1],
            [4, 2, 1],
            [1, 4, 1],
            [2, 4, 1],
            [1, 5, 1],
            [1, 1, 4],
            [2, 1, 4],
            [1, 2, 4],
            [4, 4, 1],
            [5, 4, 1],
            [4, 5, 1],
            [4, 1, 4],
            [5, 1, 4],
            [4, 2, 4],
            [1, 4, 4],
            [2, 4, 4],
            [1, 5, 4],
            [4, 4, 4],
            [5, 4, 4],
            [4, 5, 4],
        ],
        cell = [6.2085633514918648, 6.2085633514918648, 6.2085633514918648],
        pbc = True,
    )
    atoms = atoms.repeat((size, size, size))
    atoms.calc = calc

    print("Number of atoms: ", len(atoms))

    # warmup
    i_warmup = 0
    t_warmup = 0
    while i_warmup < 100 and t_warmup < 10:

        t_start = time.time()
        dyn = Langevin(
            atoms,
            timestep = 1.0 * units.fs,
            temperature_K = 300.0,
            friction = 1.0 / (10 * units.fs),
        )
        dyn.run(10)
        t_end = time.time()
        t_warmup += (t_end - t_start)
        i_warmup += 1

    # measurements
    i_timing = 0
    t_timing = 0
    times = []
    while i_timing < 100 and t_timing < 60:
        # thermalize briefly to sample different energies
        dyn = Langevin(
            atoms,
            timestep = 1.0 * units.fs,
            temperature_K = 300.0,
            friction = 1.0 / (10 * units.fs),
        )
        dyn.run(10)
        # perform measurement at constant energy
        t_start = time.time()
        dyn = VelocityVerlet(
            atoms,
            1.0 * units.fs
        )
        dyn.run(100)
        t_end = time.time()
        i_timing += 1
        t_timing += (t_end - t_start)
        times.append(t_end - t_start)


    print("Testing new MD workflow, completed ", i_timing, " timing runs in ", t_timing, " seconds")
    return np.median(times), np.std(times), len(atoms)

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
