from pathlib import Path
import ase
from ase.build import bulk
import time
import csv
import os
import sys
import numpy as np

def write_to_csv(model_name, atoms, time_ms, time_std, gpu_name, csv_filename="./data/timing_data.csv"):
    """Append timing data to CSV file"""
    file_exists = os.path.isfile(csv_filename)

    with open(csv_filename, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["model", "atoms", "time", "std", "gpu"])
        writer.writerow([model_name, atoms, time_ms, time_std, gpu_name])

def benchmark_size(size, calc):
    atoms = ase.build.bulk("C", "diamond", a=3.567, cubic=True)
    atoms = atoms.repeat((size, size, size))
    num_atoms = len(atoms)
    print("Number of atoms: ", num_atoms)
    atoms.calc = calc
    # warmup
    for _ in range(10):
        E = atoms.get_potential_energy()
        atoms.rattle()

    times = []
    for _ in range(100):
        start = time.time()
        E = atoms.get_potential_energy()
        # print(E)
        atoms.rattle()
        times.append(time.time() - start)
    return np.median(times), np.std(times), num_atoms

def torch_profile_size(model, size, calc):
    import torch
    from torch.profiler import (
    profile,
    ProfilerActivity,
    schedule,
    record_function,
    ) 

    atoms = ase.build.bulk("C", "diamond", a=3.567, cubic=True)
    atoms = atoms.repeat((size, size, size))
    num_atoms = len(atoms)
    atoms.calc = calc
    # warmup
    for _ in range(10):
        E = atoms.get_potential_energy()
        atoms.rattle()
    gpu_name = get_gpu_name()
    profile_output_path = Path(f"./profiles/{gpu_name}/{model}/{num_atoms}")
    profile_output_path.mkdir(parents=True, exist_ok=True)

    prof_sched = schedule(wait=2, warmup=3, active=5, repeat=1)  # short & light
    total_steps = 2 + 3 + 5

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=prof_sched,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for _ in range(total_steps):
            with record_function("model.get_potential_energy"):
                _ = atoms.get_potential_energy()
            atoms.rattle()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            prof.step()
        
        prof.export_chrome_trace(str(profile_output_path / "trace"))


def get_gpu_name():
    if 'torch' in sys.modules:
        import torch
        return torch.cuda.get_device_name(0)
    elif 'jax' in sys.modules:
        import jax
        return jax.devices()[0].device_kind
    else:
        raise ValueError("No GPU found")

def benchmark(calculators):
    
    gpu_name = get_gpu_name()

    for model, calculator in calculators.items():
        for i, size in enumerate([1, 1, 2, 3, 4, 5, 6, 7, 8, 9]):
            time_s, time_std, num_atoms = benchmark_size(size, calculator)
            time_ms, time_std_ms = time_s * 1000, time_std * 1000
            if i == 0:
                # extra warmup to be safe
                continue
            print(model, ": ", "atoms: ", num_atoms, "time: ", time_ms, " +/- ", time_std_ms, " ms")
            write_to_csv(model, num_atoms, time_ms, time_std_ms, gpu_name)

            if 'torch' in sys.modules:
                torch_profile_size(model, size, calculator)
