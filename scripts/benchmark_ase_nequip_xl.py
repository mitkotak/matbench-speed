# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "ase",
#     "nequip",
#     "torch==2.9.1",
#     "openequivariance==0.4.1",
#     "matbench_speed",
# ]
#
# [tool.uv.sources]
# matbench_speed = { path = "../." }
# ///

import os
from matbench_speed.benchmark import benchmark
from nequip.ase import NequIPCalculator
import openequivariance  # Required: must import before loading compiled model

if not os.path.isfile("./models/mir-group__NequIP-OAM-XL__0.1.nequip.pt2"):
    os.system("nequip-compile nequip.net:mir-group/NequIP-OAM-XL:0.1 ./models/mir-group__NequIP-OAM-XL__0.1.nequip.pt2 --mode aotinductor --device cuda --target ase --modifiers enable_OpenEquivariance")

calculators = {
    # The NequIP MPTrj and OAM models have the same hyperparams and since only the OAM
    # has kernels we use it.
    "NequIP-MP-XL": NequIPCalculator.from_compiled_model(
       compile_path="./models/mir-group__NequIP-OAM-XL__0.1.nequip.pt2",
       device="cuda",
    )
}
benchmark(calculators, atom_name="Si", lattice_constant=5.43)
