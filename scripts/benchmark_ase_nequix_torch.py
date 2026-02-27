# /// script
# requires-python = ">=3.10"
# dependencies = [
#    "pip-system-certs",
#     "ase",
#     "nequix[torch]",
#     "torch==2.7.0+cu128",
#     "matbench_speed",
# ]
#
# [tool.uv.sources]
# torch = { url = "https://download.pytorch.org/whl/cu128/torch-2.7.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl" }
# matbench_speed = { path = "../." }
# ///

from matbench_speed.benchmark import benchmark
from nequix.calculator import NequixCalculator

calculators = {
    "Nequix-MP-1": NequixCalculator(
        model_name="nequix-mp-1",
        backend="torch",
    ),
}
benchmark(calculators, atom_name="Si", lattice_constant=5.43)
