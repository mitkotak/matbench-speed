# /// script
# requires-python = ">=3.10"
# dependencies = [ 
#    "pip-system-certs",
#    "setuptools",
#     "ase",
#     "nequix[torch]",
#     "matbench_speed",
# ]
#
# [tool.uv.sources]
# nequix = { path = "../../nequix_main" }
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
benchmark(calculators, atom_name="H20")
