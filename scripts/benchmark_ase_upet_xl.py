# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pip-system-certs",
#     "ase",
#     "upet",
#     "matbench_speed",
# ]
#
# [tool.uv.sources]
# matbench_speed = { path = "../." }
# ///

from matbench_speed.benchmark import benchmark
from upet.calculator import UPETCalculator

calculators = {
    "PET-OAM-XL": UPETCalculator(
        "pet-oam-xl",
        version="1.0.0",
        device="cuda"
    ),
}
benchmark(calculators, atom_name="Si", lattice_constant=5.43)
