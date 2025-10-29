# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pip-system-certs",
#     "ase",
#     "mace-torch",
#     "torch==2.7.0",
#     "cuequivariance-torch",
#     "cuequivariance-ops-torch-cu12",
#     "matbench_speed",
# ]
#
# [tool.uv.sources]
# matbench_speed = { path = "../." }
# ///

from matbench_speed.benchmark import benchmark
from mace.calculators.foundations_models import mace_mp

calculators = {
    "MACE-MP-0": mace_mp(
        model="medium",
        default_dtype="float32",
        device="cuda",
        compile_mode=True,
        fullgraph=False,
        enable_cueq=True,
    ),
}
benchmark(calculators)
