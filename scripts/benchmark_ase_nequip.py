# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "ase",
#     "nequip",
#     "torch==2.9.0",
#     "openequivariance==0.4.1",
#     "matbench_speed",
# ]
#
# [tool.uv.sources]
# torch = { url = "https://download.pytorch.org/whl/test/cu128" }
# matbench_speed = { path = "../." }
# ///

from matbench_speed.benchmark import benchmark
from nequip.ase import NequIPCalculator
import openequivariance  # Required: must import before loading compiled model

calculators = {
    # The NequIP MPTrj and OAM models have the same hyperparams and since only the OAM
    # has kernels we use it.
    "NequIP-MP-L": NequIPCalculator.from_compiled_model(
        compile_path="./models/mir-group__NequIP-OAM-L__0.1.nequip.pt2",
        device="cuda",  # or "cpu"
    ),
}
benchmark(calculators)
