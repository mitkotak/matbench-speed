# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "ase",
#     "nequix[torch]",
# ]
#
# ///

from matbench_speed.benchmark import benchmark
from nequix.calculator import NequixCalculator

calculators = {
    "Nequix-MP-1": NequixCalculator(
        model_path="./models/nequix-mp-1.nqx",
        backend="jax",
    ),
    "Nequix-MP-1.5": NequixCalculator(
        model_path="./models/nequix-mp-1.pt",
        backend="torch",
    ),
}
benchmark(calculators)
