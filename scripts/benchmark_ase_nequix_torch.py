# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "ase",
#     "nequix[torch]",
#     "matbench_speed",
# ]
#
# [tool.uv.sources]
# matbench_speed = { path = "../." }
# ///

from matbench_speed.benchmark import benchmark
from nequix.calculator import NequixCalculator

calculators = {
    "Nequix-MP-1.5": NequixCalculator(
        model_name="nequix-mp-1",
        backend="torch",
    ),
}
benchmark(calculators)
