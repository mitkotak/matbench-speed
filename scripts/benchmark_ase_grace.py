# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pip-system-certs",
#     "ase",
#     "jax[cuda12]",
#     "tensorpotential",
#     "matbench_speed",
# ]
#
# [tool.uv.sources]
# matbench_speed = { path = "../." }
# ///

from matbench_speed.benchmark import benchmark
from tensorpotential.calculator import grace_fm

calculators = {
    "GRACE-2L-MPtrj":  grace_fm(
        model="GRACE-2L-MP-r6",
    ),
}

benchmark(calculators)
