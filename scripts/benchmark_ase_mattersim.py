# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pip-system-certs",
#     "ase",
#     "aoti_mlip",
#     "torch==2.9.1",
#     "matbench-speed",
# ]
#
# [tool.uv.sources]
# matbench_speed = { path = "../." }
# aoti_mlip = { git = "https://github.com/abhijeetgangan/aoti_mlip.git" }
# ///

from matbench_speed.benchmark import benchmark
from aoti_mlip.utils.aoti_compile import compile_mattersim
from aoti_mlip.calculators.mattersim import MatterSimCalculator

package_path = compile_mattersim(
    checkpoint_name="mattersim-v1.0.0-1M.pth",
    cutoff=5.0,
    threebody_cutoff=4.0,
    compute_force=True,
    compute_stress=True,
    device="cuda"
)

calculators = {
    "MatterSim-1M": MatterSimCalculator(
         model_path="~/.local/mattersim/pretrained_models/mattersim-v1.0.0-1M.pt2",
         device="cuda"
    )
}
benchmark(calculators)
