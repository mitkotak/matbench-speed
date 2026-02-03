# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "ase",
#     "torch",
#     "sevenn>=0.12.0",
#     "cuequivariance-torch==0.7.0",
#     "cuequivariance-ops-torch-cu12==0.7.0",
#     "matbench_speed",
# ]
#
# [tool.uv.sources]
# sevenn = {git = "https://github.com/MDIL-SNU/SevenNet.git", branch="main"}
# matbench_speed = { path = "../." }
# ///

from matbench_speed.benchmark import benchmark
from sevenn.calculator import SevenNetCalculator
from sevenn.util import model_from_checkpoint, pretrained_name_to_path
import torch

calculators = {
    "SevenNet-l3i5": SevenNetCalculator("7net-l3i5", enable_cueq=True, enable_flash=False),
}
benchmark(calculators, atom_name="H20")
