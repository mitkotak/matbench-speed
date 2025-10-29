# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "ase",
#     "torch",
#     "sevenn",
#     "flashTP_e3nn",
#     "matbench_speed",
# ]
#
# [tool.uv.sources]
# sevenn = { git = "https://github.com/MDIL-SNU/SevenNet.git", branch = "flash" }
# flashTP_e3nn = { git = "https://github.com/SNU-ARC/flashTP", branch = "main" }
# matbench_speed = { path = "../." }
# ///

from matbench_speed.benchmark import benchmark
from sevenn.calculator import SevenNetCalculator
from sevenn.util import model_from_checkpoint, pretrained_name_to_path

calculators = {
    "SevenNet-l3i5": SevenNetCalculator(
        model_from_checkpoint(pretrained_name_to_path("7net-l3i5"), enable_flash=True)[0],
    ),
}
benchmark(calculators)
