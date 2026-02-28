# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pip-system-certs",
#     "ase",
#     "mace-torch",
#     "symmetrix",
#     "matbench_speed",
# ]
#
# [tool.uv.sources]
# symmetrix = { git = "https://github.com/wcwitt/symmetrix", branch = "main", subdirectory = "symmetrix" }
# matbench_speed = { path = "../." }
# ///

from tempfile import TemporaryDirectory
import urllib.request

from matbench_speed.benchmark import benchmark
from symmetrix import Symmetrix

with TemporaryDirectory() as temp_dir:

    urllib.request.urlretrieve(
        "https://github.com/ACEsuit/mace-foundations/releases/download/mace_mp_0b3/mace-mp-0b3-medium.model",
        temp_dir + "mace-mp-0b3-medium.model")
    calculators = {
        "MACE-MP-0-Symmetrix": Symmetrix(
            temp_dir + "mace-mp-0b3-medium.model",
            use_kokkos = True,
            dtype = "float32",
            species=["C"]
        )
    }
    benchmark(calculators)
