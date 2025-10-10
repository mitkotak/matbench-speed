# /// script
# requires-python = "==3.12"
# dependencies = [
#     "fairchem-core==1.10",
#     "torch-scatter",
#     "torch-sparse",
# ]
#
# [tool.uv.sources]
# torch-scatter = { url = "https://data.pyg.org/whl/torch-2.4.0%2Bcu124/torch_scatter-2.1.2%2Bpt24cu124-cp312-cp312-linux_x86_64.whl" }
# torch-sparse = { url = "https://data.pyg.org/whl/torch-2.4.0%2Bcu124/torch_sparse-0.6.18%2Bpt24cu124-cp312-cp312-linux_x86_64.whl" }
# ///

from matbench_speed.benchmark import benchmark
from fairchem.core import OCPCalculator

calculators = {
    "eSEN-30M-MP": OCPCalculator(
        local_cache="../models",
        model_name="eSEN-30M-MP",
    ),
}
benchmark(calculators)
