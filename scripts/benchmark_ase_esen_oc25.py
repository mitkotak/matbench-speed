# /// script
# dependencies = [
#     "fairchem-core",
#     "matbench_speed",
# ]
#
# [tool.uv.sources]
# matbench_speed = { path = "../." }
# ///

from matbench_speed.benchmark import benchmark
from fairchem.core import pretrained_mlip, FAIRChemCalculator

calculators = {
    "eSEN-6M-OC25": FAIRChemCalculator(
        pretrained_mlip.load_predict_unit("./models/esen_sm_conserve.pt", device="cuda"), task_name="oc25")
}
benchmark(calculators)
