# /// script
# dependencies = [
#     "fairchem-core",
# ]
#
# ///

from matbench_speed.benchmark import benchmark
from fairchem.core import pretrained_mlip, FAIRChemCalculator

calculators = {
    "eSEN-6M-OC25": FAIRChemCalculator(
        pretrained_mlip.get_predict_unit("esen-sm-conserving-all-oc25", device="cuda"), task_name="oc25")
}
benchmark(calculators)
