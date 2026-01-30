# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pip-system-certs",
#     "ase",
#     "mace-torch",
#     "torch==2.9.1",
#     "cuequivariance-torch",
#     "cuequivariance-ops-torch-cu12",
#     "matbench_speed",
# ]
#
# [tool.uv.sources]
# matbench_speed = { path = "../." }
# ///

#from matbench_speed.benchmark import benchmark
from matbench_speed.benchmark_md_ase import benchmark
from mace.calculators.foundations_models import mace_mp

calculators = {
    #"MACE-MP-0-no-kernels": mace_mp(
    #    model="medium-0b3",
    #    default_dtype="float32",
    #    device="cuda",
    #    compile_mode="default",
    #    fullgraph=False,
    #    enable_cueq=False,
    #),
    "MACE-MP-0": mace_mp(
        mode="medium-0b3",
        default_dtype="float32",
        device="cuda",
        compile_mode="default",
        fullgraph=False,
        enable_cueq=True,
    )
}
benchmark(calculators, atom_name="Si", lattice_constant=5.43)
