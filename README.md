# Matbench-Speed

Benchmarking MLIPs using their ASECalculator

- SevenNet, GRACE, Nequix scripts out of the box.
- For MACE without kernels you might need to do [this hack](https://github.com/ACEsuit/mace/issues/1208#issuecomment-3348723384) in your local installation (typically in your conda/venv path)
- For eSEN the checkpoint needs to be downloaded from [hugging face](https://huggingface.co/facebook/OMAT24).
- For NequIP we are using the OMat24 model instead of MPTrj since it contains the kernels, and we confirmed that the hyperparams are the same for L model. XL model changes the lmax for intermediate irreps from lmax 3 to 4.

# Benchmarks (Updated 01/05/2026)

## Matbench compliant

### A100
![](./figures/inference_fig_compliant_Si_5.43_NVIDIA%20A100-SXM4-80GB_float32.png)

###  H100
![](./figures/inference_fig_compliant_Si_5.43_NVIDIA%20H100%2080GB%20HBM3_float32.png)

## Matbench non-compliant

### A100
![](./figures/inference_fig_non-compliant_Si_5.43_NVIDIA%20A100-SXM4-80GB_float32.png)

### H100
![](./figures/inference_fig_non-compliant_Si_5.43_NVIDIA%20H100%2080GB%20HBM3_float32.png)
