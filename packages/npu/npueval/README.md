# NPUEval

[![NPU](https://img.shields.io/badge/ryzenai-npu-blue)](#)

---

NPUEval is a benchmark for evaluating LLMs on their ability to write NPU kernel code on Ryzen AI hardware platforms.

## Usage

The docker image has a dependency on the **XDNA** and **IRON** ryzers and must be composed using the following:

```
ryzers build xdna iron npueval
ryzers run
```

The default test will run a single benchmark kernel canonical solution on the NPU. If successful your output should look like this:

```
Kernel: abs_int8_wrapper
Using canonical solution...
results/evaluations/abs_int8_wrapper.mlir generated successfully
abs_int8_wrapper.xclbin, abs_int8_wrapper.bin built
Trace written to results/evaluations/abs_int8_wrapper_trace.json
Result: Pass
Passed: 1/1
```

Pass means your environment and NPU are set up correctly. Now you can explore the shipped example notebooks that will walk you through how to interact with the dataset and generate your own solutions with your preferred LLMs.

## Start jupyter server

You can start a jupyter server like you would an interactive bash session:

```bash
ryzers run "python3 -m jupyterlab npueval/notebooks --ip=0.0.0.0 --port=8888 --no-browser --allow-root"
```

This will automatically open the directory containing the NPUEval notebooks.

---

For further details refer to the [AMDResearch NPUEval repository](https://github.com/amdresearch/npueval).