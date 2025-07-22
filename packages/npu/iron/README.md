# IRON

[![NPU](https://img.shields.io/badge/ryzenai-npu-blue)](#)

---

IRON is an open-source research project from AMD that enables developers to program AI Engine (AIE) devices like the NPU with a close-to-metal API.

## Building and Running the Docker Container

The docker image **must be built on top of the XDNA docker** which contains the installed drivers and runtime to interact with the NPU. Make sure to compose the ryzers together as follows:

```bash
ryzers build xdna iron
ryzers run
```

The default test will build and run one of the example kernels on the NPU. If running the ryzer results in messages like the following, your environment is setup correctly!

```
python3 /ryzers/mlir-aie/programming_examples/basic/vector_scalar_mul/test.py -x build/final_trace_in1_size.xclbin -i build/insts_in1_size.bin -k MLIR_AIE -t 8192 -i1s 8192  -i2s 4  -os 8192 
Running...

npu_time:  958680

PASS!
```

Run an interactive session

```bash
ryzers run bash
```

And explore more examples by navigating to the programming_examples directory.

```bash
cd /ryzers/mlir-aie/programming_examples
```

---

For further details and learning material, refer to the [MLIR-AIE repository](https://github.com/xilinx/mlir-aie).