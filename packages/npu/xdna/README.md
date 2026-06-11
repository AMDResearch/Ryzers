# AMD XDNA

[![NPU](https://img.shields.io/badge/ryzenai-npu-blue)](#)

---

## Building and Running the Docker Container

**Note:** To run the container you will need to have the XDNA driver installed on your host system as well.

You can install the driver using the [install_xdna.sh](install_xdna.sh) script. Inspect the contents of the script, by setting the `USE_RYZER` option you can either build it nativelly on your host machine or utilize this package to build it in a docker (the default option). If the host already has an older driver and you are bumping `DRIVER_VERSION`, rerun the script with `FORCE_REINSTALL=true` to rebuild and reinstall.

If you used the `install_xdna.sh` script with the `USE_RYZER` flag, you're ready to run the container with a one-liner.

```bash
ryzers run --name xdna
```

If you installed the driver locally and need to rebuild the docker, you can do the standard ryzers calls.

```bash
ryzers build xdna
ryzers run
```

By default the docker will run a driver validation script. You should see the following output:

```
WARNING: User doesn't have admin permissions to set performance mode. Running validate in default mode
Validate Device           : [0000:c6:00.1]
    Platform              : NPU Strix Halo
    Power Mode            : default
-------------------------------------------------------------------------------
Test 1 [0000:c6:00.1]     : gemm
    Details               : TOPS: 51.0
    Test Status           : [PASSED]
-------------------------------------------------------------------------------
Test 2 [0000:c6:00.1]     : latency
    Details               : Average latency: 48.0 us
    Test Status           : [PASSED]
-------------------------------------------------------------------------------
Test 3 [0000:c6:00.1]     : throughput
    Details               : Average throughput: 72493.0 op/s
    Test Status           : [PASSED]
-------------------------------------------------------------------------------
Validation completed
```

---

For further details, refer to the official [xdna-driver](https://github.com/amd/xdna-driver) repository.
