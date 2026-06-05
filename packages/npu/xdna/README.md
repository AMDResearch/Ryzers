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

## Host firmware

The `amdxdna` kernel driver loads the NPU firmware (`*.sbin`) from `/lib/firmware/amdnpu/` on the **host**. If it is missing, the device never enumerates and you will not see `/dev/accel/accel0` (so `xrt-smi examine` finds no NPU). On Ubuntu 24.04+ this firmware ships in the `linux-firmware` package.

Install it on the host using one of the following:

```bash
# Option A: from the distro package (simplest on Ubuntu 24.04+)
sudo apt-get update && sudo apt-get install -y linux-firmware

# Option B: extract just the amdnpu blobs with the helper script (no full package install)
sudo ./extract_npu_firmware.sh /lib/firmware/amdnpu

# Option C: copy the firmware out of the built xdna image
docker run --rm -v "$PWD:/host" xdna:latest \
    bash -c "cp -a /lib/firmware/amdnpu /host/amdnpu"
sudo cp -a ./amdnpu /lib/firmware/
```

After installing the firmware, reload the driver (or reboot) so it picks up the blobs:

```bash
sudo modprobe -r amdxdna && sudo modprobe amdxdna
# then confirm the device shows up
xrt-smi examine
```

The same `extract_npu_firmware.sh` runs automatically inside the container build, so the firmware is present in the image; this section is only about the host kernel driver. The driver's own firmware fetch (`download_npufws`, which pulls `.sbin` blobs from `repo.radeon.com`) is stubbed out during the build, so the image's firmware comes **exclusively** from the `linux-firmware` deb.

---

For further details, refer to the official [xdna-driver](https://github.com/amd/xdna-driver) repository.
