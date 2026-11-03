# CaP-X

A ROCm-enabled container for [**CaP-X**](https://github.com/capgym/cap-x) - *Code-as-Policies eXtended*, a framework for benchmarking and improving coding agents for robot manipulation.

This Ryzer runs the **CaP-Gym evaluation path** on AMD Ryzen AI hardware: a coding-agent LLM generates Python that composes perception + control primitives to solve manipulation tasks in simulation. It supports two evaluation suites:

- **Native Robosuite environments** (Robosuite 1.5.x)
- **LIBERO-PRO environments** (built on Robosuite 1.4.x)

Everything GPU-bound runs on ROCm: the perception models (SAM3 or SAM2+OWLv2, Contact-GraspNet) on ROCm PyTorch, and MuJoCo rendering via EGL on the AMD GPU. Motion planning (PyRoKi IK) runs on CPU `jax`.

---

## Build the image

```bash
# Robosuite (default)
ryzers build capx
ryzers run             # runs image and oracle tests
```

If you plan to serve the LLM locally, build llama.cpp into the **same** image instead - see [local LLM instructions](#local-llm-llamacpp).

Expected tail:

```
================ [1/2] CaP-X / ROCm sign-of-life ================
GPU ok  : True
  device 0: Radeon 8060S Graphics
capx + pyroki import OK
robosuite 1.5.1 import OK
EGL render OK, frame shape (64, 64, 3)
================ [2/2] Oracle eval: franka_robosuite_pick_place_code_env ================
PyRoKi ready (warmed JAX JIT) after 19s
Success
ORACLE EVAL PASSED (reward 1.0)
================ CaP-X tests PASSED ================
```

<details>
<summary>LIBERO-PRO variant (experimental)</summary>

The two simulator families pin conflicting Robosuite versions, so pick one per image. Set `CAPX_SIM=libero` in [`config.yaml`](config.yaml), then build under a distinct name:

```bash
# LIBERO-PRO (experimental)
ryzers build --name capx-libero capx
ryzers run --name capx-libero
```
</details>

## Recommended Setup
Requirements:
- An OpenRouter API key
- Approved access to `facebook/sam3` and a Hugging Face read token

Set `OPENROUTER_API_KEY` and `HF_TOKEN` in [`config.yaml`](config.yaml):

```yaml
environment_variables:
- "OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxx"
- "HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx"
```

Then rebuild the image to apply the change:
```bash
ryzers build capx
ryzers run bash
```

Inside the bash start the server:
```bash
cd /ryzers/cap-x
echo "$OPENROUTER_API_KEY" > .openrouterkey
python3 capx/serving/openrouter_server.py --key-file .openrouterkey --port 8110 &
```

After the openrouter server is ready, you may run the benchmark
```bash
python3 capx/envs/launch.py \
    --config-path env_configs/cube_stack/franka_robosuite_cube_stack.yaml \
    --model "openrouter/google/gemini-3.1-pro-preview" \
    --total-trials 10 --num-workers 4
```

Model names take the `openrouter/` prefix, e.g. `openrouter/google/gemini-3.1-pro-preview`.

If you **do not have both of these keys**, follow the instructions below to get them or use one of the alternative methods.

<details>
<summary><b>Get an OpenRouter key</b></summary>

1. Sign up at <https://openrouter.ai>.
2. Create a key at <https://openrouter.ai/keys>.
3. Add credit; CaP-X evaluations are image-heavy.
</details>

<details>
<summary><b>Get SAM3 access and an HF token</b></summary>

1. Create an account at <https://huggingface.co>.
2. Request access at <https://huggingface.co/facebook/sam3> and accept the license. Approval is manual.
3. Once approved, create a **read** token at <https://huggingface.co/settings/tokens>.
</details>

---

## Other Running Methods

Note that the single-turn API configs here (the `*_sam2.yaml` pick/stack tasks) send **no images** to the LLM - OWLv2/SAM2/GraspNet do the perception and the model only composes code from named objects and numeric poses, so a strong text-only coder/reasoner works well. Only the visual-feedback (`*_multiturn_vf.yaml`, `use_visual_feedback: true`) and image-differencing (`*_vdm.yaml`) configs send rendered frames to the LLM and therefore require a multimodal model.

### Local LLM: llama.cpp

Build llama.cpp into the image, open a shell, and start a model. The recommended
default is **GPT-OSS-20B** (`ggml-org/gpt-oss-20b-GGUF`, MXFP4, ~12 GB), a strong
local reasoning model that we found worked well on cube stacking tasks.

```bash
ryzers build capx llamacpp
ryzers run bash

llama-server -hf ggml-org/gpt-oss-20b-GGUF \
  --host 127.0.0.1 --port 11434 \
  --n-gpu-layers 999 --ctx-size 8192 --parallel 1 \
  --jinja --flash-attn on \
  --reasoning-format auto --temp 1.0 --top-p 1.0 --top-k 0 \
  >/tmp/llama.log 2>&1 &
```

If this is the first time you're downloading the model it may take a while, you can monitor progress by tailing the log:

```
tail -f /tmp/llama.log
```

Wait for `listening on http://127.0.0.1:11434`, then stop `tail` with
<kbd>Ctrl</kbd>+<kbd>C</kbd>. The server continues running in the background.

[Lemonade](../../llm/lemonade-sdk/) and [Ollama](../../llm/ollama/) can also be composed into the image. Use their OpenAI-compatible chat-completions URL with `--server-url`.

### SAM2 + OWLv2: no HF token

Pre-download the local vision models before launching CaP-X:

```bash
hf download facebook/sam2.1-hiera-base-plus
hf download google/owlv2-large-patch14-ensemble
```

Then use the ungated cube-stack config:

```bash
cd /ryzers/cap-x
python3 capx/envs/launch.py \
  --config-path env_configs/cube_stack/franka_robosuite_cube_stack_sam2.yaml \
  --model gpt-oss-20b \
  --server-url http://127.0.0.1:11434/v1/chat/completions \
  --max-tokens 4096 --temperature 1.0 \
  --total-trials 1 --num-workers 1
```

When running with a local config the vision pipeline and all other services may take 5 minutes or more to start. This is an upfront startup cost. When we run multiple trials each trial should take ~1 minute to complete after initial startup.

The ready ungated visual configs are cube stack above and `env_configs/two_arm_lift/franka_robosuite_two_arm_lift.yaml`. Other tasks need task-specific API changes; replacing only `api_servers:` is not sufficient.

For planning-only evaluation, use a task's `*_privileged.yaml` config under `env_configs/`. These use ground-truth poses and do not start a segmentation server.

### Molmo pointing service

Nut assembly resolves every object by asking **Molmo** to point at it, so its `get_object_pose()` and `sample_grasp_pose()` return `(None, None)` unless a Molmo endpoint is up. Upstream ships no launcher for it (the `molmo` extra pins vLLM, which conflicts with `robosuite`), so this image adds [`launch_molmo_server.py`](launch_molmo_server.py).

Start the service with:

```bash
ryzers run bash
cd /ryzers/cap-x
python3 capx/serving/launch_molmo_server.py --port 8122 >/tmp/molmo.log 2>&1 &
tail -f /tmp/molmo.log   # wait for "Starting server", Ctrl-C to stop tailing
```

The client expects port 8122. `allenai/Molmo2-8B` is ~33 GB and needs ~18 GB VRAM; pass `--model-name allenai/Molmo2-4B` if that is tight.

Then run the task:

```bash
python3 capx/envs/launch.py \
  --config-path env_configs/nut_assembly/franka_robosuite_nut_assembly.yaml \
  --model "openrouter/google/gemini-3.1-pro-preview" \
  --total-trials 10 --num-workers 1
```

Nut assembly is the only task that requires Molmo. Every task's `*_reduced_api*` configs also offer `point_prompt_molmo` as one grounding tool among several, and fall back to OWLv2/SAM when no server is up.

---

## Run with different configurations

Run the actual benchmarks with `launch.py` from inside the `ryzers run bash` container, using the correct LLM endpoint.

**OpenRouter (Cloud) + (SAM2 + OWLv2)** (no HF token needed):
```bash
cd /ryzers/cap-x
python3 capx/envs/launch.py \
    --config-path env_configs/cube_stack/franka_robosuite_cube_stack_sam2.yaml \
    --model "openrouter/google/gemini-3.1-pro-preview" \
    --total-trials 10 --num-workers 4
```

**Local LLM + SAM3** (needs HF_TOKEN and approved access):
```bash
cd /ryzers/cap-x
python3 capx/envs/launch.py \
    --config-path env_configs/cube_stack/franka_robosuite_cube_stack.yaml \
    --model gpt-oss-20b \
    --server-url http://127.0.0.1:11434/v1/chat/completions \
    --max-tokens 4096 --temperature 1.0 \
    --total-trials 10 --num-workers 1
```

**Local LLM + (SAM2 + OWLv2)** (the zero-key path):
```bash
cd /ryzers/cap-x
python3 capx/envs/launch.py \
    --config-path env_configs/cube_stack/franka_robosuite_cube_stack_sam2.yaml \
    --model gpt-oss-20b \
    --server-url http://127.0.0.1:11434/v1/chat/completions \
    --max-tokens 4096 --temperature 1.0 \
    --total-trials 10 --num-workers 1
```

<details>
<summary><b>LIBERO-PRO example</b> (experimental)</summary>

Requires the libero image (`ryzers build --name capx-libero capx` with `CAPX_SIM=libero`), an LLM server reachable, **and SAM3** - see below.

```bash
# inside `ryzers run --name capx-libero bash`
cd /ryzers/cap-x
python3 capx/envs/launch.py \
    --config-path env_configs/libero/franka_libero_spatial_0.yaml \
    --model <your-model> --server-url <your-endpoint> \
    --total-trials 5 --num-workers 1
```

Suites are selected by `low_level.suite_name` / `task_id` in the config: `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`, `libero_90`.

**Perception: this config needs an HF token.** It uses `FrankaLiberoApi` (registered with `use_sam3=True`) and launches `launch_sam3_server` on 8114, so the gated `facebook/sam3` weights are required. The [`FrankaControlApiSam2` path](#sam2--owlv2-no-hf-token) does **not** apply - it's a Robosuite control API, not the LIBERO one.

`FrankaLiberoApi` does accept `use_sam3=False`, which routes through SAM2 point-prompting (note: point prompts, not OWLv2 boxes as in the Robosuite path). No ungated LIBERO config ships in the image, so you'd register the variant and write the YAML yourself, mirroring [`franka_robosuite_cube_stack_sam2.yaml`](franka_robosuite_cube_stack_sam2.yaml).
</details>

<br>

`--total-trials` is how many episodes to average over - more means a less noisy success rate and a proportionally longer run. `--num-workers` is how many run in parallel, and must not exceed your server's concurrency: 1 for a default `llama-server`, higher for OpenRouter. Beyond that they just queue. Both override the `trials` / `num_workers` values in the config YAML.

---

## References

- CaP-X: <https://github.com/capgym/cap-x>
- LIBERO-PRO: <https://github.com/uynitsuj/LIBERO-PRO>
- Robosuite: <https://github.com/ARISE-Initiative/robosuite>
- llama.cpp: <https://github.com/ggml-org/llama.cpp>
