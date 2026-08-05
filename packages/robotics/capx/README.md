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
ryzers run             # runs sign-of-life test
```

If you plan to serve the LLM locally, build llama.cpp into the **same** image instead - see [step 2, option B](#option-b---local-llamacpp-in-the-image).

Expected tail with no LLM configured:

```
================ [1/3] CaP-X / ROCm sign-of-life ================
GPU ok  : True
  device 0: Radeon 8060S Graphics
capx + pyroki import OK
robosuite 1.5.1 import OK
EGL render OK, frame shape (64, 64, 3)
================ [2/3] Oracle eval (no LLM): franka_robosuite_pick_place_code_env ================
PyRoKi ready (warmed JAX JIT) after 19s
Success
ORACLE EVAL PASSED (reward 1.0)
================ [3/3] LLM eval (optional) ================
SKIPPED - set CAPX_LLM_SERVER_URL ...
================ CaP-X tests PASSED ================
```

<details>
<summary><b>Optional: enabling the stage-3 LLM eval</b></summary>

Stage 3 runs `env_configs/cube_stack/franka_robosuite_cube_stack.yaml`, so it needs **SAM3 weights** on top of an LLM endpoint, see [step 3, option A](#option-a---sam3-upstream-default-gated) for obtaining a `HF_TOKEN`. Without it the stage still reports complete, but every trial fails with `No sam3 detections`.

Set these to turn the stage on:

| Variable | Meaning | Local example | Cloud example |
|---|---|---|---|
| `CAPX_LLM_SERVER_URL` | OpenAI-compatible `/chat/completions` URL | `http://127.0.0.1:11434/v1/chat/completions` | `https://openrouter.ai/api/v1/chat/completions` |
| `CAPX_LLM_MODEL` | model name to request | `gemma-4-12b-it` | `google/gemini-3.1-pro-preview` |
| `CAPX_LLM_TRIALS` | number of trials (default 2) | `2` | `2` |
| `CAPX_LLM_API_KEY` | bearer token; leave unset for local | - | `sk-or-v1-...` |

Either put them in [`config.yaml`](config.yaml) and rebuild, or export them inside a `ryzers run bash` shell for a one-off:

```bash
export CAPX_LLM_SERVER_URL=http://127.0.0.1:11434/v1/chat/completions
export CAPX_LLM_MODEL=gemma-4-12b-it
export CAPX_LLM_TRIALS=2

/ryzers/test_capx.sh          # now runs all three stages
```

These only affect `test.sh`. For a real benchmark, [step 4](#4-run) passes the endpoint to `launch.py` directly and ignores them.
</details>
<br>


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
- Have a openrouter api key
- Have a SAM3 api key and installed

Set `OPENROUTER_API_KEY` and the `HF_TOKEN` in [`config.yaml`](config.yaml):

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

If you **do not have both of these keys**, continue with the instructions to pursue different methods for running the benchmark.

---

## Other Running Methods

### 1. Pick an LLM provider

CaP-X's LLM client (`capx/llm/client.py`) posts a standard OpenAI `chat/completions` payload to whatever `--server-url` you give it, for any model name not in its built-in OpenRouter/GPT/Claude lists. So **any** OpenAI-compatible endpoint works.

> **Vision is required.** CaP-X sends `image_url` content. A text-only model will not work for the perception-grounded configs - it must be multimodal (and, for GGUF, have an `mmproj`).

### Option A - OpenRouter (Cloud)

<details>
<summary><b>No OpenRouter key?</b> How to get one</summary>

1. Sign up at <https://openrouter.ai>.
2. Create a key at <https://openrouter.ai/keys>.
3. Add credit - CaP-X evals are image-heavy and a 10-trial run on a frontier model has costs.

Free-tier models exist (`:free` suffix) but are heavily rate-limited and mostly text-only, so they won't work here.

If you'd rather not pay, skip to option B, which will run entirely on your GPU.
</details>

<br>

After getting your openrouter key, follow the instructions for setting the `OPENROUTER_API_KEY` and starting the server in the [recommended setup](#recommended-setup).

### Option B - Local/on-prem OpenAI-compatible LLM (llama.cpp)

Build llama.cpp **into** the CaP-X image so both live in one container:

```bash
ryzers build llamacpp capx
```

This compiles `llama-server` with HIP into `/ryzers/llamacpp/build/bin/`.

Then start a shell and serve a vision model:

```bash
ryzers run bash
```

```bash
export PATH=/ryzers/llamacpp/build/bin:$PATH

# pulls model if not already on cache
llama-server -hf ggml-org/gemma-4-12b-it-GGUF:Q4_0 \
             --host 127.0.0.1 --port 11434 \
             --n-gpu-layers 999 --jinja \
             > /tmp/llama.log 2>&1 &
```

> <details>
> <summary><b>You may also use other local model server providers</b></summary>
>
> Any OpenAI-compatible server with a vision model will do. [Lemonade](../../llm/lemonade-sdk/) (`ryzers build lemonade-sdk capx`) and [Ollama](../../llm/ollama/) (`ryzers build ollama capx`) chain into the image the same way; vLLM-ROCm has no Ryzers package, so it needs to be installed on the host. Either way `127.0.0.1` reaches the server, since Ryzers uses host networking.
>
> Watch the URLs. llama.cpp, Ollama and vLLM all serve `/v1/chat/completions` (Ollama on 11434, which collides with the port used above; vLLM on 8000). Lemonade is the exception - it serves `/api/v1/chat/completions` on 13305.
> </details>

---

## 2. Pick a perception weight backend

Both options work with either provider.

### Option A - SAM3 (upstream default, gated)

Weights (`facebook/sam3`) are **gated** on HuggingFace, so this path needs a token.

<details>
<summary><b>No HF token?</b> How to get one</summary>

1. Create an account at <https://huggingface.co>.
2. Request access at <https://huggingface.co/facebook/sam3> and accept the license. **Approval is manual and not instant**, which means it can take a few hours.
3. Once approved, create a **read** token at <https://huggingface.co/settings/tokens>.

Without approval the token alone is not enough, you'll still get an error:

```
GatedRepoError: 401 Client Error.
Cannot access gated repo for url https://huggingface.co/facebook/sam3/resolve/main/config.json.
```
</details>

<br>

After getting your SAM3 key, you will need to install SAM3:

```bash
# inside ryzers run bash
cd /ryzers/cap-x
hf download facebook/sam3
```

Then follow the instructions for setting the `HF_TOKEN` in the [recommended setup](#recommended-setup).

### Option B - SAM2 + OWLv2 (ungated, no token, no file edits)

Swaps segmentation to **OWLv2** (open-vocab detection) + **SAM2** (segment the detected box). Both are ungated and download without credentials.

First you will need to download the models:

```bash
# inside ryzers run bash
cd /ryzers/cap-x
hf download facebook/sam2.1-hiera-large
hf download google/owlv2-large-patch14-ensemble
```

> Those are the defaults in `capx/serving/launch_sam2_server.py` and `capx/serving/launch_owlvit_server.py`. If you change the model there, download that repo instead.

Then pass this to the `--config-path` when running the `launch.py`:

```bash
--config-path env_configs/cube_stack/franka_robosuite_cube_stack_sam2.yaml
```

<details>
<summary>What that config does (and how to adapt it to other tasks)</summary>

The control APIs support ungated perception via a `use_sam3=False` path and ships [`franka_robosuite_cube_stack_sam2.yaml`](franka_robosuite_cube_stack_sam2.yaml), which points `apis:` at that variant and swaps the SAM3 `api_server` for the SAM2 + OWLv2 servers.

To use other tasks with ungated configs for the perception benchmark, you will need to change the lines below (plus `output_dir`). Leave `api_servers:` alone. Values come from each task's own YAML at `env_configs/<task>/franka_robosuite_<task>.yaml` - the one exception is `two_arm_handover/two_arm_handover.yaml`, which has no `franka_robosuite_` prefix:

| Task | `env._target_` (prefix `capx.envs.tasks.franka.`) | `cfg.low_level` | `cfg.apis` |
|---|---|---|---|
| cube_lifting | `franka_lift.FrankaLiftCodeEnv` | `franka_robosuite_cube_lift_low_level` | keep `FrankaControlApiSam2` |
| cube_restack | `franka_cube_restack.FrankaRestackCodeEnv` | `franka_robosuite_cubes_restack_low_level` | keep `FrankaControlApiSam2` |
| nut_assembly | `franka_nut_assembly.FrankaNutAssemblyCodeEnv` | `franka_robosuite_nut_assembly_low_level_visual` | `FrankaControlNutAssemblyVisualApi`\* |
| spill_wipe | `franka_spill_wipe.FrankaSpillWipeCodeEnv` | `franka_robosuite_spill_wipe_low_level` | `FrankaControlSpillWipeApi`\* |
| two_arm_handover | `two_arm_handover.TwoArmHandoverCodeEnv` | `two_arm_handover_robosuite` | `FrankaHandoverApi`\* |

The only exception is the `two_arm_lift` which has a ready config at `env_configs/two_arm_lift/franka_robosuite_two_arm_lift.yaml`. Also, there are 6 more tasks in that folder, all OWLv2+SAM2.

Another way to run would be to evaluate the planning only, which skips the perception and evaluates the model's reassoning capabilities. For these, the config files are already created, just alter the `--config-path` to:

```
# privileged - ground-truth poses, no segmentation server at all
env_configs/cube_lifting/franka_robosuite_cube_lifting_privileged.yaml
env_configs/cube_restack/franka_robosuite_cube_restack_privileged.yaml
env_configs/cube_stack/franka_robosuite_cube_stack_privileged.yaml
env_configs/nut_assembly/franka_robosuite_nut_assembly_privileged.yaml
env_configs/spill_wipe/franka_robosuite_spill_wipe_privileged.yaml
env_configs/two_arm_handover/two_arm_handover_privileged.yaml
env_configs/two_arm_lift/franka_robosuite_two_arm_lift_privileged.yaml
```
</details>

<br>

> The SAM2 server is hard-wired to port **8113** and OWLv2 to **8117** (`capx/integrations/vision/{sam2,owlvit}.py`) - match those in the config.

Trade-off: SAM3 is a single open-vocab segmentation model, while SAM2 + OWLv2 follows a detect-then-segment approach. Grounding is somewhat less accurate, so expect slightly lower rewards than published SAM3 numbers.

---

## 3. Run

`test.sh` stage 3 is a smoke test. For running the actual benchmark, use the `launch.py` yourself from inside the container of `ryzers run bash`. Make sure you supply the LLM with the correct endpoint.

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
    --model gemma-4-12b-it \
    --server-url http://127.0.0.1:11434/v1/chat/completions \
    --total-trials 10 --num-workers 1
```

**Local LLM + (SAM2 + OWLv2)** (the zero-key path):
```bash
cd /ryzers/cap-x
python3 capx/envs/launch.py \
    --config-path env_configs/cube_stack/franka_robosuite_cube_stack_sam2.yaml \
    --model gemma-4-12b-it \
    --server-url http://127.0.0.1:11434/v1/chat/completions \
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

**Perception: this config needs an HF token.** It uses `FrankaLiberoApi` (registered with `use_sam3=True`) and launches `launch_sam3_server` on 8114, so the gated `facebook/sam3` weights are required. The `FrankaControlApiSam2` shortcut from [step 3](#option-b---sam2--owlv2-ungated-no-token-no-file-edits) does **not** apply - it's a Robosuite control API, not the LIBERO one.

`FrankaLiberoApi` does accept `use_sam3=False`, which routes through SAM2 point-prompting (note: point prompts, not OWLv2 boxes as in the Robosuite path). No ungated LIBERO config ships in the image, so you'd register the variant and write the YAML yourself, mirroring [`franka_robosuite_cube_stack_sam2.yaml`](franka_robosuite_cube_stack_sam2.yaml).
</details>

<br>

`--total-trials` is how many episodes to average over - more means a less noisy success rate and a proportionally longer run. `--num-workers` is how many run in parallel, and must not exceed your server's concurrency: 1 for a default `llama-server`, higher for OpenRouter. Beyond that they just queue. Both override the `trials` / `num_workers` values in the config YAML.

---

## References

- CaP-X: <https://github.com/capgym/cap-x> · [paper](https://arxiv.org/abs/2603.22435) · [project page](https://capgym.github.io/)
- LIBERO-PRO: <https://github.com/uynitsuj/LIBERO-PRO>
- Robosuite: <https://github.com/ARISE-Initiative/robosuite>
- llama.cpp: <https://github.com/ggml-org/llama.cpp>
