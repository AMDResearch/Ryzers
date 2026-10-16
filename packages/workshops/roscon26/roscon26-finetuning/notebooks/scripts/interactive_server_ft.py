# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Interactive MolmoAct2 x LIBERO demo for FINE-TUNED checkpoints - SYNCHRONOUS rollout.

Synchronous (chunk-replay) interactive server. Unlike a real-time RTC engine (plan-ahead +
blend), it runs the exact same closed-loop receding-horizon control loop as the Step-5 LIBERO
eval: `lerobot_eval.rollout` plans an action chunk, executes it, then re-plans - the world is
effectively frozen during each model forward. This is the deterministic, best-behaved demo (no
blend artifacts), at the cost of hiding planner latency (the arm pauses briefly to think between
chunks). Use this when policy precision matters more than continuous motion (e.g. real-time
blending was dropping grasped objects).

Policy loading matches the RT server and the Step-5 eval: a LeRobot-format fine-tuned checkpoint
via `--policy.path` (processor + normalization stats restored from the checkpoint, no norm_tag),
on the trainable allenai/lerobot@molmoact2-policy stack in /opt/train-venv. If POLICY_PATH is
unset it falls back to a released HF checkpoint via `--policy.checkpoint_path` + `--policy.norm_tag`.

lerobot 0.5.2 note: `rollout` reads the task itself via `env.call("task_description")` (the old
`ev.add_envs_task` injection hook was removed), so we intercept `env.call` to feed the user's live
instruction into the loop.

Command-driven UX identical to the RT demo: idle on a scene until you send an instruction; then it
resets the env + policy and runs that one task, streaming the camera (single-frame /frame polling,
proxy-friendly) and saving a debug video. Randomize swaps to a new random scene.

Env: POLICY_PATH (LeRobot fine-tuned ckpt dir/Hub repo) OR CKPT (released HF repo),
NORM_TAG (libero, only for the CKPT fallback), SUITE, TASK_ID, SEED, PORT (8080),
VIEW_RES (720), VIDEO_RES (600), NUM_STEPS, OUT_DIR (/outputs).
Open http://localhost:PORT, or embed via jupyter-server-proxy ({PREFIX}/proxy/PORT/).
"""
import io
import json
import os
import random
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SUITE = os.environ.get("SUITE", "libero_object")
TASK_ID = int(os.environ.get("TASK_ID") or "3")
SEED = int(os.environ.get("SEED") or "1000")
POLICY_PATH = os.environ.get("POLICY_PATH", "").strip()
CKPT = os.environ.get("CKPT", "allenai/MolmoAct2-LIBERO").strip()  # fallback (non-Think)
NORM_TAG = os.environ.get("NORM_TAG", "libero")
PORT = int(os.environ.get("PORT") or "8080")
VIEW_RES = int(os.environ.get("VIEW_RES") or "720")    # live viewport (proxy-friendly single frames)
VIDEO_RES = int(os.environ.get("VIDEO_RES") or "600")  # saved debug video (kept small)
OUT_DIR = os.environ.get("OUT_DIR", "/outputs")
SUITES = ["libero_object", "libero_goal", "libero_spatial", "libero_10"]

# ---- shared state ------------------------------------------------------------
STATE = {
    "mode": "loading",        # loading | idle | running | error
    "instruction": "",        # the instruction currently being executed
    "scene_task": "",         # the scene's native LIBERO instruction
    "suite": SUITE, "task_id": TASK_ID,
    "objects": [],            # object names visible in the scene
    "step": 0, "infer_ms": 0.0, "success": False,
    "status": "starting", "frame": None, "video_url": "",
}
LOCK = threading.Lock()
PENDING = {"action": None, "instruction": ""}
EVENT = threading.Event()
STOP = {"flag": False}


class StopRollout(Exception):
    pass


def _font(size):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def _encode_jpeg(rgb, quality=85):
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb)).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _set_frame(rgb):
    with LOCK:
        STATE["frame"] = _encode_jpeg(rgb)


def _banner_frame(rgb, text, size):
    """Downscale to `size` and add a top banner with the executed command."""
    img = Image.fromarray(np.ascontiguousarray(rgb)).resize((size, size), Image.BILINEAR)
    bh = max(40, size // 12)
    bh += bh % 2
    canvas = Image.new("RGB", (size, size + bh), (15, 15, 18))
    canvas.paste(img, (0, bh))
    d = ImageDraw.Draw(canvas)
    f = _font(max(14, size // 28))
    msg = text if len(text) <= 70 else text[:67] + "..."
    d.text((10, bh // 2), msg, fill=(240, 240, 240), font=f, anchor="lm")
    return np.asarray(canvas)


class Scene:
    """Holds a single (suite, task_id) vec env + its metadata. Built via make_env so
    obs_type / processors match the lerobot eval path exactly."""

    def __init__(self, suite, task_id, env, scene_task, objects):
        self.suite, self.task_id = suite, task_id
        self.env, self.scene_task, self.objects = env, scene_task, objects

    def hi_res(self, size):
        try:
            le = self.env.envs[0]
            img = le._env.sim.render(width=size, height=size, camera_name="agentview")
            return np.asarray(img)[::-1, ::-1]
        except Exception:
            return np.asarray(self.env.call("render")[0])

    def idle_frame(self, size):
        try:
            self.env.reset(seed=SEED)
        except Exception:
            pass
        return self.hi_res(size)


# Runtime overrides applied to the loaded checkpoint config when using --policy.path.
# (lerobot's path-field filter strips inline --policy.* overrides, so we set these on the
# config object after parsing instead.)
_PATH_OVERRIDES = {
    "device": "cuda",
    "inference_action_mode": "continuous",
    "model_dtype": "bfloat16",
    "use_amp": True,
    "enable_inference_cuda_graph": False,
}


def _fallback_policy_cli():
    """Policy argv for the FALLBACK case: a released HF checkpoint via --policy.checkpoint_path
    (draccus parses these natively). Used only when POLICY_PATH is unset."""
    args = [
        "--policy.type=molmoact2",
        f"--policy.checkpoint_path={CKPT}",
        f"--policy.norm_tag={NORM_TAG}",
        "--policy.inference_action_mode=continuous",
        "--policy.model_dtype=bfloat16",
        "--policy.use_amp=true",
        "--policy.enable_inference_cuda_graph=false",
        "--policy.device=cuda",
    ]
    if os.environ.get("NUM_STEPS"):
        args.append(f"--policy.num_inference_steps={os.environ['NUM_STEPS']}")
    return args


def rollout_thread():
    import torch
    from fast_to_device import install
    install()
    import draccus
    from lerobot.configs import parser as lrparser
    from lerobot.configs.eval import EvalPipelineConfig
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    import lerobot.scripts.lerobot_eval as ev

    preprocess_observation = ev.preprocess_observation

    src = POLICY_PATH if POLICY_PATH else CKPT
    common = [
        "--env.type=libero",
        "--env.camera_name_mapping={\"agentview_image\":\"image\",\"robot0_eye_in_hand_image\":\"wrist_image\"}",
        "--eval.batch_size=1", "--eval.n_episodes=1",
        f"--seed={SEED}", "--output_dir=/tmp/interactive_ft_eval",
    ]

    def parse_eval(extra):
        """Build an EvalPipelineConfig. For a fine-tuned LeRobot checkpoint we register its
        path in lerobot's path-field registry (what `@parser.wrap()` does for --policy.path)
        so draccus loads the pretrained policy config, then apply runtime overrides on the
        config object (the path-field filter would otherwise drop inline --policy.* flags).
        For the released-HF fallback we pass --policy.type/--policy.checkpoint_path natively."""
        if POLICY_PATH:
            lrparser._config_path_args["policy"] = POLICY_PATH  # noqa: SLF001
            cfg = draccus.parse(EvalPipelineConfig, args=list(common) + list(extra))
            for k, v in _PATH_OVERRIDES.items():
                if hasattr(cfg.policy, k):
                    setattr(cfg.policy, k, v)
            if os.environ.get("NUM_STEPS") and hasattr(cfg.policy, "num_inference_steps"):
                cfg.policy.num_inference_steps = int(os.environ["NUM_STEPS"])
        else:
            cfg = draccus.parse(EvalPipelineConfig, args=_fallback_policy_cli() + list(common) + list(extra))
        return cfg

    with LOCK:
        STATE["status"] = f"loading policy from {src} (first run JIT-compiles kernels) ..."
    base_cfg = parse_eval([f"--env.task={SUITE}", f"--env.task_ids=[{TASK_ID}]"])
    policy = make_policy(cfg=base_cfg.policy, env_cfg=base_cfg.env, rename_map=base_cfg.rename_map)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=base_cfg.policy, pretrained_path=base_cfg.policy.pretrained_path,
        preprocessor_overrides={
            "device_processor": {"device": str(policy.config.device)},
            "rename_observations_processor": {"rename_map": base_cfg.rename_map},
        },
    )
    env_pre, env_post = make_env_pre_post_processors(env_cfg=base_cfg.env, policy_cfg=base_cfg.policy)

    def build_scene(suite, task_id):
        cfg = parse_eval([f"--env.task={suite}", f"--env.task_ids=[{task_id}]"])
        envs = make_env(cfg.env, n_envs=1, use_async_envs=False, trust_remote_code=cfg.trust_remote_code)
        env = envs[suite][task_id]
        # lerobot 0.5.2 rollout pulls the task via env.call("task_description"); intercept it so
        # we can inject the user's live instruction (the old ev.add_envs_task hook is gone). When
        # holder["instr"] is None (idle scene) we defer to the env's native task description.
        _true_call = env.call
        holder = {"instr": None}

        def _call(name, *a, **k):
            if holder["instr"] is not None and name in ("task_description", "task"):
                return [holder["instr"]] * env.num_envs
            return _true_call(name, *a, **k)

        env.call = _call
        env._instr_holder = holder
        env.reset(seed=SEED)
        le = env.envs[0]
        scene_task = getattr(le, "task_description", "") or ""
        try:
            objs = [getattr(o, "name", str(o)).replace("_1", "").replace("_", " ")
                    for o in le._env.env.objects]
        except Exception:
            objs = list(getattr(le._env, "obj_of_interest", []))
        return Scene(suite, task_id, env, scene_task, objs)

    def show_idle(sc):
        sc.env._instr_holder["instr"] = None  # idle scene uses its native task description
        with LOCK:
            STATE.update(mode="idle", status="idle - send an instruction", step=0,
                         success=False, suite=sc.suite, task_id=sc.task_id,
                         scene_task=sc.scene_task, objects=sc.objects,
                         instruction="", video_url="")
        STATE.pop("_video_path", None)
        _set_frame(sc.idle_frame(VIEW_RES))

    os.makedirs(os.path.join(OUT_DIR, "interactive_ft"), exist_ok=True)
    scene = build_scene(SUITE, TASK_ID)

    # Warm up flash/JIT kernels NOW (the first model forward compiles them and can take ~20s);
    # otherwise that one-time stall would freeze the first command with no feedback.
    with LOCK:
        STATE["status"] = "warming up GPU kernels (one-time JIT) ..."
    try:
        wobs, _ = scene.env.reset(seed=SEED)
        wproc = preprocess_observation(wobs)
        wproc["task"] = [scene.scene_task or "pick up the object" for _ in range(scene.env.num_envs)]
        wproc = env_pre(wproc)
        wproc = preprocessor(wproc)
        _t = time.perf_counter()
        with torch.inference_mode():
            policy.select_action(wproc)
        policy.reset()
        print(f"[warmup] done in {time.perf_counter() - _t:.1f}s", flush=True)
    except Exception as e:  # noqa: BLE001
        import traceback; traceback.print_exc()
        print("warmup failed:", e, flush=True)

    show_idle(scene)

    def run_command(sc, instruction):
        sc.env._instr_holder["instr"] = instruction  # rollout will read this via env.call
        with LOCK:
            STATE.update(mode="running", instruction=instruction, step=0, success=False,
                         status=f"running: {instruction}", video_url="")
        STOP["flag"] = False
        frames = []

        def render_cb(vec_env):
            if STOP["flag"]:
                raise StopRollout
            rgb = sc.hi_res(VIEW_RES)
            _set_frame(rgb)
            frames.append(_banner_frame(rgb, instruction, VIDEO_RES))
            v = getattr(policy, "_last_model_inference_s", 0.0) * 1000.0
            with LOCK:
                STATE["step"] += 1
                if v > 0:
                    STATE["infer_ms"] = round(v, 0)

        success = False
        try:
            with torch.no_grad():
                out = ev.rollout(sc.env, policy, env_preprocessor=env_pre, env_postprocessor=env_post,
                                 preprocessor=preprocessor, postprocessor=postprocessor,
                                 seeds=[SEED], render_callback=render_cb)
            try:
                success = bool(np.asarray(out["success"]).any())
            except Exception:
                success = False
        except StopRollout:
            with LOCK:
                STATE["status"] = "stopped"

        url = ""
        if frames:
            ts = datetime.now().strftime("%H%M%S")
            name = f"interactive_ft/{ts}_{sc.suite}_{sc.task_id}_{'ok' if success else 'run'}.mp4"
            path = os.path.join(OUT_DIR, name)
            try:
                import imageio
                with imageio.get_writer(path, fps=20, codec="libx264", quality=8,
                                        macro_block_size=1, output_params=["-pix_fmt", "yuv420p"]) as w:
                    for fr in frames:
                        w.append_data(fr)
                url = "video?ts=" + ts
                with LOCK:
                    STATE["_video_path"] = path
            except Exception as e:  # noqa: BLE001
                print("video save failed:", e, flush=True)

        with LOCK:
            STATE.update(mode="idle", success=success, video_url=url,
                         status=("success" if success else ("stopped" if STOP["flag"] else "done")))
        show_idle(sc)

    while True:
        EVENT.wait()
        EVENT.clear()
        with LOCK:
            action, instruction = PENDING["action"], PENDING["instruction"]
            PENDING["action"] = None
        if action == "randomize":
            STOP["flag"] = True
            suite = random.choice(SUITES)
            tid = random.randint(0, 9)
            with LOCK:
                STATE["status"] = f"loading scene {suite}/{tid} ..."
            try:
                new_scene = build_scene(suite, tid)
            except Exception as e:  # noqa: BLE001
                with LOCK:
                    STATE["status"] = f"scene build failed: {e}"
                continue
            try:
                scene.env.close()
            except Exception:
                pass
            scene = new_scene
            show_idle(scene)
        elif action == "run":
            run_command(scene, instruction)


# ---------------------------------------------------------------------------
PAGE = b"""<!doctype html><html><head><meta charset=utf-8>
<title>MolmoAct2 x LIBERO (fine-tuned, synchronous)</title>
<style>
 body{background:#0f1012;color:#e8e8ea;font-family:system-ui,sans-serif;margin:0;padding:20px}
 .wrap{max-width:760px;margin:0 auto}
 h1{font-size:18px;font-weight:600;margin:0 0 4px}
 .sub{font-size:12px;color:#8b94a0;margin:0 0 12px}
 img{width:640px;height:auto;border-radius:10px;background:#000;display:block}
 .row{display:flex;gap:8px;margin-top:14px}
 input{flex:1;padding:11px;border-radius:8px;border:1px solid #333;background:#1b1b1f;color:#eee;font-size:15px}
 button{padding:11px 16px;border-radius:8px;border:0;color:#fff;font-size:15px;cursor:pointer}
 .send{background:#3b82f6}.stop{background:#ef4444}.rand{background:#8b5cf6}
 .meta{margin-top:12px;font-size:13px;color:#9aa3ad}
 .panel{margin-top:12px;background:#16171b;border:1px solid #26272c;border-radius:10px;padding:12px;font-size:13px}
 .chip{display:inline-block;background:#23252b;border-radius:14px;padding:3px 10px;margin:3px 4px 0 0;color:#cdd3da}
 a{color:#7aa2ff} video{width:640px;border-radius:10px;margin-top:10px;background:#000}
</style></head><body><div class=wrap>
<h1>MolmoAct2 x LIBERO - fine-tuned checkpoint (synchronous)</h1>
<p class=sub>Deterministic closed-loop rollout: the policy plans an action chunk, the sim executes it, then it re-plans (the same receding-horizon loop as the Step-5 eval). The arm pauses briefly to think between chunks - no real-time blending.</p>
<img id="sim" alt="sim">
<div class=row>
 <input id=cmd placeholder="type an instruction, then Send (resets the scene and runs it)">
 <button class=send onclick=send()>Send</button>
 <button class=stop onclick=stop()>Stop</button>
 <button class=rand onclick=rnd()>Randomize</button>
</div>
<div class=meta id=meta>status: loading...</div>
<div class=panel><b id=scene>scene</b><div id=objs></div></div>
<div id=vidwrap></div>
</div>
<script>
async function send(){const v=document.getElementById('cmd').value;if(!v)return;
 await fetch('command',{method:'POST',headers:{'Content-Type':'application/x-www-form-urlencoded'},body:'instruction='+encodeURIComponent(v)});}
async function stop(){await fetch('stop',{method:'POST'});}
async function rnd(){await fetch('randomize',{method:'POST'});}
document.getElementById('cmd').addEventListener('keydown',e=>{if(e.key==='Enter')send();});
let lastVid='';
async function poll(){
 try{const s=await(await fetch('status')).json();
  document.getElementById('meta').textContent='['+s.mode+'] '+s.status+' | step '+s.step+' | last infer '+s.infer_ms+' ms'+(s.instruction?' | running: "'+s.instruction+'"':'');
  document.getElementById('scene').textContent='Scene: '+s.suite+' / task '+s.task_id+' - "'+s.scene_task+'"';
  document.getElementById('objs').innerHTML='Objects in scene: '+(s.objects||[]).map(o=>'<span class=chip>'+o+'</span>').join('');
  const box=document.getElementById('cmd');
  if(s.mode==='idle'&&!box.value&&s.scene_task)box.value=s.scene_task;
  if(s.video_url&&s.video_url!==lastVid){lastVid=s.video_url;
   document.getElementById('vidwrap').innerHTML='<div style=\"margin-top:8px;font-size:13px;color:#9aa3ad\">last run video:</div><video controls autoplay loop src=\"'+s.video_url+'\"></video>';}
 }catch(e){}
 setTimeout(poll,800);
}
function frameTick(){var n=new Image();n.onload=function(){var im=document.getElementById('sim');if(im)im.src=n.src;requestAnimationFrame(frameTick);};n.onerror=function(){setTimeout(frameTick,200);};n.src='frame?t='+Date.now();}
frameTick();
poll();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            self._send(200, "text/html; charset=utf-8", PAGE)
        elif path == "/status":
            with LOCK:
                s = {k: STATE[k] for k in ("mode", "status", "instruction", "scene_task",
                                           "suite", "task_id", "objects", "step", "infer_ms",
                                           "success", "video_url")}
            self._send(200, "application/json", json.dumps(s).encode())
        elif path == "/video":
            with LOCK:
                p = STATE.get("_video_path")
            if p and os.path.exists(p):
                with open(p, "rb") as f:
                    self._send(200, "video/mp4", f.read())
            else:
                self._send(404, "text/plain", b"no video")
        elif path == "/frame":
            with LOCK:
                frame = STATE.get("frame")
            if frame:
                self._send(200, "image/jpeg", frame)
            else:
                self._send(404, "text/plain", b"no frame")
        elif path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    with LOCK:
                        frame = STATE["frame"]
                    if frame:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode())
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                    time.sleep(0.06)
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self._send(404, "text/plain", b"not found")

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/command":
            n = int(self.headers.get("Content-Length", "0"))
            instr = parse_qs(self.rfile.read(n).decode()).get("instruction", [""])[0].strip()
            if instr:
                STOP["flag"] = True
                with LOCK:
                    PENDING["action"], PENDING["instruction"] = "run", instr
                EVENT.set()
            self.send_response(204)
            self.end_headers()
        elif path == "/stop":
            STOP["flag"] = True
            self.send_response(204)
            self.end_headers()
        elif path == "/randomize":
            STOP["flag"] = True
            with LOCK:
                PENDING["action"] = "randomize"
            EVENT.set()
            self.send_response(204)
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()


def _engine_guard():
    """Run the rollout thread; on any unhandled error surface it in STATE so the UI shows
    'error' instead of hanging on 'loading' forever (important for the workshop)."""
    try:
        rollout_thread()
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        with LOCK:
            STATE["mode"] = "error"
            STATE["status"] = f"engine failed: {e}"


def main():
    threading.Thread(target=_engine_guard, daemon=True).start()
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"interactive (fine-tuned, synchronous) demo on http://localhost:{PORT}  "
          f"(remote box: ssh -L {PORT}:localhost:{PORT} <host>)", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
