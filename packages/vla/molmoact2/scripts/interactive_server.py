# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Interactive MolmoAct2 x LIBERO demo (local inference, browser UI).

Command-driven: the sim sits idle on a scene until you send an instruction; then it
resets the env + policy and runs that one task to completion (or until you hit Stop),
streaming the FHD-equivalent camera to the browser as MJPEG and saving a debug video
(executed command banner on top). Randomize swaps to a new random scene.

Why reset on every command: the MolmoAct2-Think policy caches its depth/spatial plan
once per episode (and runs a receding-horizon action queue), so a new instruction only
takes effect after `policy.reset()` + a fresh `env.reset()`. Resetting per command is
both the desired UX and the fix for "it keeps doing the old task".

Reuses the lerobot eval machinery verbatim (make_env / make_policy /
processors / rollout); only the loop is ours (live instruction + streaming + video).
Stdlib http.server only (no extra deps).

Env: SUITE, TASK_ID, SEED, THINK (1), CKPT, PORT (8080), VIEW_RES (1080),
VIDEO_RES (600), OUT_DIR (/outputs). Open http://localhost:PORT (remote box:
ssh -L PORT:localhost:PORT <host>).
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
THINK = os.environ.get("THINK", "1") == "1"
CKPT = os.environ.get("CKPT", "allenai/MolmoAct2-Think-LIBERO")
PORT = int(os.environ.get("PORT") or "8080")
VIEW_RES = int(os.environ.get("VIEW_RES") or "1080")   # live viewport (FHD-equivalent)
VIDEO_RES = int(os.environ.get("VIDEO_RES") or "600")  # saved debug video (kept small)
OUT_DIR = os.environ.get("OUT_DIR", "/outputs")
SUITES = ["libero_object", "libero_goal", "libero_spatial", "libero_10"]

# ---- shared state ------------------------------------------------------------
STATE = {
    "mode": "loading",        # loading | idle | running
    "instruction": "",        # the instruction currently being executed
    "scene_task": "",         # the scene's native LIBERO instruction
    "suite": SUITE, "task_id": TASK_ID,
    "objects": [],            # object names visible in the scene
    "step": 0, "infer_ms": 0.0, "success": False,
    "status": "starting", "frame": None, "video_url": "",
}
LOCK = threading.Lock()
# command mailbox (latest wins): action in {"run","randomize",None}
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


def _encode_jpeg(rgb, quality=88):
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


def rollout_thread():
    import torch
    import draccus
    from lerobot.configs.eval import EvalPipelineConfig
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    import lerobot.scripts.lerobot_eval as ev

    # --- live instruction injection (replaces the scene's fixed task language) ---
    def _inject(env, observation):
        observation["task"] = [STATE["instruction"] for _ in range(env.num_envs)]
        return observation
    ev.add_envs_task = _inject

    depth = "True" if THINK else "False"
    base_args = [
        "--policy.type=molmoact2", f"--policy.checkpoint_path={CKPT}",
        "--policy.inference_action_mode=continuous",
        f"--policy.enable_depth_reasoning={depth}", f"--policy.enable_adaptive_depth={depth}",
        "--policy.enable_cuda_graph=False", "--policy.norm_tag=libero",
        "--policy.device=cuda", "--env.type=libero",
        "--eval.batch_size=1", "--eval.n_episodes=1",
        f"--seed={SEED}", "--output_dir=/tmp/interactive_eval",
    ]
    if os.environ.get("NUM_STEPS"):
        base_args.append(f"--policy.num_steps={os.environ['NUM_STEPS']}")

    with LOCK:
        STATE["status"] = "loading policy (first run JIT-compiles flash kernels) ..."
    base_cfg = draccus.parse(EvalPipelineConfig, args=base_args + [f"--env.task={SUITE}", f"--env.task_ids=[{TASK_ID}]"])
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
        cfg = draccus.parse(EvalPipelineConfig, args=base_args + [f"--env.task={suite}", f"--env.task_ids=[{task_id}]"])
        envs = make_env(cfg.env, n_envs=1, use_async_envs=False, trust_remote_code=cfg.trust_remote_code)
        env = envs[suite][task_id]
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
        with LOCK:
            STATE.update(mode="idle", status="idle - send an instruction", step=0,
                         success=False, suite=sc.suite, task_id=sc.task_id,
                         scene_task=sc.scene_task, objects=sc.objects,
                         instruction="", video_url="")
        STATE.pop("_video_path", None)
        _set_frame(sc.idle_frame(VIEW_RES))

    os.makedirs(os.path.join(OUT_DIR, "interactive"), exist_ok=True)
    scene = build_scene(SUITE, TASK_ID)
    show_idle(scene)

    def run_command(sc, instruction):
        with LOCK:
            STATE.update(mode="running", instruction=instruction, step=0, success=False,
                         status=f"running: {instruction}", video_url="")
        STOP["flag"] = False
        frames = []  # (banner) frames for the saved video

        def render_cb(vec_env):
            if STOP["flag"]:
                raise StopRollout
            rgb = sc.hi_res(VIEW_RES)
            _set_frame(rgb)
            frames.append(_banner_frame(rgb, instruction, VIDEO_RES))
            v = getattr(policy, "_last_model_inference_s", 0.0) * 1000.0
            with LOCK:
                STATE["step"] += 1
                if v > 0:  # only replan steps run the model; keep the last real timing
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

        # save debug video (command banner already on each frame)
        url = ""
        if frames:
            ts = datetime.now().strftime("%H%M%S")
            name = f"interactive/{ts}_{sc.suite}_{sc.task_id}_{'ok' if success else 'run'}.mp4"
            path = os.path.join(OUT_DIR, name)
            try:
                import imageio
                with imageio.get_writer(path, fps=20, codec="libx264", quality=8,
                                        macro_block_size=1, output_params=["-pix_fmt", "yuv420p"]) as w:
                    for fr in frames:
                        w.append_data(fr)
                url = "/video?ts=" + ts
                with LOCK:
                    STATE["_video_path"] = path
            except Exception as e:  # noqa: BLE001
                print("video save failed:", e, flush=True)

        with LOCK:
            STATE.update(mode="idle", success=success, video_url=url,
                         status=("success" if success else ("stopped" if STOP["flag"] else "done")))
        _set_frame(sc.idle_frame(VIEW_RES))

    # main control loop
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
                scene.env.close()  # free the old EGL/MuJoCo context
            except Exception:
                pass
            scene = new_scene
            show_idle(scene)
        elif action == "run":
            run_command(scene, instruction)


# ---------------------------------------------------------------------------
PAGE = b"""<!doctype html><html><head><meta charset=utf-8>
<title>MolmoAct2 x LIBERO (live)</title>
<style>
 body{background:#0f1012;color:#e8e8ea;font-family:system-ui,sans-serif;margin:0;padding:20px}
 .wrap{max-width:760px;margin:0 auto}
 h1{font-size:18px;font-weight:600;margin:0 0 12px}
 img{width:640px;height:auto;border-radius:10px;background:#000;display:block}
 .row{display:flex;gap:8px;margin-top:14px}
 input{flex:1;padding:11px;border-radius:8px;border:1px solid #333;background:#1b1b1f;color:#eee;font-size:15px}
 button{padding:11px 16px;border-radius:8px;border:0;color:#fff;font-size:15px;cursor:pointer}
 .send{background:#3b82f6}.stop{background:#ef4444}.rand{background:#8b5cf6}
 .meta{margin-top:12px;font-size:13px;color:#9aa3ad}
 .panel{margin-top:12px;background:#16171b;border:1px solid #26272c;border-radius:10px;padding:12px;font-size:13px}
 .chip{display:inline-block;background:#23252b;border-radius:14px;padding:3px 10px;margin:3px 4px 0 0;color:#cdd3da}
 a{color:#7aa2ff}
 video{width:640px;border-radius:10px;margin-top:10px;background:#000}
</style></head><body><div class=wrap>
<h1>MolmoAct2 x LIBERO - live sim</h1>
<img src="/stream" alt="sim">
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
 await fetch('/command',{method:'POST',headers:{'Content-Type':'application/x-www-form-urlencoded'},body:'instruction='+encodeURIComponent(v)});}
async function stop(){await fetch('/stop',{method:'POST'});}
async function rnd(){await fetch('/randomize',{method:'POST'});}
document.getElementById('cmd').addEventListener('keydown',e=>{if(e.key==='Enter')send();});
let lastVid='';
async function poll(){
 try{const s=await(await fetch('/status')).json();
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
                STOP["flag"] = True  # abort any running task, then run the new one
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


def main():
    threading.Thread(target=rollout_thread, daemon=True).start()
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"interactive demo on http://localhost:{PORT}  (remote box: ssh -L {PORT}:localhost:{PORT} <host>)", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
