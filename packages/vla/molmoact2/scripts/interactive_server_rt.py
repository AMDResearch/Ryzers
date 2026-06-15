# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Real-time MolmoAct2 + LIBERO demo: the sim keeps running while the model thinks.

Real-time sibling of interactive_server.py. Planning is decoupled from the simulator so
the planner latency is visible:
  * a sim thread steps MuJoCo at wall-clock RT_HZ, consuming one action per tick from a
    shared buffer; when the buffer is empty it applies a HOLD action so the arm stays put.
  * a planner thread runs the policy forward to refill the buffer with the next chunk.

HOLD is zeros on the 6 delta dims (OSC_POSE control_delta=True holds the current pose) and
keeps the last gripper command. Reuses the lerobot select_action path; only the loop is local.

Env: SUITE, TASK_ID, SEED, THINK (1), CKPT, PORT (8081), VIEW_RES (720), VIDEO_RES (600),
RT_HZ (20), RT_LOOKAHEAD (0), OUT_DIR (/outputs).
"""
import io
import json
import os
import random
import threading
import time
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# `or` (not get's default) so an empty -e VAR= forwarded by `ryzers run` still falls back.
SUITE = os.environ.get("SUITE") or "libero_object"
TASK_ID = int(os.environ.get("TASK_ID") or "3")
SEED = int(os.environ.get("SEED") or "1000")
THINK = (os.environ.get("THINK") or "1") == "1"
CKPT = os.environ.get("CKPT") or "allenai/MolmoAct2-Think-LIBERO"
PORT = int(os.environ.get("PORT") or "8081")
VIEW_RES = int(os.environ.get("VIEW_RES") or "720")    # live viewport (RT favors smoothness)
VIDEO_RES = int(os.environ.get("VIDEO_RES") or "600")  # saved debug video (kept small)
RT_HZ = float(os.environ.get("RT_HZ") or "20")         # wall-clock control rate
RT_LOOKAHEAD = int(os.environ.get("RT_LOOKAHEAD") or "0")  # replan when buffer <= this (pipelining knob)
RT_MAX_STEPS = int(os.environ.get("RT_MAX_STEPS") or "1200")
# Optional: multiply the simulator's real horizon (robosuite) to play around with
# long-horizon tasks without the scene resetting mid-execution. Default 1 (unchanged).
RT_HORIZON_MULT = float(os.environ.get("RT_HORIZON_MULT") or "1")  # RT runs in wall time; holds burn budget, so allow completion
OUT_DIR = os.environ.get("OUT_DIR") or "/outputs"
SUITES = ["libero_object", "libero_goal", "libero_spatial", "libero_10"]
DT = 1.0 / RT_HZ

# ---- shared state ------------------------------------------------------------
STATE = {
    "mode": "loading",        # loading | idle | running
    "instruction": "",
    "scene_task": "",
    "suite": SUITE, "task_id": TASK_ID,
    "objects": [],
    "step": 0, "infer_ms": 0.0, "success": False,
    "holding": False, "buffer": 0, "hold_pct": 0.0,
    "status": "starting", "frame": None, "video_url": "",
}
LOCK = threading.Lock()
PENDING = {"action": None, "instruction": ""}
EVENT = threading.Event()
STOP = {"flag": False}


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


def _banner_frame(rgb, text, size, thinking):
    """Downscale to `size`; top banner shows the command + a THINKING tag when holding."""
    img = Image.fromarray(np.ascontiguousarray(rgb)).resize((size, size), Image.BILINEAR)
    bh = max(40, size // 12)
    bh += bh % 2
    canvas = Image.new("RGB", (size, size + bh), (15, 15, 18))
    canvas.paste(img, (0, bh))
    d = ImageDraw.Draw(canvas)
    f = _font(max(14, size // 30))
    cap = 44 if thinking else 60  # leave room on the right for the THINKING tag
    msg = text if len(text) <= cap else text[:cap - 3] + "..."
    d.text((10, bh // 2), msg, fill=(240, 240, 240), font=f, anchor="lm")
    if thinking:
        d.text((size - 10, bh // 2), "THINKING", fill=(255, 180, 80), font=f, anchor="rm")
    return np.asarray(canvas)


class Scene:
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


def engine_thread():
    import torch
    import draccus
    from lerobot.configs.eval import EvalPipelineConfig
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    import lerobot.scripts.lerobot_eval as ev

    ACTION = ev.ACTION
    preprocess_observation = ev.preprocess_observation

    depth = "True" if THINK else "False"
    base_args = [
        "--policy.type=molmoact2", f"--policy.checkpoint_path={CKPT}",
        "--policy.inference_action_mode=continuous",
        f"--policy.enable_depth_reasoning={depth}", f"--policy.enable_adaptive_depth={depth}",
        "--policy.enable_cuda_graph=False", "--policy.norm_tag=libero",
        "--policy.device=cuda", "--env.type=libero",
        "--eval.batch_size=1", "--eval.n_episodes=1",
        f"--seed={SEED}", "--output_dir=/tmp/interactive_rt_eval",
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
        for e in env.envs:
            # robosuite's internal horizon is the real terminator (the gym wrapper never
            # truncates on _max_episode_steps); raise it by RT_HORIZON_MULT so long tasks
            # do not reset mid-execution. _max_episode_steps caps our own control loop.
            base_h = RT_MAX_STEPS
            rs = getattr(getattr(e, "_env", None), "env", None)
            if rs is not None and getattr(rs, "horizon", None):
                base_h = int(rs.horizon)
            new_h = int(base_h * RT_HORIZON_MULT)
            if rs is not None:
                try:
                    rs.horizon = new_h
                except Exception:
                    pass
            e._max_episode_steps = max(RT_MAX_STEPS, new_h)
        try:
            print(f"[scene] horizon_mult={RT_HORIZON_MULT} horizon={new_h} "
                  f"max_steps={max(RT_MAX_STEPS, new_h)}", flush=True)
        except Exception:
            pass
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
            STATE.update(mode="idle", status="idle, send an instruction", step=0,
                         success=False, suite=sc.suite, task_id=sc.task_id,
                         scene_task=sc.scene_task, objects=sc.objects,
                         instruction="", video_url="", holding=False, buffer=0, hold_pct=0.0)
        STATE.pop("_video_path", None)
        _set_frame(sc.idle_frame(VIEW_RES))

    os.makedirs(os.path.join(OUT_DIR, "interactive_rt"), exist_ok=True)
    scene = build_scene(SUITE, TASK_ID)

    # ---- one planning step: model forward -> full env-space chunk -------------
    def plan_chunk(proc_obs, generator):
        with torch.inference_mode():
            a0 = policy.select_action(proc_obs, generator=generator)  # forward + pop 1
        raw = [a0]
        q = policy._action_queues[0]
        while q:                                                      # drain the rest
            a = q.popleft()
            if a.ndim == 1:
                a = a.unsqueeze(0)
            raw.append(a.to(device=a0.device, dtype=torch.float32))
        out = []
        for a in raw:
            pa = postprocessor(a)
            pa = env_post({ACTION: pa})[ACTION]
            out.append(pa.to("cpu").numpy())
        infer_s = float(getattr(policy, "_last_model_inference_s", 0.0))
        print(f"[plan] infer={infer_s:.2f}s chunk={len(out)}", flush=True)
        return out, infer_s

    def run_command(sc, instruction):
        with LOCK:
            STATE.update(mode="running", instruction=instruction, step=0, success=False,
                         status=f"running: {instruction}", video_url="", holding=False,
                         buffer=0, hold_pct=0.0)
        STOP["flag"] = False
        policy.reset()
        obs, _ = sc.env.reset(seed=SEED)
        generator = ev._make_rollout_action_generator(policy, [SEED])
        max_steps = sc.env.call("_max_episode_steps")[0]

        buffer = deque()
        buf_lock = threading.Lock()
        shared = {"obs": obs, "done": False, "last_gripper": 0.0,
                  "hold_steps": 0, "total_steps": 0}

        def planner():
            while not STOP["flag"] and not shared["done"]:
                with buf_lock:
                    have = len(buffer)
                if have > RT_LOOKAHEAD:
                    time.sleep(0.005)
                    continue
                with LOCK:
                    cur = shared["obs"]
                try:
                    proc = preprocess_observation(cur)
                    proc["task"] = [instruction for _ in range(sc.env.num_envs)]
                    proc = env_pre(proc)
                    proc = preprocessor(proc)
                    chunk, infer_s = plan_chunk(proc, generator)
                except Exception as e:  # noqa: BLE001
                    print("plan error:", e, flush=True)
                    time.sleep(0.02)
                    continue
                with buf_lock:
                    buffer.extend(chunk)
                with LOCK:
                    if infer_s > 0:
                        STATE["infer_ms"] = round(infer_s * 1000.0, 0)

        pth = threading.Thread(target=planner, daemon=True)
        pth.start()

        frames = []
        success = False
        next_t = time.perf_counter()
        step = 0
        while not STOP["flag"] and not shared["done"] and step < max_steps:
            with buf_lock:
                action = buffer.popleft() if buffer else None
                depth_buf = len(buffer)
            holding = action is None
            if holding:
                action = np.zeros((sc.env.num_envs, 7), dtype=np.float32)
                action[:, 6] = shared["last_gripper"]
                shared["hold_steps"] += 1
            else:
                shared["last_gripper"] = float(action[0, 6])
            shared["total_steps"] += 1

            try:
                obs, reward, terminated, truncated, info = sc.env.step(action)
            except ValueError as e:
                # robosuite raises "executing action in terminated episode" when the
                # underlying env set its internal done but the gym wrapper still reported
                # terminated/truncated False; treat it as a clean episode end instead of
                # letting the exception kill the engine thread (which froze the demo).
                print("step after termination, ending episode:", e, flush=True)
                shared["done"] = True
                break
            with LOCK:
                shared["obs"] = obs
            if "final_info" in info:
                try:
                    success = bool(np.asarray(info["final_info"]["is_success"]).any())
                except Exception:
                    pass
            done = bool(np.any(terminated) or np.any(truncated))

            rgb = sc.hi_res(VIEW_RES)
            _set_frame(rgb)
            frames.append(_banner_frame(rgb, instruction, VIDEO_RES, holding))
            step += 1
            hp = 100.0 * shared["hold_steps"] / max(1, shared["total_steps"])
            with LOCK:
                STATE.update(step=step, holding=holding, buffer=depth_buf, hold_pct=round(hp, 0),
                             status=("thinking (holding pose)" if holding else f"running: {instruction}"))
            if done:
                shared["done"] = True

            next_t += DT
            sleep = next_t - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.perf_counter()  # fell behind real-time; resync

        shared["done"] = True
        STOP["flag"] = True
        pth.join(timeout=5.0)

        url = ""
        if frames:
            ts = datetime.now().strftime("%H%M%S")
            name = f"interactive_rt/{ts}_{sc.suite}_{sc.task_id}_{'ok' if success else 'run'}.mp4"
            path = os.path.join(OUT_DIR, name)
            try:
                import imageio
                with imageio.get_writer(path, fps=int(RT_HZ), codec="libx264", quality=8,
                                        macro_block_size=1, output_params=["-pix_fmt", "yuv420p"]) as w:
                    for fr in frames:
                        w.append_data(fr)
                url = "/video?ts=" + ts
                with LOCK:
                    STATE["_video_path"] = path
            except Exception as e:  # noqa: BLE001
                print("video save failed:", e, flush=True)

        with LOCK:
            STATE.update(mode="idle", success=success, video_url=url, holding=False,
                         status=("success" if success else ("stopped" if STOP["flag"] else "done")))
        _set_frame(sc.idle_frame(VIEW_RES))

    # Warm up flash/JIT kernels NOW (the first model forward compiles them and can take
    # ~20s); otherwise that one-time stall would burn the whole first real-time episode.
    with LOCK:
        STATE["status"] = "warming up GPU kernels (one-time JIT) ..."
    try:
        wobs, _ = scene.env.reset(seed=SEED)
        wproc = preprocess_observation(wobs)
        wproc["task"] = [scene.scene_task or "pick up the object" for _ in range(scene.env.num_envs)]
        wproc = env_pre(wproc)
        wproc = preprocessor(wproc)
        wgen = ev._make_rollout_action_generator(policy, [SEED])
        _t = time.perf_counter()
        plan_chunk(wproc, wgen)
        policy.reset()
        print(f"[warmup] done in {time.perf_counter()-_t:.1f}s", flush=True)
    except Exception as e:  # noqa: BLE001
        import traceback; traceback.print_exc()
        print("warmup failed:", e, flush=True)
    show_idle(scene)

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
                scene.env.close()
            except Exception:
                pass
            scene = new_scene
            show_idle(scene)
        elif action == "run":
            try:
                run_command(scene, instruction)
            except Exception as e:  # noqa: BLE001
                import traceback; traceback.print_exc()
                with LOCK:
                    STATE.update(mode="idle", holding=False, status=f"error: {e}")
                try:
                    show_idle(scene)
                except Exception:
                    pass


# ---------------------------------------------------------------------------
PAGE = b"""<!doctype html><html><head><meta charset=utf-8>
<title>MolmoAct2 x LIBERO (REAL-TIME)</title>
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
 .think{color:#ffb450;font-weight:600}
 .panel{margin-top:12px;background:#16171b;border:1px solid #26272c;border-radius:10px;padding:12px;font-size:13px}
 .chip{display:inline-block;background:#23252b;border-radius:14px;padding:3px 10px;margin:3px 4px 0 0;color:#cdd3da}
 a{color:#7aa2ff} video{width:640px;border-radius:10px;margin-top:10px;background:#000}
</style></head><body><div class=wrap>
<h1>MolmoAct2 x LIBERO real-time sim</h1>
<p class=sub>The simulator runs at wall-clock speed; the robot HOLDS its pose while the model thinks, then moves when the next chunk lands.</p>
<img src="/stream" alt="sim">
<div class=row>
 <input id=cmd placeholder="type an instruction, then Send (resets the scene and runs it in real time)">
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
  let t='['+s.mode+'] '+s.status+' | step '+s.step+' | last infer '+s.infer_ms+' ms | buffer '+s.buffer+' | hold '+s.hold_pct+'%';
  const m=document.getElementById('meta'); m.textContent=t; m.className='meta'+(s.holding?' think':'');
  document.getElementById('scene').textContent='Scene: '+s.suite+' / task '+s.task_id+': "'+s.scene_task+'"';
  document.getElementById('objs').innerHTML='Objects in scene: '+(s.objects||[]).map(o=>'<span class=chip>'+o+'</span>').join('');
  const box=document.getElementById('cmd');
  if(s.mode==='idle'&&!box.value&&s.scene_task)box.value=s.scene_task;
  if(s.video_url&&s.video_url!==lastVid){lastVid=s.video_url;
   document.getElementById('vidwrap').innerHTML='<div style=\"margin-top:8px;font-size:13px;color:#9aa3ad\">last run video:</div><video controls autoplay loop src=\"'+s.video_url+'\"></video>';}
 }catch(e){}
 setTimeout(poll,500);
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
                                           "success", "holding", "buffer", "hold_pct", "video_url")}
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
                    time.sleep(0.04)
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


def main():
    threading.Thread(target=engine_thread, daemon=True).start()
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"real-time demo on http://localhost:{PORT}  (remote box: ssh -L {PORT}:localhost:{PORT} <host>)", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
