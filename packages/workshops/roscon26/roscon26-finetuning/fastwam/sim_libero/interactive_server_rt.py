# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Real-time (_RT) LIBERO demo (model-agnostic).

The real-time sibling of interactive_server.py. The clean demo freezes the world during
each policy forward and fast-replays the chunk, hiding planner latency. Here planning is
decoupled from the simulator so you can SEE the latency:

  * a SIM thread owns the MuJoCo env and steps it at wall-clock RT_HZ (LIBERO's control
    rate). Each tick it consumes one action from a shared buffer; if the buffer is empty
    (the planner is still thinking) it applies a HOLD action so the robot stays put.
  * a PLANNER thread runs policy.predict_action_chunk on a snapshot of the latest obs to
    refill the buffer. It never touches the env (MuJoCo isn't thread-safe).

HOLD is safe because LIBERO uses an OSC_POSE controller with control_delta=True: the 7-D
action is [dx,dy,dz, droll,dpitch,dyaw, gripper]. Zeroing the 6 delta dims => target ==
current pose => the controller holds position. The gripper dim is absolute, so HOLD keeps
the LAST commanded gripper (don't drop what you're holding). Replaying the last motion
action would re-integrate the delta and drift, so HOLD must be zeros.

The policy is chosen via POLICY_FACTORY=module:function (default RandomPolicy). Use the
environment dropdown to switch tasks; the last run's video stays on screen with a download
link.

Env: SUITE, TASK_ID, SEED, PORT (8081), OUT_DIR (/sim_outputs), VIEW_RES (720),
VIDEO_RES (720), RENDER_RES (512, HD sim render), RT_HZ (20), RT_MAX_STEPS (1200),
MAX_STEPS (0=suite default), POLICY_FACTORY.
"""
import copy
import json
import os
import threading
import time
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

from sim_libero.envutil import env_float, env_int, env_str
from sim_libero.libero_env import get_libero_dummy_action, get_libero_image, get_max_steps, list_envs
from sim_libero.policy import load_policy
from sim_libero.render import banner_frame, compose_view, encode_jpeg, save_mp4
from sim_libero.scene import build_scene

SUITE = env_str("SUITE", "libero_object")
TASK_ID = env_int("TASK_ID", 0)
SEED = env_int("SEED", 1000)
PORT = env_int("PORT", 8081)
VIEW_RES = env_int("VIEW_RES", 720)
VIDEO_RES = env_int("VIDEO_RES", 720)
RENDER_RES = env_int("RENDER_RES", 512)
RT_HZ = env_float("RT_HZ", 20)
RT_MAX_STEPS = env_int("RT_MAX_STEPS", 1200)
MAX_STEPS = env_int("MAX_STEPS", 0)      # 0 -> per-suite default horizon
OUT_DIR = env_str("OUT_DIR", "/sim_outputs")
DT = 1.0 / RT_HZ

STATE = {
    "mode": "loading", "instruction": "", "scene_task": "",
    "suite": SUITE, "task_id": TASK_ID, "objects": [],
    "step": 0, "success": False, "holding": False, "buffer": 0, "hold_pct": 0.0,
    "status": "starting", "frame": None, "video_url": "",
}
LOCK = threading.Lock()
PENDING = {"action": None, "instruction": "", "max_steps": 0, "suite": SUITE, "task_id": TASK_ID}
EVENT = threading.Event()
STOP = {"flag": False}

_ENVS_CACHE = None


def _envs_json():
    global _ENVS_CACHE
    if _ENVS_CACHE is None:
        items = []
        for e in list_envs():
            desc = (e["description"] or "").strip()
            label = f"{e['suite']} / task {e['task_id']}" + (f" - {desc[:44]}" if desc else "")
            items.append({"value": f"suite={e['suite']}&task_id={e['task_id']}", "label": label})
        _ENVS_CACHE = items
    return _ENVS_CACHE


def _set_frame(rgb):
    with LOCK:
        STATE["frame"] = encode_jpeg(rgb)


def engine_thread():
    os.makedirs(os.path.join(OUT_DIR, "interactive_rt"), exist_ok=True)
    with LOCK:
        STATE["status"] = "loading policy ..."
    policy = load_policy()

    def show_idle(sc, keep_video=False):
        obs = sc.reset()
        upd = dict(mode="idle", status="idle - send an instruction", step=0,
                   success=False, suite=sc.suite, task_id=sc.task_id,
                   scene_task=sc.description, objects=sc.objects, instruction="",
                   holding=False, buffer=0, hold_pct=0.0)
        if not keep_video:
            upd["video_url"] = ""
        with LOCK:
            STATE.update(upd)
            if not keep_video:
                STATE.pop("_video_path", None)
        _set_frame(compose_view(get_libero_image(obs), height=VIEW_RES))

    scene = build_scene(SUITE, TASK_ID, seed=SEED, resolution=RENDER_RES)

    # Warm up the policy once so the first real episode isn't stalled by JIT/compile.
    with LOCK:
        STATE["status"] = "warming up policy ..."
    try:
        wobs = scene.reset()
        policy.reset(scene.description)
        policy.warmup(wobs, scene.description or "pick up the object")
    except Exception as e:  # noqa: BLE001
        print("warmup failed:", e, flush=True)
    show_idle(scene)

    def run_command(sc, instruction, max_steps):
        with LOCK:
            STATE.update(mode="running", instruction=instruction, step=0, success=False,
                         status=f"running: {instruction}", video_url="", holding=False)
            STATE.pop("_video_path", None)
        STOP["flag"] = False
        replan_steps = int(getattr(policy, "replan_steps", 5))
        num_steps_wait = int(getattr(policy, "num_steps_wait", 5))
        horizon = max_steps or get_max_steps(sc.suite)
        max_steps_eff = horizon + num_steps_wait
        if not max_steps:
            max_steps_eff = min(max_steps_eff, RT_MAX_STEPS)

        obs = sc.reset()
        policy.reset(instruction)

        buf = deque()
        buflock = threading.Lock()
        shared = {"obs": copy.deepcopy(obs), "last_gripper": -1.0, "done": False,
                  "hold": 0, "total": 0, "infer_ms": 0.0}
        frames = []

        def planner():
            while not STOP["flag"] and not shared["done"]:
                with buflock:
                    have = len(buf)
                if have > 0:
                    time.sleep(0.005)
                    continue
                with LOCK:
                    snap = shared["obs"]
                t0 = time.perf_counter()
                try:
                    chunk = policy.predict_action_chunk(snap, instruction)
                except Exception as e:  # noqa: BLE001
                    print("planner predict failed:", e, flush=True)
                    time.sleep(0.02)
                    continue
                shared["infer_ms"] = round((time.perf_counter() - t0) * 1000.0, 0)
                with buflock:
                    for a in chunk[:replan_steps]:
                        buf.append(np.asarray(a, dtype=np.float32))

        pth = threading.Thread(target=planner, daemon=True)
        pth.start()

        t = 0
        done = False
        next_tick = time.perf_counter()
        while t < max_steps_eff and not STOP["flag"]:
            if t < num_steps_wait:
                obs, _, done, _ = sc.env.step(get_libero_dummy_action())
            else:
                with buflock:
                    a = buf.popleft() if buf else None
                holding = a is None
                if holding:
                    a = np.array([0, 0, 0, 0, 0, 0, shared["last_gripper"]], dtype=np.float32)
                else:
                    shared["last_gripper"] = float(a[6])
                obs, _, done, _ = sc.env.step(list(a))
                shared["total"] += 1
                shared["hold"] += int(holding)

            with LOCK:
                shared["obs"] = copy.deepcopy(obs)
            imgs = get_libero_image(obs)
            rgb = compose_view(imgs, height=VIEW_RES)
            _set_frame(rgb)
            tag = "THINKING" if (t >= num_steps_wait and holding) else ""
            frames.append(banner_frame(compose_view(imgs), instruction, VIDEO_RES, tag=tag))
            with buflock:
                bufn = len(buf)
            with LOCK:
                STATE.update(step=t, holding=bool(tag), buffer=bufn,
                             hold_pct=round(100.0 * shared["hold"] / max(1, shared["total"]), 0))
            t += 1
            if done:
                shared["done"] = True
                break
            next_tick += DT
            sleep = next_tick - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_tick = time.perf_counter()

        shared["done"] = True
        success = bool(shared["done"] and done)
        pth.join(timeout=2.0)

        url = ""
        if frames:
            ts = datetime.now().strftime("%H%M%S")
            name = f"interactive_rt/{ts}_{sc.suite}_{sc.task_id}_{'ok' if success else 'run'}.mp4"
            path = os.path.join(OUT_DIR, name)
            try:
                save_mp4(frames, path, fps=RT_HZ)
                url = "/video?ts=" + ts
                with LOCK:
                    STATE["_video_path"] = path
            except Exception as e:  # noqa: BLE001
                print("video save failed:", e, flush=True)

        hp = 100.0 * shared["hold"] / max(1, shared["total"])
        print(f"[rt] steps={shared['total']} hold%={hp:.0f} success={success}", flush=True)
        with LOCK:
            STATE.update(mode="idle", success=success, video_url=url, holding=False,
                         status=("success" if success else ("stopped" if STOP["flag"] else "done")))
        show_idle(sc, keep_video=True)

    while True:
        EVENT.wait()
        EVENT.clear()
        with LOCK:
            action = PENDING["action"]
            instruction = PENDING["instruction"]
            max_steps = int(PENDING.get("max_steps", 0) or 0)
            sel_suite = PENDING.get("suite", SUITE)
            sel_task = int(PENDING.get("task_id", 0) or 0)
            PENDING["action"] = None
        if action == "select":
            STOP["flag"] = True
            with LOCK:
                STATE["status"] = f"loading scene {sel_suite}/{sel_task} ..."
            try:
                new_scene = build_scene(sel_suite, sel_task, seed=SEED, resolution=RENDER_RES)
            except Exception as e:  # noqa: BLE001
                with LOCK:
                    STATE["status"] = f"scene build failed: {e}"
                continue
            scene.close()
            scene = new_scene
            show_idle(scene)
        elif action == "run":
            run_command(scene, instruction, max_steps)


PAGE = b"""<!doctype html><html><head><meta charset=utf-8>
<title>FastWAM sim - LIBERO (REAL-TIME)</title>
<style>
 body{background:#0f1012;color:#e8e8ea;font-family:system-ui,sans-serif;margin:0;padding:20px}
 .wrap{max-width:960px;margin:0 auto}
 h1{font-size:18px;font-weight:600;margin:0 0 4px}
 .sub{font-size:12px;color:#8b94a0;margin:0 0 12px}
 img{width:100%;max-width:940px;height:auto;border-radius:10px;background:#000;display:block}
 .row{display:flex;gap:8px;margin-top:14px;align-items:center;flex-wrap:wrap}
 input[type=text],#cmd{flex:1;min-width:220px;padding:11px;border-radius:8px;border:1px solid #333;background:#1b1b1f;color:#eee;font-size:15px}
 select{padding:11px;border-radius:8px;border:1px solid #333;background:#1b1b1f;color:#eee;font-size:14px;max-width:100%}
 .lbl{font-size:13px;color:#9aa3ad;display:flex;gap:6px;align-items:center}
 .lbl input{width:90px;padding:9px;border-radius:8px;border:1px solid #333;background:#1b1b1f;color:#eee}
 button{padding:11px 16px;border-radius:8px;border:0;color:#fff;font-size:15px;cursor:pointer}
 .send{background:#3b82f6}.stop{background:#ef4444}
 .meta{margin-top:12px;font-size:13px;color:#9aa3ad}
 .think{color:#ffb450;font-weight:600}
 .panel{margin-top:12px;background:#16171b;border:1px solid #26272c;border-radius:10px;padding:12px;font-size:13px}
 .chip{display:inline-block;background:#23252b;border-radius:14px;padding:3px 10px;margin:3px 4px 0 0;color:#cdd3da}
 .vhead{margin-top:10px;font-size:13px;color:#9aa3ad}
 a{color:#7aa2ff} video{width:100%;max-width:940px;border-radius:10px;margin-top:8px;background:#000}
</style></head><body><div class=wrap>
<h1>FastWAM simulator - LIBERO (REAL-TIME)</h1>
<p class=sub>The simulator runs at wall-clock speed; the robot pauses (THINKING) while the policy plans, then resumes when the action buffer refills.</p>
<img src="/stream" alt="sim">
<div class=row>
 <input id=cmd type=text placeholder="type an instruction, then Send (resets the scene and runs it in real time)">
 <button class=send onclick=send()>Send</button>
 <button class=stop onclick=stop()>Stop</button>
</div>
<div class=row>
 <span class=lbl>environment</span>
 <select id=envsel onchange=selEnv()><option value="">loading environments...</option></select>
 <label class=lbl>max steps <input id=msteps type=number min=0 step=50 value=0 title="0 = suite default horizon"></label>
</div>
<div class=meta id=meta>status: loading...</div>
<div class=panel><b id=scene>scene</b><div id=objs></div></div>
<div id=vidwrap></div>
</div>
<script>
async function send(){const v=document.getElementById('cmd').value;if(!v)return;
 const ms=document.getElementById('msteps').value||'0';
 await fetch('/command',{method:'POST',headers:{'Content-Type':'application/x-www-form-urlencoded'},body:'instruction='+encodeURIComponent(v)+'&max_steps='+encodeURIComponent(ms)});}
async function stop(){await fetch('/stop',{method:'POST'});}
async function selEnv(){const s=document.getElementById('envsel');const o=s.options[s.selectedIndex];
 if(!o||!o.value)return;await fetch('/select',{method:'POST',headers:{'Content-Type':'application/x-www-form-urlencoded'},body:o.value});}
async function loadEnvs(){try{const es=await(await fetch('/envs')).json();const s=document.getElementById('envsel');
 s.innerHTML='';for(const e of es){const o=document.createElement('option');o.value=e.value;o.textContent=e.label;s.appendChild(o);}}catch(e){}}
document.getElementById('cmd').addEventListener('keydown',e=>{if(e.key==='Enter')send();});
let lastVid='';
async function poll(){
 try{const s=await(await fetch('/status')).json();
  const m=document.getElementById('meta');
  m.textContent='['+s.mode+'] '+s.status+' | step '+s.step+' | buffer '+s.buffer+' | hold '+s.hold_pct+'%';
  m.className='meta'+(s.holding?' think':'');
  document.getElementById('scene').textContent='Scene: '+s.suite+' / task '+s.task_id+' - "'+s.scene_task+'"';
  document.getElementById('objs').innerHTML='Objects in scene: '+((s.objects||[]).length?(s.objects).map(o=>'<span class=chip>'+o+'</span>').join(''):'<span class=chip>n/a</span>');
  const box=document.getElementById('cmd');
  if(s.mode==='idle'&&!box.value&&s.scene_task)box.value=s.scene_task;
  const sel=document.getElementById('envsel');const want='suite='+s.suite+'&task_id='+s.task_id;
  if(sel.value!==want&&[...sel.options].some(o=>o.value===want))sel.value=want;
  if(s.video_url&&s.video_url!==lastVid){lastVid=s.video_url;
   document.getElementById('vidwrap').innerHTML='<div class=vhead>last run video &nbsp;|&nbsp; <a href="'+s.video_url+'" download="fastwam_libero_rt.mp4">download MP4</a></div><video controls autoplay loop src="'+s.video_url+'"></video>';}
 }catch(e){}
 setTimeout(poll,500);
}
loadEnvs();poll();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, ctype, body, extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            self._send(200, "text/html; charset=utf-8", PAGE)
        elif path == "/envs":
            self._send(200, "application/json", json.dumps(_envs_json()).encode())
        elif path == "/status":
            with LOCK:
                s = {k: STATE[k] for k in ("mode", "status", "instruction", "scene_task",
                                           "suite", "task_id", "objects", "step",
                                           "success", "holding", "buffer", "hold_pct", "video_url")}
            self._send(200, "application/json", json.dumps(s).encode())
        elif path == "/video":
            with LOCK:
                p = STATE.get("_video_path")
            if p and os.path.exists(p):
                with open(p, "rb") as f:
                    self._send(200, "video/mp4", f.read(),
                               extra={"Content-Disposition": "attachment; filename=fastwam_libero_rt.mp4"})
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
            q = parse_qs(self.rfile.read(n).decode())
            instr = q.get("instruction", [""])[0].strip()
            ms = q.get("max_steps", ["0"])[0]
            if instr:
                STOP["flag"] = True
                with LOCK:
                    PENDING.update(action="run", instruction=instr, max_steps=int(ms or 0))
                EVENT.set()
            self.send_response(204)
            self.end_headers()
        elif path == "/select":
            n = int(self.headers.get("Content-Length", "0"))
            q = parse_qs(self.rfile.read(n).decode())
            suite = q.get("suite", [""])[0].strip()
            tid = q.get("task_id", ["0"])[0]
            if suite:
                STOP["flag"] = True
                with LOCK:
                    PENDING.update(action="select", suite=suite, task_id=int(tid or 0))
                EVENT.set()
            self.send_response(204)
            self.end_headers()
        elif path == "/stop":
            STOP["flag"] = True
            self.send_response(204)
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()


def main():
    threading.Thread(target=engine_thread, daemon=True).start()
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"real-time demo on http://0.0.0.0:{PORT}  (ssh -L {PORT}:localhost:{PORT} <host>)", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
