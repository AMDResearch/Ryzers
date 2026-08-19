# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Real-time RoboTwin demo (model-agnostic): planning decoupled from execution to SHOW planner latency.

EXEC thread owns the env and consumes one action/tick (HOLDs current pose when the buffer is
empty); PLANNER thread refills the buffer from an obs snapshot (never touches the env; SAPIEN
isn't thread-safe). Policy via POLICY_FACTORY (default RandomPolicy).

Env: TASK, TASK_CONFIG, SEED, PORT (8083), OUT_DIR (/sim_outputs), VIEW_RES (720),
VIDEO_RES (720), MAX_STEPS (0=task step limit), POLICY_FACTORY.
"""
import json
import os
import threading
import time
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

from sim_robotwin.envutil import env_int, env_str
from sim_robotwin.policy import load_policy
from sim_robotwin.render import banner_frame, encode_jpeg, save_mp4, view_frame
from sim_robotwin.taskenv import RoboTwinScene, list_tasks

TASK = env_str("TASK", "click_bell")
TASK_CONFIG = env_str("TASK_CONFIG", "demo_clean")
SEED = env_int("SEED", 100000)
PORT = env_int("PORT", 8083)
VIEW_RES = env_int("VIEW_RES", 720)
VIDEO_RES = env_int("VIDEO_RES", 720)
MAX_STEPS = env_int("MAX_STEPS", 0)
OUT_DIR = env_str("OUT_DIR", "/sim_outputs")

STATE = {
    "mode": "loading", "instruction": "", "task": TASK, "task_config": TASK_CONFIG,
    "seed": SEED, "objects": [], "step": 0, "success": False, "holding": False,
    "buffer": 0, "hold_pct": 0.0, "status": "starting", "frame": None, "video_url": "",
}
LOCK = threading.Lock()
PENDING = {"action": None, "instruction": "", "max_steps": 0, "task": TASK}
EVENT = threading.Event()
STOP = {"flag": False}

_ENVS_CACHE = None


def _envs_json():
    global _ENVS_CACHE
    if _ENVS_CACHE is None:
        _ENVS_CACHE = [{"value": f"task={t}", "label": t.replace("_", " ")} for t in list_tasks()]
    return _ENVS_CACHE


def _set_frame(rgb):
    with LOCK:
        STATE["frame"] = encode_jpeg(rgb)


def engine_thread():
    os.makedirs(os.path.join(OUT_DIR, "interactive_rt"), exist_ok=True)
    with LOCK:
        STATE["status"] = "loading policy ..."
    policy = load_policy()

    state = {"task": TASK, "seed": SEED, "scene": None}

    def build(task, seed):
        if state["scene"] is not None:
            state["scene"].close()
            state["scene"] = None
        with LOCK:
            STATE["status"] = f"loading scene {task} (seed {seed}) ..."
        scene = RoboTwinScene.build_stable(task, task_config=TASK_CONFIG, seed=seed)
        state["scene"] = scene
        state["task"], state["seed"] = task, scene.seed

    def show_idle(keep_video=False):
        sc = state["scene"]
        instr = sc.default_instruction()
        upd = dict(mode="idle", status="idle - press Run", step=0, success=False,
                   task=sc.task_name, task_config=sc.task_config, seed=state["seed"],
                   scene_task=instr, instruction="", objects=sc.scene_objects(),
                   holding=False, buffer=0, hold_pct=0.0)
        if not keep_video:
            upd["video_url"] = ""
        with LOCK:
            STATE.update(upd)
            if not keep_video:
                STATE.pop("_video_path", None)
        _set_frame(view_frame(sc.eval_frame(), height=VIEW_RES))

    build(TASK, SEED)
    with LOCK:
        STATE["status"] = "warming up policy ..."
    try:
        policy.reset(state["scene"].default_instruction())
        policy.warmup(state["scene"].get_obs(), state["scene"].default_instruction())
    except Exception as e:  # noqa: BLE001
        print("warmup failed:", e, flush=True)
    show_idle()

    def run_command(instruction, max_steps):
        state["seed"] += 1
        build(state["task"], state["seed"])
        sc = state["scene"]
        instr = instruction or sc.default_instruction()
        with LOCK:
            STATE.update(mode="running", instruction=instr, step=0, success=False,
                         task=sc.task_name, seed=state["seed"],
                         status=f"running: {sc.task_name}", video_url="", holding=False)
            STATE.pop("_video_path", None)
        STOP["flag"] = False
        replan_steps = int(getattr(policy, "replan_steps", 8))
        action_type = getattr(policy, "action_type", "qpos")
        limit = max_steps or MAX_STEPS or sc.step_lim
        policy.reset(instr)

        buf = deque()
        buflock = threading.Lock()
        obs0 = sc.get_obs()
        # HOLD re-issues the current pose in the policy's space (qpos vector, or EE pose for EE policies).
        hold_vec = sc.ee_state_vector(obs0) if action_type == "ee" else sc.state_vector(obs0)
        shared = {"obs": obs0, "done": False, "hold": 0, "total": 0,
                  "last_vec": hold_vec}
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
                try:
                    chunk = policy.predict_action_chunk(snap, instr)
                except Exception as e:  # noqa: BLE001
                    print("planner predict failed:", e, flush=True)
                    time.sleep(0.02)
                    continue
                with buflock:
                    for a in chunk[:replan_steps]:
                        buf.append(np.asarray(a))

        pth = threading.Thread(target=planner, daemon=True)
        pth.start()

        while sc.take_action_cnt < limit and not STOP["flag"]:
            with buflock:
                a = buf.popleft() if buf else None
            holding = a is None
            if holding:
                a = shared["last_vec"]
            else:
                shared["last_vec"] = np.asarray(a)
            sc.take_action(a, action_type=action_type)
            shared["total"] += 1
            shared["hold"] += int(holding)

            obs = sc.get_obs()
            with LOCK:
                shared["obs"] = obs
            rgb = sc.eval_frame()
            _set_frame(view_frame(rgb, height=VIEW_RES))
            tag = "THINKING" if holding else ""
            frames.append(banner_frame(rgb, f"{sc.task_name}: {instr}", VIDEO_RES, tag=tag))
            with buflock:
                bufn = len(buf)
            with LOCK:
                STATE.update(step=sc.take_action_cnt, holding=bool(tag), buffer=bufn,
                             hold_pct=round(100.0 * shared["hold"] / max(1, shared["total"]), 0))
            if sc.success:
                break

        shared["done"] = True
        success = sc.success
        pth.join(timeout=2.0)

        url = ""
        if frames:
            ts = datetime.now().strftime("%H%M%S")
            name = f"interactive_rt/{ts}_{sc.task_name}_{'ok' if success else 'run'}.mp4"
            path = os.path.join(OUT_DIR, name)
            try:
                save_mp4(frames, path, fps=15)
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
        show_idle(keep_video=True)

    while True:
        EVENT.wait()
        EVENT.clear()
        with LOCK:
            action = PENDING["action"]
            instruction = PENDING["instruction"]
            max_steps = int(PENDING.get("max_steps", 0) or 0)
            sel_task = PENDING.get("task", state["task"])
            PENDING["action"] = None
        if action == "select":
            STOP["flag"] = True
            try:
                build(sel_task, state["seed"] + 1)
            except Exception as e:  # noqa: BLE001
                with LOCK:
                    STATE["status"] = f"scene build failed: {e}"
                continue
            show_idle()
        elif action == "run":
            try:
                run_command(instruction, max_steps)
            except Exception as e:  # noqa: BLE001
                with LOCK:
                    STATE.update(mode="idle", status=f"run failed: {e}")


PAGE = b"""<!doctype html><html><head><meta charset=utf-8>
<title>FastWAM sim - RoboTwin (REAL-TIME)</title>
<style>
 body{background:#0f1012;color:#e8e8ea;font-family:system-ui,sans-serif;margin:0;padding:20px}
 .wrap{max-width:1000px;margin:0 auto}
 h1{font-size:18px;font-weight:600;margin:0 0 4px}
 .sub{font-size:12px;color:#8b94a0;margin:0 0 12px}
 img{width:100%;max-width:980px;height:auto;border-radius:10px;background:#000;display:block}
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
 a{color:#7aa2ff} video{width:100%;max-width:980px;border-radius:10px;margin-top:8px;background:#000}
</style></head><body><div class=wrap>
<h1>FastWAM simulator - RoboTwin (REAL-TIME)</h1>
<p class=sub>Execution is decoupled from planning; the arms HOLD (THINKING) while the policy plans, then resume when the action buffer refills.</p>
<img src="/stream" alt="sim">
<div class=row>
 <input id=cmd type=text placeholder="optional instruction; press Run to re-seed and roll the current task in real time">
 <button class=send onclick=send()>Run</button>
 <button class=stop onclick=stop()>Stop</button>
</div>
<div class=row>
 <span class=lbl>environment</span>
 <select id=envsel onchange=selEnv()><option value="">loading environments...</option></select>
 <label class=lbl>max steps <input id=msteps type=number min=0 step=50 value=0 title="0 = task step limit"></label>
</div>
<div class=meta id=meta>status: loading...</div>
<div class=panel><b id=scene>scene</b><div id=objs></div></div>
<div id=vidwrap></div>
</div>
<script>
async function send(){const v=document.getElementById('cmd').value;
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
  document.getElementById('scene').textContent='Task: '+s.task+' / '+s.task_config+' (seed '+s.seed+') - "'+(s.scene_task||'')+'"';
  document.getElementById('objs').innerHTML='Objects in scene: '+((s.objects||[]).length?(s.objects).map(o=>'<span class=chip>'+o+'</span>').join(''):'<span class=chip>n/a</span>');
  const box=document.getElementById('cmd');
  if(s.mode==='idle'&&!box.value&&s.scene_task)box.value=s.scene_task;
  const sel=document.getElementById('envsel');const want='task='+s.task;
  if(sel.value!==want&&[...sel.options].some(o=>o.value===want))sel.value=want;
  if(s.video_url&&s.video_url!==lastVid){lastVid=s.video_url;
   document.getElementById('vidwrap').innerHTML='<div class=vhead>last run video &nbsp;|&nbsp; <a href="'+s.video_url+'" download="fastwam_robotwin_rt.mp4">download MP4</a></div><video controls autoplay loop src="'+s.video_url+'"></video>';}
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
                s = {k: STATE.get(k) for k in ("mode", "status", "instruction", "scene_task",
                                               "task", "task_config", "seed", "objects", "step",
                                               "success", "holding", "buffer", "hold_pct",
                                               "video_url")}
            self._send(200, "application/json", json.dumps(s).encode())
        elif path == "/video":
            with LOCK:
                p = STATE.get("_video_path")
            if p and os.path.exists(p):
                with open(p, "rb") as f:
                    self._send(200, "video/mp4", f.read(),
                               extra={"Content-Disposition": "attachment; filename=fastwam_robotwin_rt.mp4"})
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
            STOP["flag"] = True
            with LOCK:
                PENDING.update(action="run", instruction=instr, max_steps=int(ms or 0))
            EVENT.set()
            self.send_response(204)
            self.end_headers()
        elif path == "/select":
            n = int(self.headers.get("Content-Length", "0"))
            q = parse_qs(self.rfile.read(n).decode())
            task = q.get("task", [""])[0].strip()
            if task:
                STOP["flag"] = True
                with LOCK:
                    PENDING.update(action="select", task=task)
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
