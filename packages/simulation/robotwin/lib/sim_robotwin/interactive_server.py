# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Interactive RoboTwin demo (model-agnostic, chunk-replay).

Command-driven showcase: the sim sits IDLE on a task scene until you press Run (or send an
instruction); then it re-seeds the scene and rolls that one task to completion (or until
Stop), streaming the live composed 4-view (head|observer over left|right wrist) to the
browser as MJPEG and saving a debug MP4. Use the environment dropdown to switch to any
RoboTwin task; the last run's video stays on screen with a download link.

The policy is chosen at runtime via POLICY_FACTORY=module:function (default: the built-in
RandomPolicy shipped with this simulator). Any model that implements sim_robotwin.Policy
can drive this demo. Stdlib http.server only.

Env: TASK, TASK_CONFIG, SEED, PORT (8082), OUT_DIR (/sim_outputs), VIEW_RES (720),
VIDEO_RES (720), MAX_STEPS (0=task step limit), POLICY_FACTORY.
View: ssh -L 8082:localhost:8082 <host>.
"""
import json
import os
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from sim_robotwin.envutil import env_int, env_str
from sim_robotwin.policy import load_policy
from sim_robotwin.render import banner_frame, encode_jpeg, save_mp4, view_frame
from sim_robotwin.rollout import run_episode
from sim_robotwin.taskenv import RoboTwinScene, list_tasks

TASK = env_str("TASK", "click_bell")
TASK_CONFIG = env_str("TASK_CONFIG", "demo_clean")
SEED = env_int("SEED", 100000)
PORT = env_int("PORT", 8082)
VIEW_RES = env_int("VIEW_RES", 720)
VIDEO_RES = env_int("VIDEO_RES", 720)
MAX_STEPS = env_int("MAX_STEPS", 0)  # 0 -> use the task's step limit
OUT_DIR = env_str("OUT_DIR", "/sim_outputs")

STATE = {
    "mode": "loading", "instruction": "", "task": TASK,
    "task_config": TASK_CONFIG, "seed": SEED, "objects": [],
    "step": 0, "success": False, "status": "starting", "frame": None, "video_url": "",
}
LOCK = threading.Lock()
PENDING = {"action": None, "instruction": "", "max_steps": 0, "task": TASK}
EVENT = threading.Event()
STOP = {"flag": False}

_ENVS_CACHE = None


def _envs_json():
    """Cached env-picker payload: [{value, label}] over every RoboTwin task."""
    global _ENVS_CACHE
    if _ENVS_CACHE is None:
        _ENVS_CACHE = [{"value": f"task={t}", "label": t.replace("_", " ")} for t in list_tasks()]
    return _ENVS_CACHE


def _set_frame(rgb):
    with LOCK:
        STATE["frame"] = encode_jpeg(rgb)


def engine_thread():
    os.makedirs(os.path.join(OUT_DIR, "interactive"), exist_ok=True)
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
                   scene_task=instr, instruction="", objects=sc.scene_objects())
        if not keep_video:
            upd["video_url"] = ""
        with LOCK:
            STATE.update(upd)
            if not keep_video:
                STATE.pop("_video_path", None)
        _set_frame(view_frame(sc.eval_frame(), height=VIEW_RES))

    build(TASK, SEED)
    show_idle()

    def run_command(instruction, max_steps):
        state["seed"] += 1
        build(state["task"], state["seed"])
        sc = state["scene"]
        instr = instruction or sc.default_instruction()
        with LOCK:
            STATE.update(mode="running", instruction=instr, step=0, success=False,
                         task=sc.task_name, seed=state["seed"],
                         status=f"running: {sc.task_name}", video_url="")
            STATE.pop("_video_path", None)
        STOP["flag"] = False
        frames = []

        def on_frame(rgb, step, replanning):
            _set_frame(view_frame(rgb, height=VIEW_RES))
            frames.append(banner_frame(rgb, f"{sc.task_name}: {instr}", VIDEO_RES))
            with LOCK:
                STATE["step"] = step

        limit = max_steps or MAX_STEPS or None
        success, _ = run_episode(sc, policy, instr, on_frame=on_frame,
                                 should_stop=lambda: STOP["flag"], max_steps=limit)

        url = ""
        if frames:
            ts = datetime.now().strftime("%H%M%S")
            name = f"interactive/{ts}_{sc.task_name}_{'ok' if success else 'run'}.mp4"
            path = os.path.join(OUT_DIR, name)
            try:
                save_mp4(frames, path, fps=15)
                url = "/video?ts=" + ts
                with LOCK:
                    STATE["_video_path"] = path
            except Exception as e:  # noqa: BLE001
                print("video save failed:", e, flush=True)

        with LOCK:
            STATE.update(mode="idle", success=success, video_url=url,
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
<title>FastWAM sim - RoboTwin (live)</title>
<style>
 body{background:#0f1012;color:#e8e8ea;font-family:system-ui,sans-serif;margin:0;padding:20px}
 .wrap{max-width:1000px;margin:0 auto}
 h1{font-size:18px;font-weight:600;margin:0 0 12px}
 img{width:100%;max-width:980px;height:auto;border-radius:10px;background:#000;display:block}
 .row{display:flex;gap:8px;margin-top:14px;align-items:center;flex-wrap:wrap}
 input[type=text],#cmd{flex:1;min-width:220px;padding:11px;border-radius:8px;border:1px solid #333;background:#1b1b1f;color:#eee;font-size:15px}
 select{padding:11px;border-radius:8px;border:1px solid #333;background:#1b1b1f;color:#eee;font-size:14px;max-width:100%}
 .lbl{font-size:13px;color:#9aa3ad;display:flex;gap:6px;align-items:center}
 .lbl input{width:90px;padding:9px;border-radius:8px;border:1px solid #333;background:#1b1b1f;color:#eee}
 button{padding:11px 16px;border-radius:8px;border:0;color:#fff;font-size:15px;cursor:pointer}
 .send{background:#3b82f6}.stop{background:#ef4444}
 .meta{margin-top:12px;font-size:13px;color:#9aa3ad}
 .panel{margin-top:12px;background:#16171b;border:1px solid #26272c;border-radius:10px;padding:12px;font-size:13px}
 .chip{display:inline-block;background:#23252b;border-radius:14px;padding:3px 10px;margin:3px 4px 0 0;color:#cdd3da}
 .vhead{margin-top:10px;font-size:13px;color:#9aa3ad}
 a{color:#7aa2ff} video{width:100%;max-width:980px;border-radius:10px;margin-top:8px;background:#000}
</style></head><body><div class=wrap>
<h1>FastWAM simulator - RoboTwin (live)</h1>
<img src="/stream" alt="sim">
<div class=row>
 <input id=cmd type=text placeholder="optional instruction; press Run to re-seed and roll the current task">
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
  document.getElementById('meta').textContent='['+s.mode+'] '+s.status+' | step '+s.step+(s.instruction?' | "'+s.instruction+'"':'');
  document.getElementById('scene').textContent='Task: '+s.task+' / '+s.task_config+' (seed '+s.seed+') - "'+(s.scene_task||'')+'"';
  document.getElementById('objs').innerHTML='Objects in scene: '+((s.objects||[]).length?(s.objects).map(o=>'<span class=chip>'+o+'</span>').join(''):'<span class=chip>n/a</span>');
  const box=document.getElementById('cmd');
  if(s.mode==='idle'&&!box.value&&s.scene_task)box.value=s.scene_task;
  const sel=document.getElementById('envsel');const want='task='+s.task;
  if(sel.value!==want&&[...sel.options].some(o=>o.value===want))sel.value=want;
  if(s.video_url&&s.video_url!==lastVid){lastVid=s.video_url;
   document.getElementById('vidwrap').innerHTML='<div class=vhead>last run video &nbsp;|&nbsp; <a href="'+s.video_url+'" download="fastwam_robotwin.mp4">download MP4</a></div><video controls autoplay loop src="'+s.video_url+'"></video>';}
 }catch(e){}
 setTimeout(poll,800);
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
                                               "success", "video_url")}
            self._send(200, "application/json", json.dumps(s).encode())
        elif path == "/video":
            with LOCK:
                p = STATE.get("_video_path")
            if p and os.path.exists(p):
                with open(p, "rb") as f:
                    self._send(200, "video/mp4", f.read(),
                               extra={"Content-Disposition": "attachment; filename=fastwam_robotwin.mp4"})
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
    print(f"interactive demo on http://0.0.0.0:{PORT}  (ssh -L {PORT}:localhost:{PORT} <host>)", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
