# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Headless front-end for RAI's manipulation demo: adds a live view of the
# simulation to the upstream Streamlit app so no local display is needed.
# The view is the MJPEG stream web_video_server publishes from the simulation
# camera topic - the same images the agent reasons about.

import importlib.util
import os
import sys

import streamlit as st
import streamlit.components.v1 as components

RAI_EXAMPLES = "/ryzers/rai/examples"
DEMO_APP = f"{RAI_EXAMPLES}/manipulation-demo-streamlit.py"
O3DE_CONFIG = (
    "src/rai_bench/rai_bench/manipulation_o3de/predefined/configs/o3de_config.yaml"
)
CAMERA_TOPIC = "/color_image5"
JUPYTERHUB_SERVICE_PREFIX = os.environ.get("JUPYTERHUB_SERVICE_PREFIX")
STREAM_BASE_URL = (
    f"{JUPYTERHUB_SERVICE_PREFIX.rstrip('/')}/proxy/8080"
    if JUPYTERHUB_SERVICE_PREFIX
    else "http://localhost:8080"
)
STREAM_URL = (
    f"{STREAM_BASE_URL}/stream?topic={CAMERA_TOPIC}&quality=70&width=640&height=360"
)

st.set_page_config(page_title="RAI Manipulation Demo", page_icon=":robot:")

# Upstream calls set_page_config as well, and Streamlit allows only one call
# per page - drop the second one instead of patching the example itself.
st.set_page_config = lambda *args, **kwargs: None

st.markdown(
    "<style>[data-testid='stSidebar'] {min-width: 620px; max-width: 620px;}</style>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Simulation")
    # While O3DE is still loading, web_video_server accepts the connection but
    # sends no frames - that never fires onerror, so reconnect on a timer until
    # a frame actually arrives (naturalWidth turns non-zero).
    components.html(
        f"""
        <img id="sim" style="width:100%; border-radius:8px">
        <p id="msg" style="font:14px sans-serif; color:#888">Waiting for the simulation...</p>
        <script>
          const img = document.getElementById("sim");
          const msg = document.getElementById("msg");
          const connect = () => img.src = "{STREAM_URL}&t=" + Date.now();
          connect();
          const timer = setInterval(() => {{
              if (img.naturalWidth > 0) {{
                  msg.remove();
                  clearInterval(timer);
              }} else {{
                  connect();
              }}
          }}, 5000);
        </script>
        """,
        height=310,
    )

sys.path.insert(0, RAI_EXAMPLES)
spec = importlib.util.spec_from_file_location("rai_manipulation_demo", DEMO_APP)
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)

demo.main(O3DE_CONFIG)
