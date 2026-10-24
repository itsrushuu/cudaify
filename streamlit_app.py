"""CUDAify Streamlit frontend — professional UI/UX.

Clean, production-grade interface with:
- NVIDIA-inspired green/dark theme
- Two-column layout (Python input / CUDA output)
- Agentic pipeline visualization
- Copy-to-clipboard, download, verification badges
- Searchable history, auth, example code loader
"""

import json
from typing import Any, Dict, List, Optional
import os
import sys
import threading
import time

import requests
import streamlit as st

# Make the project root importable (needed on Streamlit Community Cloud).
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from app.services import report, advisor  # noqa: E402

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(
    page_title="CUDAify — Code-to-CUDA Translator",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — professional dark theme
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --primary: #76B900;
        --primary-dark: #5C8A00;
        --bg: #0B0E11;
        --card: #14181C;
        --card-hover: #1E2329;
        --border: #2A3138;
        --text: #F0F2F5;
        --muted: #8B949E;
        --success: #3FB950;
        --warning: #D29922;
        --error: #F85149;
        --info: #58A6FF;
    }

    .stApp {
        background-color: var(--bg);
        color: var(--text);
    }

    /* Hide Streamlit default chrome */
    header[data-testid="stHeader"] { display: none !important; }
    .stDeployButton { display: none !important; }
    .stActionButton { display: none !important; }
    [data-testid="stSidebarNav"] { display: none !important; }

    /* Main header banner */
    .main-header {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
        padding: 1.75rem 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 4px 16px rgba(118, 185, 0, 0.25);
        overflow: hidden;
    }
    .main-header h1 {
        margin: 0;
        font-size: 1.75rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        letter-spacing: -0.02em;
    }
    .main-header p {
        margin: 0.5rem 0 0;
        opacity: 0.9;
        font-size: 0.95rem;
    }

    /* Cards */
    .card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }

    .auth-card {
        max-width: 420px;
        margin: 2rem auto;
        padding: 2.5rem;
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 12px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.5);
    }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: white;
    }
    .badge-generated { background: var(--warning); }
    .badge-ai_reviewed { background: var(--info); }
    .badge-static_passed { background: var(--success); }
    .badge-compiled { background: #A371F7; }
    .badge-runtime_tested { background: #22D3EE; }
    .badge-benchmarked { background: #B453F9; }

    .stage-progress {
        display: flex;
        gap: 5px;
        margin: 8px 0;
    }
    .stage-dot {
        width: 9px;
        height: 9px;
        border-radius: 50%;
    }
    .stage-dot.active { background: var(--primary); box-shadow: 0 0 8px var(--primary); }
    .stage-dot.done { background: var(--success); }
    .stage-dot.pending { background: var(--border); }

    /* Empty state */
    .empty-state {
        text-align: center;
        padding: 3rem 1.5rem;
        color: var(--muted);
    }
    .empty-state h3 { margin: 0 0 0.5rem; color: var(--text); }
    .empty-state p { margin: 0; font-size: 0.95rem; }

    /* Findings */
    .finding {
        border-left: 3px solid;
        padding: 8px 12px;
        margin: 6px 0;
        background: rgba(255,255,255,0.03);
        border-radius: 0 6px 6px 0;
        font-size: 0.9rem;
    }
    .finding-error { border-color: var(--error); }
    .finding-warning { border-color: var(--warning); }
    .finding-info { border-color: var(--info); }

    /* Buttons */
    .stButton button[kind="primary"] {
        background: var(--primary);
        border: none;
        color: white;
        font-weight: 600;
        border-radius: 6px;
        padding: 0.6rem 1.25rem;
    }
    .stButton button[kind="primary"]:hover {
        background: var(--primary-dark);
    }

    /* Form inputs */
    .stTextInput input, .stTextArea textarea {
        background: #0D1117 !important;
        border: 1px solid var(--border) !important;
        color: var(--text) !important;
        border-radius: 6px !important;
    }
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 3px rgba(118, 185, 0, 0.15) !important;
    }

    .stSelectbox > div > div {
        background: #0D1117 !important;
        border: 1px solid var(--border) !important;
        color: var(--text) !important;
        border-radius: 6px !important;
    }

    .stTextArea textarea {
        font-family: "Consolas", "Monaco", "Courier New", monospace;
        font-size: 0.9rem;
        color: var(--text) !important;
    }
    .stTextArea textarea::placeholder {
        color: #6E7681 !important;
        opacity: 1 !important;
    }

    .stMetric {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1rem;
    }

    .toast {
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 12px 20px;
        border-radius: 8px;
        color: white;
        font-weight: 600;
        z-index: 9999;
        animation: slideIn 0.3s ease;
    }
    @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #0D1117 !important;
        border-right: 1px solid var(--border) !important;
    }

    h1, h2, h3 {
        color: var(--text) !important;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
if "token" not in st.session_state:
    st.session_state.token = None
if "user" not in st.session_state:
    st.session_state.user = None
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "review_result" not in st.session_state:
    st.session_state.review_result = None
if "toast" not in st.session_state:
    st.session_state.toast = None
if "source_code" not in st.session_state:
    st.session_state.source_code = ""
if "history_search" not in st.session_state:
    st.session_state.history_search = ""
if "last_source" not in st.session_state:
    st.session_state.last_source = ""


def _toast(msg: str, kind: str = "info"):
    st.session_state.toast = {"msg": msg, "kind": kind}


def _render_toast():
    t = st.session_state.toast
    if not t:
        return
    color = {"success": "#3FB950", "error": "#F85149", "info": "#58A6FF"}.get(t["kind"], "#58A6FF")
    st.markdown(
        f'<div class="toast" style="background:{color};">{t["msg"]}</div>',
        unsafe_allow_html=True,
    )
    st.session_state.toast = None


def _auth_headers():
    if st.session_state.token:
        return {"Authorization": f"Bearer {st.session_state.token}"}
    return {}


# ---------------------------------------------------------------------------
# Verification status badge + stage progress
# ---------------------------------------------------------------------------
STATUS_ORDER = ["generated", "ai_reviewed", "static_passed", "compiled", "runtime_tested", "benchmarked"]
STATUS_LABELS = {
    "generated": "Generated (AI, unverified)",
    "ai_reviewed": "AI Reviewed (not compiled)",
    "static_passed": "Static Passed (not executed)",
    "compiled": "Compiled (nvcc)",
    "runtime_tested": "Runtime Tested",
    "benchmarked": "Benchmarked",
}
STATUS_COLORS = {
    "generated": "#D29922",
    "ai_reviewed": "#58A6FF",
    "static_passed": "#3FB950",
    "compiled": "#A371F7",
    "runtime_tested": "#22D3EE",
    "benchmarked": "#B453F9",
}


def _verification_badge(status: str):
    color = STATUS_COLORS.get(status, "#8B949E")
    label = STATUS_LABELS.get(status, status)
    return f'<span class="badge badge-{status}">{label}</span>'


def _stage_progress(status: str):
    idx = STATUS_ORDER.index(status) if status in STATUS_ORDER else 0
    dots = ""
    for i, s in enumerate(STATUS_ORDER):
        cls = "done" if i < idx else ("active" if i == idx else "pending")
        dots += f'<span class="stage-dot {cls}" title="{STATUS_LABELS[s]}"></span>'
    return f'<div class="stage-progress">{dots}</div>'


# ---------------------------------------------------------------------------
# Copy-to-clipboard
# ---------------------------------------------------------------------------
def _copy_button(text: str, key: str, label: str = "Copy"):
    if st.button(label, key=f"copy_{key}", help="Copy to clipboard"):
        st.session_state["_clipboard_" + key] = text
        _toast("Copied to clipboard!", "success")


# ---------------------------------------------------------------------------
# Backend bootstrap (single-process embedded mode)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _ensure_backend() -> str:
    if os.getenv("BACKEND_URL"):
        return "external"
    import uvicorn
    from app.main import app as api_app

    import asyncio

    def _run_server():
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            server = uvicorn.Server(
                uvicorn.Config(api_app, host="127.0.0.1", port=8000, log_level="warning")
            )
            loop.run_until_complete(server.serve())
        except Exception:
            pass

    threading.Thread(target=_run_server, daemon=True).start()
    for _ in range(100):
        try:
            requests.get("http://127.0.0.1:8000/health", timeout=0.5)
            return "embedded"
        except requests.exceptions.RequestException:
            time.sleep(0.1)
    return "failed"


_BACKEND_MODE = _ensure_backend()
if _BACKEND_MODE == "failed":
    st.error("The CUDAify backend failed to start. Check the server logs.")
    st.stop()

_render_toast()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="main-header">
        <h1>⚡ CUDAify</h1>
        <p>Agentic self-healing Python/NumPy → CUDA translation using
        <strong>NVIDIA Nemotron</strong> on <strong>Nebius Token Factory</strong></p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Auth UI
# ---------------------------------------------------------------------------
def auth_ui():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="auth-card">', unsafe_allow_html=True)
        st.markdown("### Welcome to CUDAify")
        tab1, tab2 = st.tabs(["🔐 Login", "📝 Sign up"])

        with tab1:
            with st.form("login_form", clear_on_submit=False):
                u = st.text_input("Username", key="login_user")
                p = st.text_input("Password", type="password", key="login_pass")
                submitted = st.form_submit_button("Login", use_container_width=True)
                if submitted:
                    if not u or not p:
                        st.warning("Enter username and password.")
                    else:
                        try:
                            r = requests.post(
                                f"{BACKEND_URL}/auth/login",
                                data={"username": u, "password": p},
                                timeout=10,
                            )
                            if r.ok:
                                data = r.json()
                                st.session_state.token = data["token"]
                                st.session_state.user = {
                                    "user_id": data["user_id"],
                                    "username": data["username"],
                                }
                                _toast("Logged in successfully.", "success")
                                st.rerun()
                            else:
                                st.error(r.json().get("detail", "Login failed"))
                        except requests.exceptions.RequestException as exc:
                            st.error(f"Cannot reach backend: {exc}")

        with tab2:
            with st.form("signup_form", clear_on_submit=False):
                u = st.text_input("New username", key="signup_user")
                p = st.text_input("New password", type="password", key="signup_pass")
                submitted = st.form_submit_button("Create account", use_container_width=True)
                if submitted:
                    if not u or not u.strip():
                        st.warning("Username cannot be empty.")
                    elif not p or len(p) < 6:
                        st.warning("Password must be at least 6 characters.")
                    else:
                        try:
                            r = requests.post(
                                f"{BACKEND_URL}/auth/signup",
                                data={"username": u.strip(), "password": p},
                                timeout=10,
                            )
                            if r.ok:
                                st.success("Account created. Please log in.")
                            else:
                                st.error(r.json().get("detail", "Signup failed"))
                        except requests.exceptions.RequestException as exc:
                            st.error(f"Cannot reach backend: {exc}")
        st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Pipeline visualization
# ---------------------------------------------------------------------------
AGENT_LABELS = {
    "translator": "🔤 Translator",
    "reviewer": "🔍 Reviewer",
    "fixer": "🔧 Fixer",
    "verifier": "✅ Verifier",
}


def _render_pipeline(pipeline: Dict[str, Any]):
    if not pipeline:
        return

    agent_results = pipeline.get("agent_results", [])
    iterations = pipeline.get("iterations", 0)
    converged = pipeline.get("converged", False)
    final_status = pipeline.get("final_status", "generated")

    st.markdown("### 🔄 Agentic Pipeline Trace")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Iterations", iterations)
    with c2:
        st.metric("Final Status", STATUS_LABELS.get(final_status, final_status))
    with c3:
        st.metric("Converged", "Yes" if converged else "No")

    flow = '<div style="display:flex; align-items:center; gap:6px; margin:10px 0; flex-wrap:wrap;">'
    for i, ar in enumerate(agent_results):
        agent = ar.get("agent", "unknown")
        success = ar.get("success", False)
        label = AGENT_LABELS.get(agent, agent)
        color = "#3FB950" if success else "#F85149"
        flow += f'''
        <div style="text-align:center;">
            <div style="width:36px;height:36px;background:{color};border-radius:50%;
                display:flex;align-items:center;justify-content:center;margin:0 auto;
                color:white;font-weight:700;font-size:11px;">{i+1}</div>
            <div style="font-size:10px;margin-top:4px;max-width:70px;line-height:1.2;">{label}</div>
        </div>
        '''
        if i < len(agent_results) - 1:
            flow += '<div style="font-size:14px;color:#8B949E;">▼</div>'
    flow += "</div>"
    st.markdown(flow, unsafe_allow_html=True)

    for i, ar in enumerate(agent_results):
        agent = ar.get("agent", "unknown")
        success = ar.get("success", False)
        model = ar.get("model", "?")
        iteration = ar.get("iteration", 0)
        error = ar.get("error")
        sections = ar.get("sections", {})

        exp_label = f"{AGENT_LABELS.get(agent, agent)} · Iter {iteration} · {'✅' if success else '❌'} · {model}"
        with st.expander(exp_label, expanded=(i == len(agent_results) - 1 and success)):
            if error:
                st.error(f"Agent error: {error}")
            else:
                shown_any = False
                if sections.get("CUDA KERNEL"):
                    st.markdown("**CUDA Kernel**")
                    st.code(sections["CUDA KERNEL"], language="cpp")
                    _copy_button(sections["CUDA KERNEL"], f"pipe_kernel_{i}")
                    shown_any = True
                if sections.get("HOST CODE"):
                    st.markdown("**Host Code**")
                    st.code(sections["HOST CODE"], language="cpp")
                    _copy_button(sections["HOST CODE"], f"pipe_host_{i}")
                    shown_any = True
                if sections.get("FINDINGS"):
                    st.markdown("**Review Findings**")
                    st.write(sections["FINDINGS"])
                    shown_any = True
                if sections.get("BLOCKING"):
                    st.markdown(f"**Blocking:** {sections['BLOCKING']}")
                    shown_any = True
                if sections.get("VERDICT"):
                    st.markdown(f"**Verdict:** {sections['VERDICT']}")
                    shown_any = True
                if sections.get("CONFIDENCE"):
                    st.markdown(f"**Confidence:** {sections['CONFIDENCE']}")
                    shown_any = True
                if sections.get("NOTES"):
                    st.markdown(f"**Notes:** {sections['NOTES']}")
                    shown_any = True
                if sections.get("CHANGES MADE"):
                    st.markdown(f"**Changes Made:** {sections['CHANGES MADE']}")
                    shown_any = True
                if not shown_any and ar.get("content"):
                    st.markdown(ar["content"])


# ---------------------------------------------------------------------------
# Example code snippets
# ---------------------------------------------------------------------------
EXAMPLES = {
    "vec_add": ("Vector Addition", "def vec_add(a, b, c, n):\n    for i in range(n):\n        c[i] = a[i] + b[i]\n"),
    "array_sum": ("Array Sum (Reduction)", "def array_sum(a, n):\n    total = 0.0\n    for i in range(n):\n        total += a[i]\n    return total\n"),
    "stencil_2d": ("2D Stencil", "def stencil_2d(input_arr, output, rows, cols):\n    for i in range(1, rows - 1):\n        for j in range(1, cols - 1):\n            output[i * cols + j] = (\n                input_arr[(i - 1) * cols + j]\n                + input_arr[(i + 1) * cols + j]\n                + input_arr[i * cols + (j - 1)]\n                + input_arr[i * cols + (j + 1)]\n            ) / 4.0\n    return output\n"),
    "mat_mul": ("Matrix Multiply", "def mat_mul(A, B, C, n):\n    for i in range(n):\n        for j in range(n):\n            total = 0.0\n            for k in range(n):\n                total += A[i * n + k] * B[k * n + j]\n            C[i * n + j] = total\n"),
}


def _load_example(key: str):
    _, code = EXAMPLES[key]
    st.session_state.source_code = code


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
def main_app():
    user = st.session_state.user

    # ---------- Sidebar ----------
    with st.sidebar:
        st.markdown(f"**👤 {user['username']}**")
        if st.button("Logout", use_container_width=True):
            try:
                requests.post(
                    f"{BACKEND_URL}/auth/logout",
                    headers=_auth_headers(),
                    timeout=10,
                )
            except requests.exceptions.RequestException:
                pass
            st.session_state.token = None
            st.session_state.user = None
            st.session_state.last_result = None
            st.session_state.review_result = None
            st.session_state.source_code = ""
            st.rerun()

        st.divider()

        st.markdown("### ⚙️ Settings")
        use_pipeline = st.checkbox(
            "Use agentic self-healing pipeline",
            value=True,
            help="Runs Translator → Reviewer → Fixer → Verifier loop for correctness.",
        )
        model_options = [
            "",
            "nvidia/Nemotron-3.5-Super-VL",
            "nvidia/nemotron-3-super-120b-a12b",
            "nvidia/Nemotron-3-Ultra-550b-a55b",
            "nvidia/Nemotron-3_5-Lightning",
            "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
        ]
        model = st.selectbox(
            "Nemotron model (optional)",
            options=model_options,
            index=0,
            format_func=lambda m: m or "Default (per-agent models from .env)",
            help="Leave at default to use per-agent models from .env.",
        )

        st.divider()
        st.markdown("### 📊 Verification Stages")
        st.caption(
            "generated → ai_reviewed → static_passed → compiled → "
            "runtime_tested → benchmarked"
        )
        st.divider()
        st.markdown("### 🔄 Pipeline Mode")
        st.caption("Agentic self-healing: Translate → Review → Fix → Verify")

    # ---------- Main content ----------
    col1, col2 = st.columns([1, 1], gap="large")

    # ---- Left: Input ----
    with col1:
        st.markdown("### 🐍 Python / NumPy Code")

        source = st.text_area(
            "Code",
            value=st.session_state.source_code,
            height=420,
            placeholder="def vec_add(a, b, c, n):\n    for i in range(n):\n        c[i] = a[i] + b[i]\n",
            label_visibility="collapsed",
        )
        st.session_state.source_code = source

        st.markdown("**Quick examples:**")
        ex_cols = st.columns(2)
        for i, (key, (name, _)) in enumerate(EXAMPLES.items()):
            with ex_cols[i % 2]:
                if st.button(f"📄 {name}", key=f"ex_{key}", use_container_width=True):
                    _load_example(key)
                    st.rerun()

        col_translate, col_clear = st.columns([3, 1])
        with col_translate:
            translate_clicked = st.button(
                "🚀 Translate to CUDA",
                type="primary",
                use_container_width=True,
            )
        with col_clear:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.source_code = ""
                st.session_state.last_result = None
                st.session_state.review_result = None
                st.rerun()

        if translate_clicked:
            if not source or not source.strip():
                st.warning("Enter some code first.")
            else:
                with st.spinner("Running agentic pipeline..."):
                    try:
                        r = requests.post(
                            f"{BACKEND_URL}/translate",
                            data={
                                "source_code": source.strip(),
                                "model": model if model else "",
                                "use_pipeline": "true" if use_pipeline else "false",
                            },
                            headers=_auth_headers(),
                            timeout=180,
                        )
                        if r.ok:
                            st.session_state.last_result = r.json()
                            st.session_state.last_source = source.strip()
                            st.session_state.review_result = None
                            _toast("Translation complete.", "success")
                            st.rerun()
                        else:
                            st.error(r.json().get("detail", "Translation failed"))
                    except requests.exceptions.RequestException as exc:
                        st.error(f"Request failed: {exc}")

    # ---- Right: Output ----
    with col2:
        st.markdown("### 🔧 CUDA Output")

        if st.session_state.last_result:
            res = st.session_state.last_result
            sections = res.get("sections", {})

            status = res.get("verification_status", "generated")
            st.markdown(
                f"{_verification_badge(status)}  {_stage_progress(status)}",
                unsafe_allow_html=True,
            )
            # Clear disclaimer: AI review is NOT compilation or execution
            if status in ("generated", "ai_reviewed", "static_passed"):
                st.error(
                    "⚠️ This code has NOT been compiled or executed. "
                    f"Status '{STATUS_LABELS.get(status, status)}' means "
                    "AI-assisted review only. Verify with nvcc before production use."
                )

            ir_summary = res.get("ir_summary", "")
            if ir_summary:
                st.caption(f"🔍 {ir_summary}")

            # Pipeline trace
            if res.get("pipeline"):
                _render_pipeline(res["pipeline"])

            # Structured sections
            if sections:
                with st.expander("🔧 CUDA Kernel", expanded=True):
                    kernel = sections.get("CUDA KERNEL", "")
                    if kernel:
                        st.code(kernel, language="cpp")
                        _copy_button(kernel, f"res_kernel_{res['id']}")
                    else:
                        st.info("No kernel section returned.")

                with st.expander("🖥️ Host Code", expanded=False):
                    host = sections.get("HOST CODE", "")
                    if host:
                        st.code(host, language="cpp")
                        _copy_button(host, f"res_host_{res['id']}")
                    else:
                        st.info("No host code section returned.")

                with st.expander("📝 Explanation", expanded=False):
                    st.write(sections.get("EXPLANATION", "No explanation provided."))

                with st.expander("⚠️ Correctness Risks", expanded=False):
                    st.write(sections.get("CORRECTNESS RISKS", "No risks identified."))

                with st.expander("🔍 Input Assumptions", expanded=False):
                    st.write(sections.get("INPUT ASSUMPTIONS", "None stated."))

                with st.expander("📊 Data Types", expanded=False):
                    st.write(sections.get("DATA TYPES", "Not specified."))

                with st.expander("🚧 Boundary Conditions", expanded=False):
                    st.write(sections.get("BOUNDARY CONDITIONS", "Not stated."))

                with st.expander("💾 Memory Considerations", expanded=False):
                    st.write(sections.get("MEMORY CONSIDERATIONS", "Not stated."))

                with st.expander("🚀 Optimization Recommendations", expanded=False):
                    st.write(sections.get("OPTIMIZATION RECOMMENDATIONS", "None."))

                with st.expander("⚠️ Limitations", expanded=False):
                    st.write(sections.get("LIMITATIONS", "None stated."))

            # Report: trust ladder, diff, advisor
            st.divider()
            last_source = st.session_state.get("last_source", "")
            advice_obj = advisor.analyze(last_source) if last_source else None
            advice = advice_obj.to_dict() if advice_obj else None
            report_md = report.build_report_markdown(last_source, res, advice)
            st.markdown(report_md)

            # Diff: translator vs fixer output
            initial_kernel_code = report.initial_kernel(res.get("pipeline") or {})
            if initial_kernel_code:
                diff = report.unified_diff(initial_kernel_code, res.get("result", ""))
                if diff:
                    with st.expander("🔀 What the review/fix loop changed", expanded=False):
                        st.code(diff, language="diff")

            # Advisor: do you need custom CUDA?
            if last_source and advice_obj:
                icon = {"recommended": "✅", "consider-alternatives": "⚖️", "likely-unnecessary": "💡"}.get(
                    advice_obj.cuda_verdict, "⚖️"
                )
                with st.expander(f"{icon} Do you actually need custom CUDA?", expanded=False):
                    st.write(advice_obj.summary)
                    for o in advice.get("options", []):
                        st.markdown(f"**{o['name']}** _{o['fit']}_")
                        for why in o.get("why", []):
                            st.markdown(f"- {why}")
                        if o.get("snippet"):
                            st.code(o["snippet"], language="python")
                        if o.get("note"):
                            st.caption(o["note"])

            st.download_button(
                "💾 Download full output (.txt)",
                data=res["result"],
                file_name=f"translation_{res['id']}.txt",
                mime="text/plain",
                use_container_width=True,
            )

            st.divider()
            if st.button("🔍 Run Static Code Review", use_container_width=True):
                with st.spinner("Reviewing..."):
                    try:
                        rr = requests.post(
                            f"{BACKEND_URL}/translate/{res['id']}/review",
                            headers=_auth_headers(),
                            timeout=30,
                        )
                        if rr.ok:
                            st.session_state.review_result = rr.json()
                            _toast("Review complete.", "success")
                            st.rerun()
                        else:
                            st.error(rr.json().get("detail", "Review failed"))
                    except requests.exceptions.RequestException as exc:
                        st.error(f"Review request failed: {exc}")
        else:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(
                '<div class="empty-state">'
                '<h3>No translation yet</h3>'
                '<p>Paste Python/NumPy code on the left and click <b>Translate to CUDA</b>.</p>'
                '</div>'
                '</div>',
                unsafe_allow_html=True,
            )
            # Example quick-load buttons in empty state
            st.markdown("**Try an example:**")
            ex_cols = st.columns(2)
            for i, (key, (name, _)) in enumerate(EXAMPLES.items()):
                with ex_cols[i % 2]:
                    if st.button(f"📄 {name}", key=f"ex_empty_{key}", use_container_width=True):
                        _load_example(key)
                        st.rerun()

    # ---- Review findings ----
    if st.session_state.review_result:
        st.divider()
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🔍 Code Review Findings")
        review = st.session_state.review_result
        review_data = review.get("review", {})
        st.markdown(
            f"{_verification_badge(review.get('verification_status', 'ai_reviewed'))}",
            unsafe_allow_html=True,
        )
        st.write(review_data.get("summary", "No summary available."))
        st.caption(review_data.get("note", ""))

        findings = review_data.get("findings", [])
        if findings:
            for f in findings:
                sev = f.get("severity", "info")
                color = {"error": "#F85149", "warning": "#D29922", "info": "#58A6FF"}.get(sev, "#8B949E")
                st.markdown(
                    f"""
                    <div class="finding finding-{sev}">
                        <strong style="color:{color};">{sev.upper()}</strong>
                        [{f.get('category','general')}] {f.get('message','')}
                        {f'<br><small>Block: {f.get("block","")}, Line: {f.get("line","")}</small>' if f.get('line') else ""}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.success("✅ No issues found by static review.")
        st.markdown("</div>", unsafe_allow_html=True)

    # ---- History ----
    st.divider()
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📜 Translation History")

    search_col, clear_col = st.columns([4, 1])
    with search_col:
        st.session_state.history_search = st.text_input(
            "Search history",
            value=st.session_state.history_search,
            placeholder="Filter by model, status, or ID…",
            label_visibility="collapsed",
        )
    with clear_col:
        if st.button("✕", help="Clear search"):
            st.session_state.history_search = ""
            st.rerun()

    try:
        hist_resp = requests.get(
            f"{BACKEND_URL}/history",
            headers=_auth_headers(),
            timeout=10,
        )
        if hist_resp.status_code == 401:
            st.session_state.token = None
            st.session_state.user = None
            st.session_state.last_result = None
            st.warning("Your session expired. Please log in again.")
            st.rerun()
        hist_resp.raise_for_status()
        hist = hist_resp.json()
        if not isinstance(hist, list):
            hist = []

        q = st.session_state.history_search.strip().lower()
        if q:
            hist = [
                h for h in hist
                if q in str(h.get("id", "")).lower()
                or q in str(h.get("model_used", "")).lower()
                or q in str(h.get("verification_status", "")).lower()
            ]

        if not hist:
            st.info("No translations found." if not q else "No matching translations.")
        else:
            for item in hist[:10]:
                with st.expander(
                    f"Translation #{item['id']} · {item.get('model_used', '?')} · "
                    f"{STATUS_LABELS.get(item.get('verification_status','generated'), item.get('verification_status','generated'))}"
                ):
                    st.markdown(
                        f"{_verification_badge(item.get('verification_status', 'generated'))}",
                        unsafe_allow_html=True,
                    )
                    sections = item.get("sections", {})
                    if sections and sections.get("CUDA KERNEL"):
                        st.code(sections["CUDA KERNEL"], language="cpp")
                        _copy_button(sections["CUDA KERNEL"], f"hist_{item['id']}")
                    else:
                        st.code(item.get("cuda_code", ""), language="cpp")
                    st.caption(
                        f"Status: {item.get('verification_status', 'generated')} | "
                        f"Iterations: {item.get('pipeline_iterations', 0)} | "
                        f"Model: {item.get('model_used', '?')}"
                    )
    except requests.exceptions.RequestException as exc:
        st.error(f"Failed to load history: {exc}")
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------- Disclaimer ----------
    st.divider()
    st.markdown(
        '<div class="card" style="border-color: var(--border); background: rgba(210, 153, 34, 0.08);">'
        '⚠️ <strong>Disclaimer:</strong> Generated CUDA code is AI-assisted and has NOT been '
        'compiled or executed. Verify correctness, run static analysis, and benchmark before '
        'production use.'
        '</div>',
        unsafe_allow_html=True,
    )


if not st.session_state.user:
    auth_ui()
else:
    main_app()
