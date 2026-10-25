# CUDAify — Code-to-CUDA Translator

AI-powered Python/NumPy to CUDA translation using **NVIDIA Nemotron** models served on **Nebius Token Factory**.

## Track
Coding and Agentic Engineering Track — Nebius x NVIDIA Global AI Hackathon

## How it works
1. User pastes Python/NumPy code into the Streamlit frontend
2. FastAPI backend sends the code to Nemotron via Nebius Token Factory REST API
3. Nemotron returns optimized CUDA C++ (kernel + host launcher + explanation)
4. The agentic self-healing pipeline (Translator → Reviewer → Fixer → Verifier)
   iterates to self-correct the generated CUDA
5. Results are saved to SQLite history with verification status, IR analysis,
   and auto-tuning recommendations

## Tech stack
- **Backend:** FastAPI (Python)
- **Frontend:** Streamlit (pure Python)
- **AI:** NVIDIA Nemotron (Nano / Super / Ultra) via Nebius Token Factory
- **Database:** SQLite (with WAL mode, session management)
- **Deployment:** Render / Railway (CPU instances — inference runs on Nebius GPUs)

## Project structure
```
cudaify/
├── .env.example
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── SUBMISSION.md
├── scripts/
│   ├── check_setup.py      # one real call per agent role
│   ├── list_models.py      # Nemotron model IDs on your account
│   └── preflight.py        # pre-submission checklist
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── models.py
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── auth.py
│   │   ├── translate.py
│   │   └── history.py
│   └── services/
│       ├── __init__.py
│       ├── nemotron.py
│       ├── agent_pipeline.py
│       ├── code_review.py
│       ├── cuda_ir.py
│       ├── auto_tune.py
│       ├── semantic_equiv.py
│       ├── advisor.py      # "do you need custom CUDA?" (CuPy/Numba/cuBLAS)
│       ├── report.py       # trust ladder, diff, Markdown report
│       └── grounding.py    # optional Tavily doc grounding
├── frontend/
│   └── streamlit_app.py
└── tests/
    ├── __init__.py
    ├── test_models.py
    ├── test_nemotron.py
    ├── test_cuda_ir.py
    ├── test_code_review.py
    ├── test_auto_tune.py
    ├── test_semantic_equiv.py
    ├── test_pipeline.py
    ├── test_e2e.py
    ├── test_parsing.py
    ├── test_advisor.py
    ├── test_report.py
    ├── test_grounding.py
    ├── test_frontend.py
    └── corpus/
        ├── vec_add.py
        ├── element_wise.py
        ├── array_sum.py
        ├── mat_mul_square.py
        └── stencil_2d.py
```

## What makes CUDAify trustworthy

AI-written GPU kernels often look right and are not (several published speedup claims were later
walked back after correctness-check problems), so CUDAify is built around *how far to trust* a result:

- **Trust summary.** Every result shows a verification ladder (`generated → ai_reviewed → static_passed →
  compiled → runtime_tested → benchmarked`). Only rungs that really happened are ticked; the UI lists the manual
  steps still needed (e.g. compile with `nvcc`, compare against the NumPy reference, benchmark fairly).
- **Review/fix diff.** See exactly what the Reviewer and Fixer changed relative to the Translator's first draft.
- **"Do you actually need custom CUDA?" advisor.** A deterministic, AST-based check that points to CuPy (drop-in
  NumPy), Numba, or vendor libraries (cuBLAS/cuFFT) when they are likely enough, and recommends a custom kernel only
  for patterns where it pays off (e.g. stencils with shared-memory tiling).
- **Docs-grounded review (optional, Tavily).** With `TAVILY_API_KEY` set, the Reviewer receives excerpts from NVIDIA's
  official CUDA documentation relevant to your code pattern, and the UI lists the sources it consulted. Without a key
  the feature is off and nothing else changes.
- **Downloadable trust report (.md)** with status, findings, diff, sources and the remaining steps.

## Related work and inspiration
CUDAify's design follows lessons from existing projects: verifier-first agent loops (Meta's KernelAgent),
skepticism about unverified speedups (KernelBench-Verified), source-to-output side-by-side thinking (Compiler Explorer),
and plan → progress → summary → diff review flows (Amazon Q Developer code transformation).

## Setup

### Prerequisites
- Python 3.10+
- A Nebius Token Factory API key (no local GPU needed)

### Local development

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
# Windows (PowerShell):  .venv\Scripts\Activate.ps1
# macOS/Linux:           source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure
cp .env.example .env          # Windows: copy .env.example .env
# Edit .env: set NEBIUS_API_KEY (and optionally TAVILY_API_KEY)

# 4. Find your model IDs and verify the connection
python scripts/list_models.py
python scripts/check_setup.py

# 5. Run (single process: the app starts its own backend when BACKEND_URL is not set)
streamlit run streamlit_app.py
```

Open `http://localhost:8501`.

**Two-process mode (optional):** run `uvicorn app.main:app --port 8000`, set `BACKEND_URL=http://localhost:8000`,
then start Streamlit.

## Configuration
| Variable | Description | Required |
|----------|-------------|----------|
| `NEBIUS_API_KEY` | Your Nebius Token Factory API key | Yes |
| `NEBIUS_BASE_URL` | API base URL (default: `https://api.tokenfactory.nebius.com/v1`) | No |
| `NEBIUS_MODEL` | Default Nemotron model (default: `nvidia/Nemotron-3.5-Super-VL`) | No |
| `NEMOTRON_TRANSLATOR_MODEL` / `_REVIEWER_` / `_FIXER_` / `_VERIFIER_` | Per-agent model overrides (`NEMOTRON_<ROLE>_MODEL`) | No |
| `NEMOTRON_MAX_TOKENS` | Max tokens per API call (default: 8192; raise for reasoning models) | No |
| `TAVILY_API_KEY` | Enables docs-grounded review (free tier available at tavily.com) | No |
| `CUDAIFY_DB_PATH` | SQLite database path (default: `cudaify.db`) | No |
| `BACKEND_URL` | Set only for two-process mode (default: embedded backend) | No |

## Running tests

```bash
pytest tests/
```
No network or API key needed; the model and Tavily calls are mocked.

## Deployment (one service)
The Streamlit app runs its own backend, so you only need to deploy **one** service.

**Streamlit Community Cloud** (simplest):
1. Push to a public GitHub repo (the MIT `LICENSE` is at the root).
2. On share.streamlit.io choose the repo and set the main file to `streamlit_app.py`.
3. Under *Advanced settings -> Secrets* add:
   ```toml
   NEBIUS_API_KEY = "your-key"
   NEBIUS_MODEL = "nvidia/Nemotron-3.5-Super-VL"
   # optional
   TAVILY_API_KEY = "your-tavily-key"
   ```
4. Deploy. Free-tier storage is ephemeral: accounts and history reset on restart, and the app returns users to the
   login screen when that happens.

**Render / Railway** (alternative): one Web Service with build `pip install -r requirements.txt` and start
`streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0`. Do not set `BACKEND_URL`.

## Demo video script (3 min)
1. Problem: AI-written CUDA often looks right and isn't. How far can you trust it?
2. Paste a loop, click Translate, watch Translator → Reviewer → Fixer → Verifier run on Nemotron / Token Factory
3. Trust summary, the review/fix diff, and the sources the Reviewer consulted (Tavily)
4. The "do you need custom CUDA?" advisor on a vectorized example (it recommends CuPy)
5. Download the trust report; roadmap: compile and test on Nebius GPUs

## License
MIT