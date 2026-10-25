# CUDAify — Code-to-CUDA Translator

AI-powered Python/NumPy to CUDA translation with an agentic self-healing pipeline.
Runs on **NVIDIA Nemotron** models via **Nebius Token Factory**, or on a local
**LM Studio** OpenAI-compatible endpoint.

## Track

Coding and Agentic Engineering Track — Nebius x NVIDIA Global AI Hackathon

## Problem

AI-written GPU kernels often look correct but contain subtle bugs. Published
speedup claims are sometimes walked back after correctness-check problems.
CUDAify addresses this with:

- **Agentic self-healing**: Translator → Reviewer → Fixer → Reviewer → … → Verifier loop
- **Honest trust ladder**: Every result shows a verification status (`generated → ai_reviewed → static_passed → compiled → runtime_tested → benchmarked`). Only rungs that truly happened are marked; AI review alone never implies compilation or execution.
- **Review/fix diff**: See exactly what each agent changed.
- **"Do you actually need custom CUDA?" advisor**: Deterministic AST analysis that points to CuPy, Numba, or vendor libraries when a custom kernel is unnecessary.
- **Downloadable trust report**: Status, findings, diff, and next steps in Markdown.

## How It Works

1. User pastes Python/NumPy code into the Streamlit frontend.
2. FastAPI backend sends the code to an AI provider (Nebius Token Factory or local LM Studio).
3. The agentic self-healing pipeline iterates: Translator generates CUDA, Reviewer checks for correctness issues, Fixer applies fixes, Verifier confirms.
4. Results are saved to SQLite history with verification status, IR analysis, and auto-tuning recommendations.
5. The frontend shows CUDA kernel/host code, review findings, pipeline trace, and a disclaimer when code has not been compiled or executed.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI (Python) |
| Frontend | Streamlit (pure Python) |
| AI Provider | NVIDIA Nemotron via Nebius Token Factory, or local LM Studio (OpenAI-compatible) |
| Database | SQLite (WAL mode, session management) |
| Rate Limiting | slowapi (in-memory) |
| Password Hashing | PBKDF2-HMAC-SHA256 (600k iterations) |

## AI Provider Selection

Set `AI_PROVIDER` in `.env` to choose the backend:

- `AI_PROVIDER=nebius` (default) — NVIDIA Nemotron on Nebius Token Factory. Requires `NEBIUS_API_KEY`.
- `AI_PROVIDER=local` — LM Studio OpenAI-compatible endpoint. Requires `LOCAL_BASE_URL` and `LOCAL_MODEL`. Does not require `NEBIUS_API_KEY`.

The provider is chosen at call time; no silent fallback occurs between providers.

## Project Structure

```
cudaify/
├── .env.example
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── SUBMISSION.md
├── scripts/
│   ├── check_setup.py      # environment setup check
│   ├── list_models.py      # Nemotron model IDs on your account
│   └── preflight.py        # pre-submission checklist
├── app/
│   ├── __init__.py
│   ├── main.py             # FastAPI entrypoint
│   ├── models.py           # SQLite database models
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── auth.py         # signup/login/logout/session
│   │   ├── translate.py    # translation + verification endpoints
│   │   └── history.py      # translation history endpoint
│   └── services/
│       ├── __init__.py
│       ├── nemotron.py     # AI provider integration (Nebius + local)
│       ├── agent_pipeline.py  # Translator→Reviewer→Fixer→Verifier loop
│       ├── code_review.py  # static code review
│       ├── cuda_ir.py      # CUDA intermediate representation
│       ├── auto_tune.py    # tuning recommendations
│       ├── semantic_equiv.py  # semantic equivalence tests
│       ├── advisor.py      # "do you need custom CUDA?" advisor
│       ├── report.py       # trust ladder, diff, Markdown report
│       └── grounding.py    # optional Tavily doc grounding
├── streamlit_app.py        # Streamlit frontend
├── tests/
│   ├── test_models.py
│   ├── test_nemotron.py
│   ├── test_cuda_ir.py
│   ├── test_code_review.py
│   ├── test_auto_tune.py
│   ├── test_semantic_equiv.py
│   ├── test_pipeline.py
│   ├── test_e2e.py
│   ├── test_parsing.py
│   ├── test_advisor.py
│   ├── test_report.py
│   ├── test_grounding.py
│   ├── test_frontend.py
│   └── corpus/
│       ├── vec_add.py
│       ├── element_wise.py
│       ├── array_sum.py
│       ├── mat_mul_square.py
│       └── stencil_2d.py
```

## Setup

### Prerequisites

- Python 3.10+
- A Nebius Token Factory API key (for `AI_PROVIDER=nebius`), **or**
- LM Studio running locally with a downloaded model (for `AI_PROVIDER=local`)

### Local Development

```powershell
# 1. Create and activate a virtual environment
python -m venv .venv
# Windows (PowerShell):  .venv\Scripts\Activate.ps1
# macOS/Linux:           source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure
copy .env.example .env      # Windows
# Edit .env: set AI_PROVIDER, NEBIUS_API_KEY (nebius) or LOCAL_MODEL (local)

# 4. Verify setup
python scripts/check_setup.py
python scripts/preflight.py --tests

# 5. Run (single process: the app starts its own backend)
streamlit run streamlit_app.py
```

Open `http://localhost:8501`.

**Two-process mode (optional):** run `uvicorn app.main:app --port 8000`, set `BACKEND_URL=http://localhost:8000`, then start Streamlit.

## Configuration

| Variable | Description | Required |
|----------|-------------|----------|
| `AI_PROVIDER` | `nebius` (default) or `local` | No |
| `NEBIUS_API_KEY` | Your Nebius Token Factory API key | Yes (for nebius) |
| `NEBIUS_BASE_URL` | API base URL (default: `https://api.tokenfactory.nebius.com/v1`) | No |
| `NEBIUS_MODEL` | Default Nemotron model (default: `nvidia/Nemotron-3.5-Super-VL`) | No |
| `NEMOTRON_TRANSLATOR_MODEL` / `_REVIEWER_` / `_FIXER_` / `_VERIFIER_` | Per-agent model overrides | No |
| `NEMOTRON_MAX_TOKENS` | Max tokens per API call (default: 8192) | No |
| `LOCAL_BASE_URL` | Local OpenAI-compatible endpoint (default: `http://localhost:1235/v1`) | Yes (for local) |
| `LOCAL_MODEL` | Local model ID (e.g. `qwen2.5-coder-3b-instruct`) | Yes (for local) |
| `LOCAL_API_KEY` | Local API key placeholder (default: `lm-studio`) | No |
| `TAVILY_API_KEY` | Enables docs-grounded review (optional) | No |
| `CUDAIFY_DB_PATH` | SQLite database path (default: `cudaify.db`) | No |
| `BACKEND_URL` | Set only for two-process mode | No |

## Running Tests

```powershell
pytest tests/
```

No network or API key needed; the model and Tavily calls are mocked.

## Deployment (one service)

The Streamlit app runs its own embedded backend, so you only need to deploy **one** service.

**Streamlit Community Cloud** (simplest):

1. Push to a public GitHub repo (the MIT `LICENSE` is at the root).
2. On share.streamlit.io choose the repo and set the main file to `streamlit_app.py`.
3. Under *Advanced settings -> Secrets* add:
   ```toml
   AI_PROVIDER = "nebius"
   NEBIUS_API_KEY = "your-key"
   NEBIUS_MODEL = "nvidia/Nemotron-3.5-Super-VL"
   # optional
   TAVILY_API_KEY = "your-tavily-key"
   ```
4. Deploy. Free-tier storage is ephemeral: accounts and history reset on restart.

**Render / Railway** (alternative): one Web Service with build `pip install -r requirements.txt` and start `streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0`. Do not set `BACKEND_URL`.

## Honest Limitations

- **No CUDA compilation**: Generated CUDA code is not compiled by nvcc. The `compiled` status cannot be earned yet.
- **No runtime execution**: Generated CUDA is not executed against reference inputs. The `runtime_tested` status cannot be earned yet.
- **No benchmarking**: Performance is not measured. The `benchmarked` status cannot be earned yet.
- **AI review only**: `ai_reviewed` means an AI reviewer checked the code, not that it was compiled or tested.
- **Nebius budget**: The default `AI_PROVIDER=nebius` requires an active Nebius Token Factory account with credits. If credits are exhausted, switch to `AI_PROVIDER=local` with LM Studio.
- **Local model limitations**: Qwen2.5-Coder-3B-Instruct (3B parameters) may not follow the structured output format as reliably as larger Nemotron models.

## Security

- `NEBIUS_API_KEY` and `LOCAL_API_KEY` are never logged or exposed in error messages.
- `.env` is git-ignored; never commit it.
- Passwords are hashed with PBKDF2-HMAC-SHA256 at 600k iterations.
- Sessions are rotated on login (session fixation protection).
- Rate limiting: 3/min signup, 5/min login.
- CORS: fail-closed by default (`CORS_ALLOWED_ORIGINS` empty).

## Demo Walkthrough (3 min)

1. **Problem**: AI-written CUDA often looks right and isn't. How far can you trust it?
2. **Translate**: Paste a loop, click Translate, watch Translator → Reviewer → Fixer → Verifier run.
3. **Trust summary**: The verification ladder and disclaimer show exactly what was (and wasn't) verified.
4. **Review/fix diff**: See exactly what the review loop changed.
5. **Advisor**: The "do you need custom CUDA?" advisor recommends CuPy for simple vectorized examples.
6. **Trust report**: Download findings, diff, and remaining steps as Markdown.

## License

MIT