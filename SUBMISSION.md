# CUDAify Submission — Nebius x NVIDIA Global AI Hackathon

## Project: CUDAify — Code-to-CUDA Translator

**Track:** Coding and Agentic Engineering Track

## What we built
CUDAify is an agentic self-healing Python/NumPy-to-CUDA translation platform.
It takes Python/NumPy source code and produces reviewed CUDA C++ together with an honest
statement of how far to trust it, using NVIDIA Nemotron models served through Nebius Token Factory.

Unlike one-shot code generators, CUDAify runs a multi-agent pipeline:
Translator → Reviewer → Fixer → Reviewer → … → Verifier.
Each stage is a specialized Nemotron prompt, and the pipeline iterates until
the reviewer reports no blocking issues or the max iteration limit is reached.

## Key technical differentiators
1. **CUDA Intermediate Representation (IR)** — A structured AST-derived IR sits
   between Python and CUDA, enabling independent verification of translations.
2. **Agentic self-healing pipeline** — Multiple Nemotron roles (translator,
   reviewer, fixer, verifier) iterate to self-correct generated CUDA code.
3. **Static code review** — AI-generated findings for bounds guarding,
   synchronization, error checking, shared memory, race conditions, and more.
4. **Auto-tuning recommendations** — Block size, grid size, memory, occupancy,
   and pattern-specific tuning hints.
5. **Semantic equivalence testing** — Structured test corpus with Python
   reference implementations for validating CUDA correctness.
6. **Trust summary + honest status ladder** — Every result shows which verification
   rungs really happened (generated, AI-reviewed, verifier-passed) and which are still
   open (compile, runtime test, benchmark), with the concrete steps to close them.
7. **Review/fix diff and downloadable trust report** — See exactly what the review loop
   changed and export findings, diff, sources and remaining steps as Markdown.
8. **"Do you actually need custom CUDA?" advisor** — Deterministic AST analysis that points
   to CuPy, Numba or cuBLAS/cuFFT when they are likely enough, and recommends a custom
   kernel only for patterns where it pays off.
9. **Docs-grounded review via Tavily** — The Reviewer receives excerpts from NVIDIA's CUDA
   documentation matched to the user's code pattern, and the UI shows the sources consulted.

## How Nebius Token Factory is used
All LLM inference runs through the Nebius Token Factory REST API
(OpenAI-compatible `/chat/completions` endpoint). Nemotron models (translator,
reviewer, fixer, verifier) are resolved per-agent-role from environment
variables. No local GPU is required — inference runs entirely on Nebius GPU
infrastructure.

Key integration: `app/services/nemotron.py` calls
`POST {NEBIUS_BASE_URL}/chat/completions` with Bearer auth using
`NEBIUS_API_KEY`.

## How Tavily is used
When `TAVILY_API_KEY` is set, `app/services/grounding.py` queries the Tavily Search API restricted to
`docs.nvidia.com` and `developer.nvidia.com`, using a query chosen from the detected code pattern (stencil,
reduction, matrix product, or general indexing). The top excerpts are added to the Reviewer's prompt and the
source links are shown to the user and included in the downloadable report. Failures degrade silently to the
normal pipeline.

## Models used
- Default: `nvidia/Nemotron-3.5-Super-VL` (configurable via `NEBIUS_MODEL`)
- Per-agent overrides: `NEMOTRON_TRANSLATOR_MODEL`,
  `NEMOTRON_REVIEWER_MODEL`, `NEMOTRON_FIXER_MODEL`,
  `NEMOTRON_VERIFIER_MODEL`

## How to run
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env        # set NEBIUS_API_KEY (optional: TAVILY_API_KEY)
python scripts/check_setup.py
streamlit run frontend/streamlit_app.py
```
The app starts its own backend, so there is a single process to run and deploy.

## Demo
- Working demo URL: (Streamlit Community Cloud; see README)
- 3-minute video: (YouTube link)

## Repository
Public GitHub repo with MIT License.

## License
MIT

## Feedback on Nebius Token Factory and NVIDIA models (edit with YOUR experience)
Replace the bracketed parts with what you actually observed:
- **Setup:** [time to first successful call; was the base URL / model ID obvious? Model IDs are case-sensitive
  `nvidia/...` names and a wrong guess only appears as an API error.]
- **Reasoning models:** Nemotron replies can come back with empty `content` when the token budget is spent on
  reasoning; we handle it in code, but a clearer finish reason or a documented recommended `max_tokens` would help.
- **Latency/cost in a multi-agent loop:** [measured seconds per agent; which model for which role; credits used.]
- **What worked well:** [e.g. the OpenAI-compatible API needed no SDK changes.]
- **Wishlist:** [e.g. a hosted CUDA compile/run endpoint so agent pipelines can reach compiled / runtime_tested.]
- **Tavily:** [what worked, latency, result quality for NVIDIA docs queries.]
