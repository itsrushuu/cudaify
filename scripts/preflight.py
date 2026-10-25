"""Pre-submission checklist for CUDAify.  Run:  python scripts/preflight.py [--tests] [--live]

  --tests   also run the test suite
  --live    also make one real Token Factory call (uses your key / credits)

Exit code 0 = ready to submit, 1 = something must be fixed. Warnings do not fail the run.
"""
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

fails, warns = [], []


def ok(msg):
    print(f"  OK    {msg}")


def fail(msg):
    fails.append(msg)
    print(f"  FAIL  {msg}")


def warn(msg):
    warns.append(msg)
    print(f"  WARN  {msg}")


def read(path):
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            return f.read()
    except OSError:
        return ""


def git(*args):
    try:
        r = subprocess.run(["git", *args], capture_output=True, text=True, cwd=ROOT, timeout=30)
        return r.returncode, r.stdout
    except (OSError, subprocess.TimeoutExpired):
        return 1, ""


print("Files")
for f in ("LICENSE", "README.md", "requirements.txt", ".env.example", "streamlit_app.py", "SUBMISSION.md"):
    ok(f) if os.path.exists(f) else fail(f"missing {f}")
lic = read("LICENSE")
ok("MIT license text present") if "MIT License" in lic and "Permission is hereby granted" in lic else fail("LICENSE is not the MIT text")
req = read("requirements.txt").lower()
for pkg in ("fastapi", "uvicorn", "python-multipart", "streamlit", "requests", "python-dotenv"):
    ok(f"requirements lists {pkg}") if pkg in req else fail(f"requirements.txt is missing {pkg} (clean installs will break)")

print("Secrets and repo hygiene")
code, tracked = git("ls-files")
if code != 0:
    warn("not a git repository (skipped tracked-file checks)")
else:
    files = tracked.split("\n")
    for bad in (".env",):
        ok(f"{bad} is not tracked") if bad not in files else fail(f"{bad} is tracked by git. Remove it and rotate your key")
    risky = [f for f in files if f.endswith(".db") or f.startswith(".venv/") or "__pycache__" in f]
    ok("no database / venv / caches tracked") if not risky else fail(f"tracked junk: {risky[:3]}")
    if "SESSION_PROGRESS.md" in files:
        warn("SESSION_PROGRESS.md is tracked (agent scratch file). Run: git rm SESSION_PROGRESS.md")
    pat = re.compile(r"tvly-[A-Za-z0-9]{10,}|sk-[A-Za-z0-9]{16,}|eyJ[A-Za-z0-9_-]{30,}|(?:API_KEY|TOKEN)\s*=\s*['\"]?[A-Za-z0-9_\-]{30,}")
    hits = []
    for f in files:
        if f and os.path.isfile(f) and not f.endswith((".png", ".jpg", ".zip")):
            for m in pat.finditer(read(f)):
                if not re.search(r"your|test|example|xxx", m.group(0), re.I):
                    hits.append(f)
                    break
    ok("no key-like strings in tracked files") if not hits else fail(f"key-like strings in: {hits[:3]}")
    _, hist = git("log", "--all", "--name-only", "--pretty=format:")
    ok(".env never committed in history") if not re.search(r"(^|/)\.env$", hist, re.M) else fail(".env exists in git history. Rotate the key")
if ".env" in read(".gitignore"):
    ok(".env is git-ignored")
else:
    fail(".gitignore does not ignore .env")

print("Configuration")
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    warn("python-dotenv not installed (run pip install -r requirements.txt)")
key = os.getenv("NEBIUS_API_KEY", "").strip()
ok("NEBIUS_API_KEY is set") if len(key) > 20 else warn("NEBIUS_API_KEY missing or placeholder-sized (needed to run, not to submit)")
base = os.getenv("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1")
ok("base URL is the Token Factory endpoint") if "tokenfactory.nebius.com" in base else fail(f"NEBIUS_BASE_URL looks wrong: {base}")
ok("Tavily grounding enabled") if os.getenv("TAVILY_API_KEY", "").strip() else warn("TAVILY_API_KEY not set (optional, but needed for the Best Use of Tavily prize demo)")

print("App")
try:
    import app.main  # noqa: F401
    ok("backend imports cleanly")
except Exception as exc:  # noqa: BLE001
    fail(f"backend import failed: {type(exc).__name__}: {exc}")

print("Submission text")
sub = read("SUBMISSION.md")
if re.search(r"YouTube link|\(deploy to", sub):
    warn("SUBMISSION.md still has a placeholder for the demo URL / video link")
if re.search(r"\[(?:e\.g\.|time to|what worked|measured)", sub):
    warn("SUBMISSION.md feedback section still has [bracketed] placeholders. Write your real experience")
else:
    ok("feedback section filled in")
if re.search(r"optimized,? correct", read("README.md") + sub, re.I):
    warn("wording claims 'optimized, correct' CUDA. Prefer 'reviewed' (nothing here compiles or runs the kernel)")

if "--tests" in sys.argv:
    print("Tests")
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=ROOT)
    ok("all tests pass") if r.returncode == 0 else fail("tests failed")

if "--live" in sys.argv:
    print("Live Token Factory call")
    r = subprocess.run([sys.executable, "scripts/check_setup.py"], cwd=ROOT)
    ok("check_setup passed") if r.returncode == 0 else fail("check_setup failed")

print()
print(f"{len(fails)} failure(s), {len(warns)} warning(s)")
sys.exit(1 if fails else 0)
