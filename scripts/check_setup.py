"""Check that the CUDAify environment is set up correctly."""

import os
import sys
import importlib


def check_python_version():
    print(f"Python version: {sys.version}")
    return True


def check_dependencies():
    required = ["fastapi", "uvicorn", "python-dotenv", "requests", "streamlit", "pytest"]
    missing = []
    for dep in required:
        try:
            importlib.import_module(dep)
            print(f"  OK: {dep}")
        except ImportError:
            missing.append(dep)
            print(f"  MISSING: {dep}")
    if missing:
        print(f"\nInstall missing deps: pip install {' '.join(missing)}")
        return False
    return True


def check_env_file():
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    if os.path.exists(env_path):
        print(f"  OK: .env exists at {env_path}")
        with open(env_path) as f:
            content = f.read()
        if "NEBIUS_API_KEY=" in content and "your_nebius" not in content:
            print("  OK: NEBIUS_API_KEY appears configured")
            return True
        else:
            print("  WARN: NEBIUS_API_KEY not configured in .env")
            return False
    else:
        print(f"  MISSING: .env not found (copy .env.example to .env)")
        return False


def check_project_structure():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    expected = [
        "app/main.py",
        "app/models.py",
        "app/services/nemotron.py",
        "app/services/agent_pipeline.py",
        "app/services/code_review.py",
        "app/services/cuda_ir.py",
        "app/services/auto_tune.py",
        "app/services/semantic_equiv.py",
        "app/routes/auth.py",
        "app/routes/translate.py",
        "app/routes/history.py",
        "streamlit_app.py",
        "requirements.txt",
        "README.md",
        "LICENSE",
        ".env.example",
        "tests/test_models.py",
    ]
    all_ok = True
    for rel in expected:
        path = os.path.join(root, rel)
        if os.path.exists(path):
            print(f"  OK: {rel}")
        else:
            print(f"  MISSING: {rel}")
            all_ok = False
    return all_ok


if __name__ == "__main__":
    print("=== CUDAify Setup Check ===\n")
    results = []
    results.append(("Python version", check_python_version()))
    print()
    results.append(("Dependencies", check_dependencies()))
    print()
    results.append(("Project structure", check_project_structure()))
    print()
    results.append((".env file", check_env_file()))
    print()
    all_pass = all(results)
    if all_pass:
        print("=== All checks passed ===")
        sys.exit(0)
    else:
        print("=== Some checks failed ===")
        sys.exit(1)