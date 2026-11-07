"""Command-line environment check for Module 1.

Usage:  python check_env.py
Exits with code 0 when every required package imports, otherwise 1.
"""
import importlib
import sys

REQUIRED = [
    ("streamlit", "Streamlit"),
    ("pandas", "Pandas"),
    ("transformers", "Hugging Face Transformers"),
    ("torch", "PyTorch"),
    ("altair", "Altair"),
    ("pytest", "Pytest (development only)"),
    ("sqlite3", "sqlite3 (standard library)"),
]


def check_packages():
    """Return a list of (module, label, version_or_None, error_or_None)."""
    results = []
    for module, label in REQUIRED:
        try:
            mod = importlib.import_module(module)
            version = getattr(mod, "__version__", getattr(mod, "sqlite_version", "built-in"))
            results.append((module, label, str(version), None))
        except Exception as exc:  # noqa: BLE001 - report any import failure
            results.append((module, label, None, f"{type(exc).__name__}: {exc}"))
    return results


def main() -> int:
    print(f"Python {sys.version.split()[0]}")
    failed = 0
    for module, label, version, error in check_packages():
        if error:
            failed += 1
            print(f"  [FAIL] {label:<30} {error}")
        else:
            print(f"  [ OK ] {label:<30} {version}")
    if failed:
        print(f"\n{failed} package(s) missing. Run: pip install -r requirements.txt")
        return 1
    print("\nEnvironment looks good.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
