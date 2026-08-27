from __future__ import annotations
import importlib, platform, sys

EXPECTED = {"python": "3.9.16", "tensorflow": "2.13.0", "numpy": "1.24.3", "pandas": "2.0.3", "h5py": "3.9.0"}

def version(name):
    try:
        return importlib.import_module(name).__version__
    except Exception:
        return "NOT_INSTALLED"

actual = {"python": platform.python_version(), "tensorflow": version("tensorflow"), "numpy": version("numpy"), "pandas": version("pandas"), "h5py": version("h5py")}
print("Environment verification")
status = "PASS"
for k, exp in EXPECTED.items():
    got = actual[k]
    ok = got == exp
    print(f"{k}: {got} expected {exp} {'PASS' if ok else 'WARN'}")
    if k in {"python", "tensorflow", "numpy", "pandas", "h5py"} and not ok:
        status = "WARN_VERSION_MISMATCH"
print(f"STATUS={status}")
