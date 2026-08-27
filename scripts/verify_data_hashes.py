from __future__ import annotations
import hashlib
from pathlib import Path

EXPECTED = {
    "data/who/WHO31_external_daily_new_cases.csv": "f9f6af1c1217a118b14ac76eddf5bae4080e461eba43ad350d0bb98f2695eeee",
    "data/electricity/Electricity37_daily_load.csv": "4873492cb3855b713f236a79dea581dc3be320cd1613187eacddffd4bb179c70",
    "data/dengue/Dengue7_daily_notifications.csv": "24f09a7ac8f32f9eb5213d809bf1b30c1e0621aaac183337b1c86dc80f5a4a1b",
}
def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
status = "PASS"
root = Path(__file__).resolve().parents[1]
for rel, expected in EXPECTED.items():
    got = sha(root / rel)
    ok = got == expected
    print(f"{rel}: {got} {'PASS' if ok else 'FAIL'}")
    if not ok:
        status = "FAIL"
print(f"STATUS={status}")
raise SystemExit(0 if status == "PASS" else 1)
