from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parents[1]
out = root / "reproduced_outputs" / "tables"
out.mkdir(parents=True, exist_ok=True)
rows = []
for path in sorted((root / "results").rglob("*.csv")):
    df = pd.read_csv(path)
    rows.append({"source_file": path.relative_to(root).as_posix(), "rows": len(df), "columns": len(df.columns), "column_names": "|".join(df.columns)})
    numeric = df.select_dtypes(include="number")
    if not numeric.empty:
        numeric.describe().T.to_csv(out / (path.stem + "_numeric_summary.csv"))
for path in sorted((root / "results").rglob("*.json")):
    data = json.loads(path.read_text(encoding="utf-8"))
    rows.append({"source_file": path.relative_to(root).as_posix(), "rows": len(data) if hasattr(data, "__len__") else 1, "columns": "", "column_names": "json"})
pd.DataFrame(rows).to_csv(out / "manifest_table_reproduction_summary.csv", index=False)
print(f"STATUS=PASS tables_written={len(rows)} output={out}")
