#!/usr/bin/env python3
from pathlib import Path
import hashlib
import pandas as pd
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
NC4 = ROOT / "posthoc_kbs" / "nc_pr4"
NC7 = ROOT / "posthoc_kbs" / "nc_pr7r"

required = [
    NC4 / "NC_PR4_ALIGNMENT_PANEL_SENSITIVITY_SUMMARY.csv",
    NC4 / "NC_PR4_CROSS_ARCH_SUMMARY.csv",
    NC4 / "NC_PR4_EXECUTION_REPORT.md",
    NC4 / "NC_PR4_FREEZE.json",
    NC7 / "NC_PR7R_HARMONIZED_LEVEL3_EVIDENCE.csv",
]
missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
if missing:
    raise SystemExit("MISSING REQUIRED KBS EXTENSION FILES: " + ", ".join(missing))

freeze = json.loads((NC4 / "NC_PR4_FREEZE.json").read_text(encoding="utf-8"))
assert freeze["canonical_96_reproduced"] is True
assert freeze["panel_72_portfolio_median_signal_preserved"] is True
assert freeze["panel_48_portfolio_median_signal_preserved"] is True
assert freeze["new_model_training_performed"] is False
assert freeze["optimizer_invoked"] is False
assert freeze["model_weights_modified"] is False

sens = pd.read_csv(NC4 / "NC_PR4_ALIGNMENT_PANEL_SENSITIVITY_SUMMARY.csv")
assert set(sens["panel_size"].unique()) == {48,72,96}
assert (sens["CANONICAL_96_REPRODUCED"] == "YES").all()
assert (sens["PANEL_72_PORTFOLIO_MEDIAN_SIGNAL_PRESERVED"] == "YES").all()
assert (sens["PANEL_48_PORTFOLIO_MEDIAN_SIGNAL_PRESERVED"] == "YES").all()

cross = pd.read_csv(NC4 / "NC_PR4_CROSS_ARCH_SUMMARY.csv")
primary = cross[cross["stratum_scope"] == "PRIMARY"].set_index("receiver_architecture")
expected = {
    "GRU": -0.0244219,
    "TRANSFORMER": -0.00476188,
}
for arch, value in expected.items():
    observed = float(primary.loc[arch, "equal_country_median_delta_rho"])
    assert abs(observed - value) < 1e-6, (arch, observed, value)

harm = pd.read_csv(NC7 / "NC_PR7R_HARMONIZED_LEVEL3_EVIDENCE.csv")
expected_context = {
    "Reference LSTM — primary cross-seed": -0.016129,
    "GRU — primary cross-seed": -0.024422,
    "TRANSFORMER — primary cross-seed": -0.004762,
}
for ctx, value in expected_context.items():
    row = harm[harm["receiving_context"] == ctx]
    assert len(row) == 1, ctx
    observed = float(row.iloc[0]["median_delta_rho"])
    assert abs(observed - value) < 1e-6, (ctx, observed, value)

print("RP-KBS1 extension verification: PASS")
print("Canonical 96 reproduced: YES")
print("72/48 portfolio structural signal preserved: YES")
print("GRU primary median delta-rho:", float(primary.loc["GRU","equal_country_median_delta_rho"]))
print("Transformer primary median delta-rho:", float(primary.loc["TRANSFORMER","equal_country_median_delta_rho"]))
print("New training performed: NO")
