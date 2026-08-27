from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

import numpy as np
import pandas as pd

DATE_COL = "Date_reported"
TARGET_COL = "New_cases"
COUNTRY_COL = "Country"
EPS = 1e-8

NUM_FEATURE_COLS = ["level_log", "bounded_slope", "slope_volatility"]
STATE_FEATURE_COLS = [f"state_phase_{k}" for k in range(9)] + [
    "state_transition_intensity",
    "state_entropy_norm",
    "state_phase_switch_ema",
    "state_phase_age_norm",
    "state_growth_innovation",
    "state_decline_innovation",
    "state_high_volatility_innovation",
    "state_growth_high_volatility_innovation",
]

DIRECTION_NAMES = {0: "decline", 1: "stable", 2: "growth"}
VOL_NAMES = {0: "low_volatility", 1: "moderate_volatility", 2: "high_volatility"}
CANONICAL_COUNTRY = {"USA": "United States"}


@dataclass(frozen=True)
class PY1Config:
    # Controlled thesis / legacy-workflow parameters
    expected_rows_per_country: int = 963
    initial_train_fraction: float = 0.445
    initial_outer_train_size: int = 428
    n_windows: int = 10
    stride: int = 56
    validation_size: int = 28
    test_size: int = 28
    lookback: int = 7
    horizon: int = 1

    vol_window: int = 7
    direction_tau: float = 0.05
    vol_low_thr: float = 0.05
    vol_high_thr: float = 0.15
    rho_state: float = 0.85
    rho_switch: float = 0.85
    max_phase_age: int = 14
    eps: float = 1e-8

    amp_quantile: float = 0.75
    amp_score_mode: str = "slope_plus_volatility"

    doc_feature_mode: str = "dynamic_state"

    def validate(self) -> None:
        if self.initial_outer_train_size != math.floor(
            self.initial_train_fraction * self.expected_rows_per_country
        ):
            raise ValueError(
                "Initial outer-training size is inconsistent with the controlled "
                "fraction/trajectory length."
            )
        if self.horizon != 1:
            raise ValueError("PY-1 reference contract is fixed to horizon=1.")
        if self.lookback != 7:
            raise ValueError("PY-1 reference contract is fixed to lookback=7.")
        if self.amp_score_mode != "slope_plus_volatility":
            raise ValueError(
                "PY-1 reference contract is fixed to amplitude mode "
                "'slope_plus_volatility'."
            )


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load the controlled input workbook/CSV without changing source values."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported input format: {suffix}")

    required = [DATE_COL, TARGET_COL, COUNTRY_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")

    df = df[required].copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="raise")
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    if df[TARGET_COL].isna().any():
        raise ValueError("New_cases contains missing/non-numeric values.")
    df[COUNTRY_COL] = df[COUNTRY_COL].astype(str)
    return df


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate_dataset(df: pd.DataFrame, cfg: PY1Config) -> pd.DataFrame:
    """Fail-hard validation of the controlled six-country daily panel."""
    cfg.validate()
    rows: List[dict] = []

    if df.duplicated([COUNTRY_COL, DATE_COL]).any():
        dup = df.loc[df.duplicated([COUNTRY_COL, DATE_COL], keep=False), [COUNTRY_COL, DATE_COL]]
        raise ValueError(f"Duplicate country-date rows found: {dup.head().to_dict('records')}")

    for country, g in df.groupby(COUNTRY_COL, sort=True):
        g = g.sort_values(DATE_COL).reset_index(drop=True)
        if len(g) != cfg.expected_rows_per_country:
            raise ValueError(
                f"{country}: expected {cfg.expected_rows_per_country} rows, got {len(g)}."
            )
        if g[TARGET_COL].isna().any():
            raise ValueError(f"{country}: missing New_cases values detected.")

        date_diff = g[DATE_COL].diff().dropna()
        bad_gap = date_diff != pd.Timedelta(days=1)
        if bad_gap.any():
            idx = int(np.flatnonzero(bad_gap.to_numpy())[0]) + 1
            raise ValueError(
                f"{country}: non-daily date gap around {g.loc[idx - 1, DATE_COL]} -> "
                f"{g.loc[idx, DATE_COL]}."
            )

        rows.append(
            {
                "Country": country,
                "Country_canonical": CANONICAL_COUNTRY.get(country, country),
                "n_rows": len(g),
                "start_date": g[DATE_COL].iloc[0],
                "end_date": g[DATE_COL].iloc[-1],
                "n_missing_target": int(g[TARGET_COL].isna().sum()),
                "n_duplicate_dates": int(g[DATE_COL].duplicated().sum()),
                "daily_contiguous": True,
            }
        )

    if len(rows) != 6:
        raise ValueError(f"Expected 6 country trajectories, found {len(rows)}.")

    return pd.DataFrame(rows)


def generate_window_table(country_df: pd.DataFrame, cfg: PY1Config) -> pd.DataFrame:
    """Exact reproduction of the KNIME Window Table Generator convention."""
    g = country_df.sort_values(DATE_COL).reset_index(drop=True)
    T = len(g)
    if T != cfg.expected_rows_per_country:
        raise ValueError(f"Expected T={cfg.expected_rows_per_country}, got T={T}.")

    rows = []
    for w0 in range(cfg.n_windows):
        train_start = 0
        train_end = cfg.initial_outer_train_size + w0 * cfg.stride - 1
        test_start = train_end + 1
        test_end = test_start + cfg.test_size - 1
        if test_end >= T:
            break
        insample_end = train_end
        insample_start = train_end - cfg.validation_size + 1
        if insample_start < train_start:
            raise ValueError("Training segment too short for validation block.")
        rows.append(
            {
                "window_id": w0 + 1,
                "train_start_row": train_start + 1,
                "train_end_row": train_end + 1,
                "insample_start_row": insample_start + 1,
                "insample_end_row": insample_end + 1,
                "test_start_row": test_start + 1,
                "test_end_row": test_end + 1,
                "train_size": train_end - train_start + 1,
                "insample_size": cfg.validation_size,
                "test_size": cfg.test_size,
                "train_start_date": g.loc[train_start, DATE_COL],
                "train_inner_end_date": g.loc[insample_start - 1, DATE_COL],
                "validation_start_date": g.loc[insample_start, DATE_COL],
                "validation_end_date": g.loc[insample_end, DATE_COL],
                "test_start_date": g.loc[test_start, DATE_COL],
                "test_end_date": g.loc[test_end, DATE_COL],
            }
        )
    out = pd.DataFrame(rows)
    if len(out) != cfg.n_windows:
        raise ValueError(f"Expected {cfg.n_windows} valid windows, generated {len(out)}.")
    return out


def _slice_current_window(country_df: pd.DataFrame, window_row: Mapping[str, object]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Exact 1-based row slicing used by the legacy Current Window Splitter."""
    g = country_df.sort_values(DATE_COL).reset_index(drop=True).copy()
    g["time_index_1based"] = np.arange(1, len(g) + 1)
    a, b = int(window_row["train_start_row"]), int(window_row["train_end_row"])
    c, d = int(window_row["test_start_row"]), int(window_row["test_end_row"])
    if c != b + 1:
        raise ValueError("Window row continuity contract failed.")
    train = g[(g["time_index_1based"] >= a) & (g["time_index_1based"] <= b)].copy()
    test = g[(g["time_index_1based"] >= c) & (g["time_index_1based"] <= d)].copy()
    if test[DATE_COL].min() != train[DATE_COL].max() + pd.Timedelta(days=1):
        raise ValueError("Date continuity failed between outer train and test.")
    train["window_id"] = int(window_row["window_id"])
    test["window_id"] = int(window_row["window_id"])
    return train.reset_index(drop=True), test.reset_index(drop=True)


def _normalized_entropy(p: np.ndarray, eps: float) -> float:
    # Deliberately reproduces legacy executable semantics: clip without renormalising.
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, 1.0)
    h = -np.sum(p * np.log(p))
    return float(h / np.log(len(p)))


def preprocess_train_test(train_df: pd.DataFrame, test_df: pd.DataFrame, cfg: PY1Config) -> pd.DataFrame:
    """Faithful port of the Stage-1 Dynamic Phase State Representation node."""
    train = train_df.copy()
    test = test_df.copy()
    train["set"] = "train"
    test["set"] = "test"
    df = pd.concat([train, test], ignore_index=True)

    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce").fillna(0.0)
    df = df.sort_values([COUNTRY_COL, DATE_COL]).reset_index(drop=True)

    df["New_cases_nonneg"] = df[TARGET_COL].clip(lower=0).astype(float)
    df["level_log"] = np.log1p(df["New_cases_nonneg"])
    df["y_prev"] = df.groupby(COUNTRY_COL)["New_cases_nonneg"].shift(1)
    df["raw_diff"] = df["New_cases_nonneg"] - df["y_prev"]
    df["bounded_slope"] = (
        (df["New_cases_nonneg"] - df["y_prev"])
        / (df["New_cases_nonneg"] + df["y_prev"] + cfg.eps)
    )
    df["raw_diff"] = df["raw_diff"].fillna(0.0)
    df["bounded_slope"] = df["bounded_slope"].fillna(0.0)

    pieces = []
    for _, g in df.groupby(COUNTRY_COL, sort=False):
        g = g.sort_values(DATE_COL).copy()
        g["slope_acceleration"] = (g["bounded_slope"] - g["bounded_slope"].shift(1)).fillna(0.0)
        g["slope_volatility"] = (
            g["bounded_slope"]
            .rolling(window=cfg.vol_window, min_periods=2)
            .std(ddof=0)
            .fillna(0.0)
        )
        g["volatility_change"] = (g["slope_volatility"] - g["slope_volatility"].shift(1)).fillna(0.0)
        pieces.append(g)
    df = pd.concat(pieces, ignore_index=True)

    df["positive_slope"] = df["bounded_slope"].clip(lower=0.0)
    df["growth_pressure"] = df["level_log"] * df["positive_slope"]
    df["volatility_pressure"] = df["level_log"] * df["slope_volatility"]
    df["surge_pressure"] = df["level_log"] * df["positive_slope"] * df["slope_volatility"]

    slope = df["bounded_slope"].to_numpy(dtype=float)
    direction = np.where(slope < -cfg.direction_tau, 0, np.where(slope <= cfg.direction_tau, 1, 2)).astype(int)
    df["direction_id"] = direction
    df["direction_name"] = pd.Series(direction).map(DIRECTION_NAMES).to_numpy()

    vol = df["slope_volatility"].to_numpy(dtype=float)
    volatility = np.where(vol <= cfg.vol_low_thr, 0, np.where(vol <= cfg.vol_high_thr, 1, 2)).astype(int)
    df["vol_low_threshold"] = cfg.vol_low_thr
    df["vol_high_threshold"] = cfg.vol_high_thr
    df["vol_threshold_source"] = "fixed_semantic"
    df["volatility_id"] = volatility
    df["volatility_name"] = pd.Series(volatility).map(VOL_NAMES).to_numpy()

    df["phase_id"] = (df["direction_id"] * 3 + df["volatility_id"]).astype(int)
    df["phase_name"] = df["direction_name"] + "_" + df["volatility_name"]
    for k in range(9):
        df[f"phase_{k}_onehot"] = (df["phase_id"] == k).astype(int)

    state_pieces = []
    for _, g in df.groupby(COUNTRY_COL, sort=False):
        g = g.sort_values(DATE_COL).copy().reset_index(drop=True)
        phase_ids = g["phase_id"].astype(int).to_numpy()
        onehots = np.zeros((len(g), 9), dtype=float)
        onehots[np.arange(len(g)), phase_ids] = 1.0

        state_rows = []
        P_prev = None
        prev_phase = None
        switch_ema_prev = 0.0
        phase_age = 1

        for i in range(len(g)):
            o_t = onehots[i]
            current_phase = int(phase_ids[i])
            if P_prev is None:
                P_prev = o_t.copy()
                transition_intensity = 0.0
                phase_switch = 0
                switch_ema = 0.0
                phase_age = 1
                P_t = o_t.copy()
            else:
                innovation = o_t - P_prev
                transition_intensity = 0.5 * float(np.sum(np.abs(innovation)))
                phase_switch = int(current_phase != prev_phase)
                phase_age = 1 if phase_switch == 1 else phase_age + 1
                switch_ema = cfg.rho_switch * switch_ema_prev + (1.0 - cfg.rho_switch) * phase_switch
                P_t = cfg.rho_state * P_prev + (1.0 - cfg.rho_state) * o_t
                P_t = np.clip(P_t, 0.0, 1.0)
                # Reproduce legacy +EPS denominator exactly.
                P_t = P_t / (np.sum(P_t) + cfg.eps)

            entropy_norm = _normalized_entropy(P_t, cfg.eps)

            current_decline = float(o_t[0] + o_t[1] + o_t[2])
            current_growth = float(o_t[6] + o_t[7] + o_t[8])
            current_high_vol = float(o_t[2] + o_t[5] + o_t[8])
            current_growth_high_vol = float(o_t[8])
            prev_decline = float(P_prev[0] + P_prev[1] + P_prev[2])
            prev_growth = float(P_prev[6] + P_prev[7] + P_prev[8])
            prev_high_vol = float(P_prev[2] + P_prev[5] + P_prev[8])
            prev_growth_high_vol = float(P_prev[8])

            row = {f"state_phase_{k}": float(P_t[k]) for k in range(9)}
            row.update(
                {
                    "state_transition_intensity": float(transition_intensity),
                    "state_entropy_norm": float(entropy_norm),
                    "state_phase_switch_ema": float(switch_ema),
                    "state_phase_age_norm": float(min(phase_age, cfg.max_phase_age) / float(cfg.max_phase_age)),
                    "state_growth_innovation": float(current_growth - prev_growth),
                    "state_decline_innovation": float(current_decline - prev_decline),
                    "state_high_volatility_innovation": float(current_high_vol - prev_high_vol),
                    "state_growth_high_volatility_innovation": float(current_growth_high_vol - prev_growth_high_vol),
                    "diag_phase_switch": int(phase_switch),
                    "diag_phase_age": int(phase_age),
                }
            )
            state_rows.append(row)
            P_prev = P_t.copy()
            prev_phase = current_phase
            switch_ema_prev = switch_ema

        state_df = pd.DataFrame(state_rows)
        g = pd.concat([g, state_df], axis=1)
        state_pieces.append(g)

    df = pd.concat(state_pieces, ignore_index=True)

    for k in range(9):
        df[f"doc_phase_{k}_wprop"] = df[f"state_phase_{k}"]
    df["growth_any_volatility_signal"] = df[["state_phase_6", "state_phase_7", "state_phase_8"]].sum(axis=1)
    df["high_volatility_signal"] = df[["state_phase_2", "state_phase_5", "state_phase_8"]].sum(axis=1)
    df["growth_high_volatility_signal"] = df["state_phase_8"]
    return df.reset_index(drop=True)


def partition_enriched(enriched: pd.DataFrame, cfg: PY1Config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Faithful port of the Stage-2 partition node including 7-row carry-over."""
    train_df = enriched[enriched["set"] == "train"].copy().sort_values(DATE_COL).reset_index(drop=True)
    test_df = enriched[enriched["set"] == "test"].copy().sort_values(DATE_COL).reset_index(drop=True)
    n_train = len(train_df)
    if n_train <= cfg.validation_size + cfg.lookback:
        raise ValueError("Outer training prefix is too short.")
    if len(test_df) == 0:
        raise ValueError("Test partition is empty.")
    train_df["original_set"] = "train"
    test_df["original_set"] = "test"

    split_idx = n_train - cfg.validation_size
    train_inner = train_df.iloc[:split_idx].copy()
    validation = train_df.iloc[split_idx:].copy()

    insample_context = train_inner.tail(cfg.lookback).copy()
    insample_context["partition"] = "in_sample"
    insample_context["partition_role"] = "context"
    insample_context["is_context"] = 1
    insample_context["is_eval"] = 0
    insample_context["is_train"] = 0
    validation["partition"] = "in_sample"
    validation["partition_role"] = "eval"
    validation["is_context"] = 0
    validation["is_eval"] = 1
    validation["is_train"] = 0
    insample_out = pd.concat([insample_context, validation], ignore_index=True)

    outsample_context = train_df.tail(cfg.lookback).copy()
    outsample_context["partition"] = "out_sample"
    outsample_context["partition_role"] = "context"
    outsample_context["is_context"] = 1
    outsample_context["is_eval"] = 0
    outsample_context["is_train"] = 0
    test_df["partition"] = "out_sample"
    test_df["partition_role"] = "eval"
    test_df["is_context"] = 0
    test_df["is_eval"] = 1
    test_df["is_train"] = 0
    outsample_out = pd.concat([outsample_context, test_df], ignore_index=True)

    train_inner["partition"] = "train_inner"
    train_inner["partition_role"] = "train"
    train_inner["is_context"] = 0
    train_inner["is_eval"] = 0
    train_inner["is_train"] = 1

    for part in (train_inner, insample_out, outsample_out):
        part.reset_index(drop=True, inplace=True)
        part["partition_row_index"] = np.arange(len(part))
        part["lookback"] = cfg.lookback
        part["validation_size"] = cfg.validation_size

    return train_inner, insample_out, outsample_out


def _prepare_sequence_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values(DATE_COL).reset_index(drop=True)
    for c in NUM_FEATURE_COLS + STATE_FEATURE_COLS + [TARGET_COL]:
        if c not in out.columns:
            raise ValueError(f"Missing required sequence feature: {c}")
        out[c] = pd.to_numeric(out[c], errors="coerce")
    feature_cols = NUM_FEATURE_COLS + STATE_FEATURE_COLS
    out[feature_cols] = out[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out[TARGET_COL] = out[TARGET_COL].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def build_sequences(df: pd.DataFrame, partition_name: str, mode: str, cfg: PY1Config) -> pd.DataFrame:
    """Sequence-to-one builder: inputs [i-L, i), target i."""
    df = _prepare_sequence_table(df)
    if len(df) <= cfg.lookback:
        raise ValueError(f"{partition_name}: insufficient rows for lookback={cfg.lookback}.")

    x_y_all = df[TARGET_COL].to_numpy(dtype=float).reshape(-1, 1)
    x_num_all = df[NUM_FEATURE_COLS].to_numpy(dtype=float)
    x_doc_all = df[STATE_FEATURE_COLS].to_numpy(dtype=float)
    y_all = df[TARGET_COL].to_numpy(dtype=float)
    date_all = df[DATE_COL].to_numpy()
    is_eval_all = df.get("is_eval", pd.Series(np.ones(len(df), dtype=int))).astype(int).to_numpy()
    is_context_all = df.get("is_context", pd.Series(np.zeros(len(df), dtype=int))).astype(int).to_numpy()
    role_all = df.get("partition_role", pd.Series(["train"] * len(df))).astype(str).to_numpy()

    optional_target_cols = [
        "direction_id", "direction_name", "volatility_id", "volatility_name", "phase_id", "phase_name",
        "bounded_slope", "slope_volatility", "slope_acceleration", "volatility_change",
        "growth_pressure", "volatility_pressure", "surge_pressure", "diag_phase_switch", "diag_phase_age",
        "state_transition_intensity", "state_entropy_norm", "state_phase_switch_ema", "state_phase_age_norm",
        "state_growth_innovation", "state_decline_innovation", "state_high_volatility_innovation",
        "state_growth_high_volatility_innovation",
    ]
    available_optional_cols = [c for c in optional_target_cols if c in df.columns]

    rows = []
    for i in range(cfg.lookback, len(df)):
        if mode == "eval" and is_eval_all[i] != 1:
            continue
        x_y_seq = x_y_all[i - cfg.lookback : i, :]
        x_num_seq = x_num_all[i - cfg.lookback : i, :]
        x_doc_seq = x_doc_all[i - cfg.lookback : i, :]
        input_start = pd.to_datetime(date_all[i - cfg.lookback])
        input_end = pd.to_datetime(date_all[i - 1])
        target_date = pd.to_datetime(date_all[i])
        if not (input_end < target_date):
            raise AssertionError("Leakage gate failed: input_end_date >= target_date.")
        if target_date - input_end != pd.Timedelta(days=1):
            raise AssertionError("Sequence target is not the immediate next day.")
        row = {
            "seq_id": len(rows),
            "partition": partition_name,
            "target_date": target_date,
            "input_start_date": input_start,
            "input_end_date": input_end,
            "y": float(y_all[i]),
            "New_cases_actual": float(y_all[i]),
            "target_is_eval": int(is_eval_all[i]),
            "target_is_context": int(is_context_all[i]),
            "target_partition_role": str(role_all[i]),
            "lookback": cfg.lookback,
            "n_y_features": 1,
            "n_num_features": len(NUM_FEATURE_COLS),
            "n_doc_features": len(STATE_FEATURE_COLS),
            "target_col": TARGET_COL,
            "num_feature_cols": ",".join(NUM_FEATURE_COLS),
            "doc_feature_cols": ",".join(STATE_FEATURE_COLS),
            "doc_feature_mode": cfg.doc_feature_mode,
            "X_y_json": json.dumps(x_y_seq.tolist()),
            "X_num_json": json.dumps(x_num_seq.tolist()),
            "X_doc_json": json.dumps(x_doc_seq.tolist()),
        }
        for c in available_optional_cols:
            val = df.loc[i, c]
            key = f"target_{c}"
            if pd.isna(val):
                row[key] = ""
            elif isinstance(val, (np.integer, int)):
                row[key] = int(val)
            elif isinstance(val, (np.floating, float)):
                row[key] = float(val)
            else:
                row[key] = str(val)
        rows.append(row)
    return pd.DataFrame(rows)


def _numeric_col(df: pd.DataFrame, col: str, default: float = 0.0) -> np.ndarray:
    if col in df.columns:
        return (
            pd.to_numeric(df[col], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(default)
            .to_numpy(dtype=float)
        )
    return np.full(len(df), default, dtype=float)


def _amp_score(df: pd.DataFrame, cfg: PY1Config) -> np.ndarray:
    if cfg.amp_score_mode != "slope_plus_volatility":
        raise ValueError("Reference PY-1 only permits slope_plus_volatility.")
    bounded = _numeric_col(df, "target_bounded_slope")
    vol = _numeric_col(df, "target_slope_volatility")
    score = np.abs(bounded) + vol
    return np.where(np.isfinite(score), score, 0.0).astype(float)


def add_transition_labels(
    train_seq: pd.DataFrame,
    val_seq: pd.DataFrame,
    test_seq: pd.DataFrame,
    cfg: PY1Config,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    """Faithful label construction; target direction/volatility take precedence."""
    amp_train = _amp_score(train_seq, cfg)
    if len(amp_train) == 0:
        amp_threshold = 1.0
    else:
        amp_threshold = float(np.quantile(amp_train[np.isfinite(amp_train)], cfg.amp_quantile))
        if not np.isfinite(amp_threshold) or amp_threshold < cfg.eps:
            amp_threshold = 1.0

    def label(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "target_direction_id" in out.columns:
            d = pd.to_numeric(out["target_direction_id"], errors="coerce").fillna(1).astype(int).clip(0, 2).to_numpy()
        else:
            s = _numeric_col(out, "target_bounded_slope")
            d = np.where(s > cfg.direction_tau, 2, np.where(s < -cfg.direction_tau, 0, 1)).astype(int)
        if "target_volatility_id" in out.columns:
            v = pd.to_numeric(out["target_volatility_id"], errors="coerce").fillna(1).astype(int).clip(0, 2).to_numpy()
        else:
            sv = _numeric_col(out, "target_slope_volatility")
            v = np.where(sv <= cfg.vol_low_thr, 0, np.where(sv <= cfg.vol_high_thr, 1, 2)).astype(int)
        amp = _amp_score(out, cfg)
        out["future_dir_label"] = d
        out["future_vol_label"] = v
        out["future_amp_score"] = amp
        out["future_amp_label"] = (amp >= amp_threshold).astype(int)
        out["r2_amp_threshold"] = amp_threshold
        out["r2_amp_quantile"] = cfg.amp_quantile
        out["r2_amp_score_mode"] = cfg.amp_score_mode
        out["r2_vol_low_threshold"] = cfg.vol_low_thr
        out["r2_vol_high_threshold"] = cfg.vol_high_thr
        return out

    return label(train_seq), label(val_seq), label(test_seq), amp_threshold


def _frame_hash(df: pd.DataFrame, cols: Iterable[str]) -> str:
    """Stable semantic hash for deterministic parity diagnostics."""
    cols = list(cols)
    h = hashlib.sha256()
    for row in df[cols].itertuples(index=False, name=None):
        norm = []
        for v in row:
            if isinstance(v, pd.Timestamp):
                norm.append(v.isoformat())
            elif isinstance(v, np.generic):
                norm.append(v.item())
            elif isinstance(v, float):
                norm.append(format(v, ".17g"))
            else:
                norm.append(v)
        h.update(json.dumps(norm, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def build_window_bundle(country_df: pd.DataFrame, window_id: int, cfg: PY1Config) -> Dict[str, object]:
    cfg.validate()
    g = country_df.sort_values(DATE_COL).reset_index(drop=True)
    windows = generate_window_table(g, cfg)
    wr = windows.loc[windows["window_id"] == window_id].iloc[0].to_dict()
    train_raw, test_raw = _slice_current_window(g, wr)
    enriched = preprocess_train_test(train_raw, test_raw, cfg)
    train_inner, val_with_context, test_with_context = partition_enriched(enriched, cfg)
    train_seq = build_sequences(train_inner, "train_inner", "train", cfg)
    val_seq = build_sequences(val_with_context, "in_sample", "eval", cfg)
    test_seq = build_sequences(test_with_context, "out_sample", "eval", cfg)
    train_lab, val_lab, test_lab, amp_threshold = add_transition_labels(train_seq, val_seq, test_seq, cfg)

    expected_train_seq = int(wr["train_size"]) - cfg.validation_size - cfg.lookback
    if len(train_lab) != expected_train_seq:
        raise AssertionError(f"Train sequence count mismatch: {len(train_lab)} != {expected_train_seq}")
    if len(val_lab) != cfg.validation_size or len(test_lab) != cfg.test_size:
        raise AssertionError("Validation/test sequence count mismatch.")
    if train_lab["n_num_features"].nunique() != 1 or int(train_lab["n_num_features"].iloc[0]) != 3:
        raise AssertionError("Numeric stream dimension parity failed.")
    if train_lab["n_doc_features"].nunique() != 1 or int(train_lab["n_doc_features"].iloc[0]) != 17:
        raise AssertionError("Dynamic phase-state dimension parity failed.")

    # State numerical invariants on all daily records available in the window.
    state = enriched[[f"state_phase_{k}" for k in range(9)]].to_numpy(dtype=float)
    if not np.isfinite(enriched[NUM_FEATURE_COLS + STATE_FEATURE_COLS].to_numpy(dtype=float)).all():
        raise AssertionError("Non-finite deterministic features found.")
    if (state < -1e-12).any():
        raise AssertionError("Negative phase-state weight found.")
    state_sums = state.sum(axis=1)
    if np.max(np.abs(state_sums - 1.0)) > 2e-8:
        raise AssertionError("Phase-state approximate unit-sum invariant failed.")

    country = str(g[COUNTRY_COL].iloc[0])
    audit = {
        "Country": country,
        "Country_canonical": CANONICAL_COUNTRY.get(country, country),
        "window_id": int(window_id),
        "outer_train_rows": len(train_raw),
        "train_inner_rows": len(train_inner),
        "validation_rows": cfg.validation_size,
        "test_rows": len(test_raw),
        "train_sequences": len(train_lab),
        "validation_sequences": len(val_lab),
        "test_sequences": len(test_lab),
        "train_start_date": train_raw[DATE_COL].min(),
        "train_inner_end_date": train_inner[DATE_COL].max(),
        "validation_start_date": val_lab["target_date"].min(),
        "validation_end_date": val_lab["target_date"].max(),
        "test_start_date": test_lab["target_date"].min(),
        "test_end_date": test_lab["target_date"].max(),
        "amp_threshold": amp_threshold,
        "max_state_sum_abs_error": float(np.max(np.abs(state_sums - 1.0))),
        "daily_feature_hash": _frame_hash(
            enriched,
            [DATE_COL, TARGET_COL, "level_log", "bounded_slope", "slope_volatility", "direction_id", "volatility_id", "phase_id"] + STATE_FEATURE_COLS,
        ),
        "train_sequence_hash": _frame_hash(train_lab, ["target_date", "X_y_json", "X_num_json", "X_doc_json", "future_dir_label", "future_vol_label", "future_amp_label"]),
        "validation_sequence_hash": _frame_hash(val_lab, ["target_date", "X_y_json", "X_num_json", "X_doc_json", "future_dir_label", "future_vol_label", "future_amp_label"]),
        "test_sequence_hash": _frame_hash(test_lab, ["target_date", "X_y_json", "X_num_json", "X_doc_json", "future_dir_label", "future_vol_label", "future_amp_label"]),
    }
    return {
        "window": wr,
        "enriched": enriched,
        "train_inner": train_inner,
        "validation_with_context": val_with_context,
        "test_with_context": test_with_context,
        "train_sequences": train_lab,
        "validation_sequences": val_lab,
        "test_sequences": test_lab,
        "audit": audit,
    }


def run_audit(df: pd.DataFrame, cfg: PY1Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute all 6 x 10 deterministic country-window reconstructions."""
    dataset_audit = validate_dataset(df, cfg)
    audit_rows = []
    for country, g in df.groupby(COUNTRY_COL, sort=True):
        g = g.sort_values(DATE_COL).reset_index(drop=True)
        for window_id in range(1, cfg.n_windows + 1):
            bundle = build_window_bundle(g, window_id, cfg)
            audit_rows.append(bundle["audit"])
    return dataset_audit, pd.DataFrame(audit_rows)


def save_reference_config(path: str | Path, cfg: PY1Config) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, sort_keys=True)
