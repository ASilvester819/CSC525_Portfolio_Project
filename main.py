#!/usr/bin/env python3
# ============================================================
# Portfolio Allocation Agent (TensorFlow + Ken French Daily)
#
# User enters max acceptable 1-month loss in percent (e.g., -10)
# Risk constraint: VaR-style, 5th percentile of SIMULATED 1-month return >= threshold
# Forward-looking distribution:
#   daily return = TF mean forecast + bootstrapped residual VECTOR
#   simulate 21 trading days, compound across days to 1-month return distribution
# ============================================================

import os
import io
import zipfile
import argparse
import urllib.request
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
import tensorflow as tf


# ============================================================
# ========================= USER INPUTS ======================
# ============================================================

FRENCH_49_DAILY_ZIP_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
    "ftp/49_Industry_Portfolios_daily_CSV.zip"
)

FF_FACTORS_DAILY_ZIP_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
    "ftp/F-F_Research_Data_Factors_daily_CSV.zip"
)

ARTIFACT_DIR = "artifacts"
PLOTS_DIR = os.path.join(ARTIFACT_DIR, "plots")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "tf_model.keras")
SCALER_PATH = os.path.join(ARTIFACT_DIR, "scaler.npy")
RESID_PATH = os.path.join(ARTIFACT_DIR, "residual_pool.npy")

# ML hyperparameters
LAGS_DAYS = 20
TRAIN_PCT = 0.60
VAL_PCT = 0.20
EPOCHS = 80
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
PATIENCE = 10
DROPOUT_RATE = 0.10
HIDDEN_SIZES = [512, 256, 128]

# Portfolio universe
N_PORTFOLIOS = 15000
DIRICHLET_CONC = 1.0
MAX_WEIGHT = 0.15
RF_MAX_WEIGHT = 0.95

# Risk horizon + VaR definition
MONTH_DAYS = 21
VAR_ALPHA = 0.05  # 5% tail
MIN_FEASIBLE = 5
TOP_K_TO_SHOW = 5

# Simulation controls
N_SIMS = 2000
CLIP_MIN_DAILY = -0.50
CLIP_MAX_DAILY = 0.50

# Reward weights (among feasible portfolios)
REWARD_WEIGHTS = {
    "lambda_vol":  0.20,  # penalty on monthly std
    "lambda_conc": 0.10,  # penalty on HHI
}

SEED = 42
MISSING_CODES = {-99.99, -999.0}


# ============================================================
# ====================== SIGNATURE HELPERS ===================
# ============================================================

def make_signature(assets: List[str], lags: int, input_dim: int, output_dim: int) -> Dict:
    return {
        "assets": list(assets),
        "lags": int(lags),
        "input_dim": int(input_dim),
        "output_dim": int(output_dim),
    }

def signature_matches(saved: Dict, current: Dict) -> bool:
    keys = ["assets", "lags", "input_dim", "output_dim"]
    return all(saved.get(k) == current.get(k) for k in keys)


# ============================================================
# ====================== UTILITIES ===========================
# ============================================================

def ensure_dirs():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

def download_zip(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r:
        return r.read()

def prompt_max_monthly_loss() -> float:
    print("\nRisk tolerance input (VaR-style):")
    print("  Enter the maximum 1-month loss you are willing to tolerate (in %).")
    print(f"  Constraint: {int(VAR_ALPHA*100)}th percentile of simulated 1-month return >= your value.")
    raw = input("\nEnter max 1-month loss (e.g., -10): ").strip()

    try:
        val_pct = float(raw)
    except ValueError:
        raise ValueError("Invalid input. Please enter a numeric value like -10 or -5.")

    if val_pct > 0:
        print("Note: converting positive input to negative loss limit.")
        val_pct = -abs(val_pct)

    return val_pct / 100.0

def _is_yyyymmdd(token: str) -> bool:
    t = token.strip()
    return len(t) == 8 and t.isdigit()

def _locate_data_block(lines: List[str], mode: str) -> Tuple[int, Optional[int]]:
    """
    Find first YYYYMMDD data row and attempt to identify a header row above it.

    mode="factors": look for factor tokens (mkt-rf, smb, hml, rf)
    mode="industries": look for typical industry header tokens
    """
    mode = mode.lower().strip()
    if mode not in {"factors", "industries"}:
        raise ValueError("mode must be 'factors' or 'industries'")

    # tokens to detect in the header line
    if mode == "factors":
        header_tokens = ["rf", "mkt-rf", "mktrf", "smb", "hml"]
    else:
        # Several common 49-industry abbreviations (Ken French)
        header_tokens = [
            "agric", "food", "soda", "beer", "smoke", "toys", "fun",
            "books", "hshld", "clths", "hlth", "medeq", "drugs",
            "chems", "rubbr", "txtls", "bldmt", "cnstr", "steel",
            "mach", "elceq", "autos", "aero", "ships", "guns",
            "gold", "mines", "coal", "oil", "util", "telcm",
            "perSv".lower(), "busSv".lower(), "hardw", "softw",
            "chips", "labEq".lower(), "paper", "boxes", "trans",
            "whlsl".lower(), "rtail".lower(), "meals", "banks",
            "insur".lower(), "rlEst".lower(), "fin", "other"
        ]

    def header_looks_right(header_line: str) -> bool:
        s = header_line.strip().lower()
        # require at least 5 columns
        if s.count(",") < 5:
            return False
        # if any known token is present, treat as header
        return any(tok in s for tok in header_tokens)

    for i, line in enumerate(lines):
        if not line.strip():
            continue
        first = line.split(",")[0].strip()
        if _is_yyyymmdd(first):
            header_idx = i - 1 if i > 0 else None
            if header_idx is not None and header_looks_right(lines[header_idx]):
                return i, header_idx
            return i, None

    raise ValueError("Could not locate a numeric YYYYMMDD data block in the file.")



# ============================================================
# ====================== PARSERS =============================
# ============================================================

def _read_zip_csv_lines(zip_bytes: bytes) -> List[str]:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        csv_name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
        raw = z.read(csv_name)

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    return text.splitlines()

def parse_french_daily_table(zip_bytes: bytes, mode: str) -> pd.DataFrame:
    lines = _read_zip_csv_lines(zip_bytes)
    start_line, header_line = _locate_data_block(lines, mode=mode)

    if header_line is not None:
        block = "\n".join(lines[header_line:])
        df = pd.read_csv(io.StringIO(block), dtype=str, low_memory=False)
        df = df.rename(columns={df.columns[0]: "Date"})
    else:
        block = "\n".join(lines[start_line:])
        df = pd.read_csv(io.StringIO(block), header=None, dtype=str, low_memory=False)
        df.columns = ["Date"] + [f"V{i}" for i in range(1, df.shape[1])]

    df["Date"] = pd.to_numeric(df["Date"].astype(str).str.strip(), errors="coerce")
    df = df.dropna(subset=["Date"]).copy()
    df["Date"] = pd.to_datetime(df["Date"].astype(np.int64).astype(str), format="%Y%m%d", errors="raise")

    for c in df.columns:
        if c == "Date":
            continue
        df[c] = pd.to_numeric(df[c].astype(str).str.strip(), errors="coerce")
        df[c] = df[c].replace(list(MISSING_CODES), np.nan)

    df = df.sort_values("Date").reset_index(drop=True)
    return df


def load_industries_daily() -> pd.DataFrame:
    z = download_zip(FRENCH_49_DAILY_ZIP_URL)
    df = parse_french_daily_table(z, mode="industries")


    for c in df.columns:
        if c != "Date":
            df[c] = df[c] / 100.0

    med = df.drop(columns=["Date"]).median(numeric_only=True)
    df.loc[:, med.index] = df.loc[:, med.index].fillna(med)
    return df

    v_cols = [c for c in df.columns if c.startswith("V")]
    if len(v_cols) > 0:
        raise RuntimeError(
            "Industry file header was not detected; columns are V1..Vn. "
            "Stop here to avoid ambiguous allocations. "
            "We need to fix header detection for your environment."
        )


def load_ff_factors_daily_rf() -> pd.DataFrame:
    z = download_zip(FF_FACTORS_DAILY_ZIP_URL)
    df = parse_french_daily_table(z, mode="factors")


    if "RF" in df.columns:
        rf_col = "RF"
    else:
        # F-F factors are typically: Mkt-RF, SMB, HML, RF -> RF in V4 if no header
        if "V4" not in df.columns:
            raise ValueError(f"Could not locate RF column. Columns: {df.columns.tolist()}")
        rf_col = "V4"

    out = df[["Date", rf_col]].copy().rename(columns={rf_col: "RF"})
    out["RF"] = (out["RF"] / 100.0).astype(float)
    out["RF"] = out["RF"].fillna(out["RF"].median())
    return out

def load_data_with_rf() -> Tuple[pd.DataFrame, List[str]]:
    ensure_dirs()
    print("\nDownloading data...")
    ind = load_industries_daily()
    rf = load_ff_factors_daily_rf()
    df = pd.merge(ind, rf, on="Date", how="inner")

    assets = [c for c in df.columns if c != "Date"]
    if "RF" not in assets:
        raise RuntimeError("RF missing after merge; cannot proceed.")

    print(f"Assets: {len(assets)} (includes RF)")
    print(f"Date range: {df['Date'].min().date()} → {df['Date'].max().date()}")
    return df, assets


# ============================================================
# ====================== ML MODEL ============================
# ============================================================

def make_supervised(df: pd.DataFrame, assets: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    R = df[assets].to_numpy(dtype=np.float32)
    X, y = [], []
    for i in range(LAGS_DAYS, len(R) - 1):
        X.append(R[i - LAGS_DAYS:i].reshape(-1))
        y.append(R[i + 1])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)

def build_model(input_dim: int, output_dim: int) -> tf.keras.Model:
    layers = [tf.keras.layers.Input(shape=(input_dim,))]
    for h in HIDDEN_SIZES:
        layers.append(tf.keras.layers.Dense(h, activation="relu"))
        layers.append(tf.keras.layers.Dropout(DROPOUT_RATE))
    layers.append(tf.keras.layers.Dense(output_dim))
    model = tf.keras.Sequential(layers)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse",
        metrics=["mae"]
    )
    return model

def train_or_load_model(X: np.ndarray, y: np.ndarray, assets: List[str], retrain: bool) -> Tuple[tf.keras.Model, StandardScaler]:
    n = len(X)
    train_end = int(TRAIN_PCT * n)
    val_end = int((TRAIN_PCT + VAL_PCT) * n)

    current_sig = make_signature(assets, LAGS_DAYS, X.shape[1], y.shape[1])

    scaler = StandardScaler().fit(X[:train_end])
    Xs = scaler.transform(X).astype(np.float32)

    if (not retrain) and os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        try:
            saved = np.load(SCALER_PATH, allow_pickle=True).item()
            saved_sig = saved.get("signature", {})

            if signature_matches(saved_sig, current_sig):
                model = tf.keras.models.load_model(MODEL_PATH)
                scaler = StandardScaler()
                scaler.mean_ = saved["mean_"]
                scaler.scale_ = saved["scale_"]
                scaler.var_ = scaler.scale_ ** 2
                scaler.n_features_in_ = len(scaler.mean_)
                print("\nLoaded saved model + scaler (signature match).")
                return model, scaler

            print("\nSaved model/scaler signature mismatch — retraining.")
        except Exception as e:
            print(f"\nFailed to load saved model/scaler ({e}) — retraining.")

    model = build_model(X.shape[1], y.shape[1])

    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        restore_best_weights=True
    )

    history = model.fit(
        Xs[:train_end], y[:train_end],
        validation_data=(Xs[train_end:val_end], y[train_end:val_end]),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[es],
        verbose=1
    )

    hist = pd.DataFrame(history.history)
    plt.figure()
    plt.plot(hist["loss"], label="train loss")
    plt.plot(hist["val_loss"], label="val loss")
    plt.legend()
    plt.title("TF training vs validation loss")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "tf_loss.png"), dpi=150)
    plt.close()

    model.save(MODEL_PATH)
    np.save(
        SCALER_PATH,
        {"mean_": scaler.mean_, "scale_": scaler.scale_, "signature": current_sig},
        allow_pickle=True
    )

    print("\nTrained and saved model + scaler (new signature).")
    return model, scaler


# ============================================================
# ====================== RESIDUAL POOL =======================
# ============================================================

def build_or_load_residual_pool(model: tf.keras.Model,
                               scaler: StandardScaler,
                               X: np.ndarray,
                               y: np.ndarray,
                               assets: List[str],
                               retrain: bool) -> np.ndarray:
    n = len(X)
    train_end = int(TRAIN_PCT * n)
    current_sig = make_signature(assets, LAGS_DAYS, X.shape[1], y.shape[1])

    if (not retrain) and os.path.exists(RESID_PATH):
        try:
            saved = np.load(RESID_PATH, allow_pickle=True).item()
            resid = saved["resid"].astype(np.float32, copy=False)
            saved_sig = saved.get("signature", {})
            if signature_matches(saved_sig, current_sig):
                print("\nLoaded saved residual pool (signature match).")
                return resid
            print("\nSaved residual pool signature mismatch — rebuilding.")
        except Exception as e:
            print(f"\nFailed to load residual pool ({e}) — rebuilding.")

    X_train = X[:train_end]
    y_train = y[:train_end]

    Xs = scaler.transform(X_train).astype(np.float32)
    yhat = model.predict(Xs, verbose=0).astype(np.float32)

    resid = (y_train - yhat).astype(np.float32)
    resid = resid - resid.mean(axis=0, keepdims=True)

    np.save(RESID_PATH, {"resid": resid, "signature": current_sig}, allow_pickle=True)
    print("\nBuilt and saved residual pool (new signature).")
    return resid


# ============================================================
# ====================== PORTFOLIOS + SIM ====================
# ============================================================

def sample_weights_with_rf(assets: List[str]) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    n_assets = len(assets)
    alpha = np.ones(n_assets, dtype=np.float64) * DIRICHLET_CONC

    rf_idx = assets.index("RF")
    W = np.empty((N_PORTFOLIOS, n_assets), dtype=np.float32)
    filled = 0

    while filled < N_PORTFOLIOS:
        batch_size = min(5000, N_PORTFOLIOS - filled)
        batch = rng.dirichlet(alpha, size=batch_size)

        max_non_rf = np.max(np.delete(batch, rf_idx, axis=1), axis=1)
        rf_w = batch[:, rf_idx]

        mask = (max_non_rf <= MAX_WEIGHT) & (rf_w <= RF_MAX_WEIGHT)
        accepted = batch[mask]

        if accepted.size:
            take = min(len(accepted), N_PORTFOLIOS - filled)
            W[filled:filled+take] = accepted[:take].astype(np.float32, copy=False)
            filled += take

    return W

def forecast_mean_daily(model: tf.keras.Model, scaler: StandardScaler, X_last: np.ndarray) -> np.ndarray:
    Xs = scaler.transform(X_last).astype(np.float32)
    pred = model.predict(Xs, verbose=0)[0].astype(np.float64)
    pred = np.clip(pred, CLIP_MIN_DAILY, CLIP_MAX_DAILY)
    return pred.astype(np.float32)

def expected_1m_from_mean_daily(mean_daily: np.ndarray) -> np.ndarray:
    md = np.clip(mean_daily.astype(np.float64), -0.999999, None)
    return (np.power(1.0 + md, MONTH_DAYS) - 1.0).astype(np.float64)

def simulate_daily_paths(mean_daily: np.ndarray,
                         resid_pool: np.ndarray,
                         n_sims: int,
                         n_days: int,
                         seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_resid = resid_pool.shape[0]
    n_assets = resid_pool.shape[1]

    idx = rng.integers(0, n_resid, size=(n_sims, n_days), endpoint=False)
    resid_samples = resid_pool[idx]  # (n_sims, n_days, n_assets)

    mean = mean_daily.reshape(1, 1, n_assets).astype(np.float32)
    sim = mean + resid_samples.astype(np.float32)
    sim = np.clip(sim, CLIP_MIN_DAILY, CLIP_MAX_DAILY)
    return sim.astype(np.float32, copy=False)

def compound_over_days(port_daily: np.ndarray) -> np.ndarray:
    """
    Correct compounding across the DAYS axis.

    Accepts:
      port_daily shape (n_sims, n_days, B) -> returns (n_sims, B)
    """
    if port_daily.ndim != 3:
        raise ValueError(f"compound_over_days expects 3D array, got shape {port_daily.shape}")

    # Use log1p summation for stability: exp(sum(log(1+r))) - 1
    x = np.clip(port_daily, -0.999999, None)
    log1p = np.log1p(x)                    # (n_sims, n_days, B)
    summed = np.sum(log1p, axis=1)         # sum across days -> (n_sims, B)
    return np.expm1(summed)                # (n_sims, B)

def evaluate_and_select_topk(max_loss_1m: float,
                             W: np.ndarray,
                             sim_daily_assets: np.ndarray,
                             exp_1m_assets: np.ndarray,
                             assets: List[str]) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Fix: compound across days (axis=1), not across portfolios.
    """
    n_sims, n_days, n_assets = sim_daily_assets.shape
    n_ports = W.shape[0]

    if n_days != MONTH_DAYS:
        raise ValueError(f"Simulation days mismatch: got {n_days}, expected {MONTH_DAYS}")

    W64 = W.astype(np.float64, copy=False)
    hhi_all = np.sum(W64 * W64, axis=1)
    forecast_1m_all = W64 @ exp_1m_assets.astype(np.float64)

    lam_v = REWARD_WEIGHTS["lambda_vol"]
    lam_c = REWARD_WEIGHTS["lambda_conc"]

    records = []
    B = 500

    for start in range(0, n_ports, B):
        end = min(start + B, n_ports)
        Wb = W64[start:end]  # (B, n_assets)

        # (n_sims, n_days, n_assets) dot (n_assets, B) -> (n_sims, n_days, B)
        rp = np.tensordot(sim_daily_assets, Wb.T, axes=([2], [0]))

        # Shape check (prevents this exact bug from returning)
        if rp.shape != (n_sims, n_days, end - start):
            raise ValueError(f"Unexpected rp shape {rp.shape}, expected {(n_sims, n_days, end-start)}")

        # Correct: compound across DAYS -> (n_sims, B_batch)
        r1m = compound_over_days(rp)  # (n_sims, end-start)

        # Now stats are length B_batch, as intended
        p5 = np.quantile(r1m, VAR_ALPHA, axis=0)    # (B_batch,)
        med = np.quantile(r1m, 0.50, axis=0)
        p20 = np.quantile(r1m, 0.20, axis=0)
        p80 = np.quantile(r1m, 0.80, axis=0)
        std = np.std(r1m, axis=0, ddof=0)
        mean = np.mean(r1m, axis=0)

        B_batch = end - start
        for j in range(B_batch):
            idx = start + j
            if float(p5[j]) < max_loss_1m:
                continue

            reward = float(mean[j]) - lam_v * float(std[j]) - lam_c * float(hhi_all[idx])

            records.append({
                "idx": idx,
                "reward": reward,
                "mean_1m": float(mean[j]),
                "forecast_1m": float(forecast_1m_all[idx]),
                "p5_1m": float(p5[j]),
                "median_1m": float(med[j]),
                "p20_1m": float(p20[j]),
                "p80_1m": float(p80[j]),
                "std_1m": float(std[j]),
                "hhi": float(hhi_all[idx]),
            })

    feasible = pd.DataFrame(records)
    if len(feasible) < MIN_FEASIBLE:
        raise RuntimeError(
            f"Only {len(feasible)} portfolios satisfy VaR constraint (p5_1m >= {max_loss_1m:.2%}). "
            f"Loosen max loss, increase RF_MAX_WEIGHT, increase N_PORTFOLIOS, or increase N_SIMS."
        )

    feasible = feasible.sort_values(["reward", "p5_1m", "std_1m"], ascending=[False, False, True])
    top = feasible.head(TOP_K_TO_SHOW).copy()
    topW = W[top["idx"].astype(int).to_numpy()].astype(np.float64, copy=False)
    return top, topW


# ============================================================
# ============================= MAIN =========================
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", action="store_true", help="Force retrain model")
    args, _unknown = parser.parse_known_args()

    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    ensure_dirs()

    max_loss_1m = prompt_max_monthly_loss()
    print(f"\nConstraint: VaR_{int(VAR_ALPHA*100)}% (simulated 1-month return) >= {max_loss_1m:.2%}")

    df, assets = load_data_with_rf()
    X, y = make_supervised(df, assets)

    model, scaler = train_or_load_model(X, y, assets, retrain=args.retrain)
    resid_pool = build_or_load_residual_pool(model, scaler, X, y, assets, retrain=args.retrain)

    X_last = X[-1:].astype(np.float32)
    mean_daily = forecast_mean_daily(model, scaler, X_last)
    exp_1m_assets = expected_1m_from_mean_daily(mean_daily)

    print(f"\nSimulating {N_SIMS} forward paths of {MONTH_DAYS} trading days (mean + bootstrapped residual vectors)...")
    sim_daily_assets = simulate_daily_paths(
        mean_daily=mean_daily,
        resid_pool=resid_pool,
        n_sims=N_SIMS,
        n_days=MONTH_DAYS,
        seed=SEED + 123
    )

    print(f"\nSampling {N_PORTFOLIOS} long-only portfolios (MAX_WEIGHT={MAX_WEIGHT:.0%}, RF_MAX_WEIGHT={RF_MAX_WEIGHT:.0%})...")
    W = sample_weights_with_rf(assets)
    print(f"Generated W: {W.shape}, {W.nbytes/1e6:.2f} MB")

    top_df, top_W = evaluate_and_select_topk(
        max_loss_1m=max_loss_1m,
        W=W,
        sim_daily_assets=sim_daily_assets,
        exp_1m_assets=exp_1m_assets,
        assets=assets
    )

    print(f"\nTOP {TOP_K_TO_SHOW} feasible portfolios (sorted by Reward):")
    print(top_df[[
        "idx", "reward", "mean_1m", "forecast_1m",
        "p5_1m", "median_1m", "p20_1m", "p80_1m",
        "std_1m", "hhi"
    ]].to_string(index=False))

    best_weights = pd.Series(top_W[0], index=assets).sort_values(ascending=False)

    # percent formatting
    best_weights_pct = (best_weights * 100)
    out = best_weights_pct.head(15).map(lambda x: f"{x:,.2f}%")
    print("\nBEST PORTFOLIO allocation (top 15 weights, %):")
    print(out.to_string())


    #print("\nBEST PORTFOLIO allocation (top 15 weights, %):")
    #print(best_weights_pct.head(15).astype(str).radd("").to_string())

    #print(best_weights.head(15).to_string())

    #print("\nDone.")


if __name__ == "__main__":
    main()
