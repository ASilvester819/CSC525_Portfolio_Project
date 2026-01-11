import os
import io
import json
import zipfile
import argparse
import urllib.request
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
import tensorflow as tf


# User Inputs / Constants
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
MODELS_DIR = os.path.join(ARTIFACT_DIR, "models")
MODEL_META_PATH = os.path.join(ARTIFACT_DIR, "model_meta.json")
SCALER_PATH = os.path.join(ARTIFACT_DIR, "scaler.npy")
RESID_PATH = os.path.join(ARTIFACT_DIR, "residual_buckets.npy")

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

# Ensemble controls
ENSEMBLE_K = 5
BOOTSTRAP_TRAIN_FOR_ENSEMBLE = False  

# Portfolio universe
N_PORTFOLIOS = 15000
DIRICHLET_CONC = 1.0
MAX_WEIGHT = 0.15
RF_MAX_WEIGHT = 0.95

# Risk horizon + VaR definition 
MONTH_DAYS = 21
VAR_ALPHA = 0.05  
MIN_FEASIBLE = 5
TOP_K_TO_SHOW = 5

# Simulation controls
N_SIMS = 2000
CLIP_MIN_DAILY = -0.50
CLIP_MAX_DAILY = 0.50

# Reward weights 
REWARD_WEIGHTS = {
    "lambda_vol":  0.20,  # penalty on monthly volatility
    "lambda_conc": 0.10,  # penalty on concentration
}

# Regime conditioning 
N_VOL_BUCKETS = 3
REGIME_FEATURES_SPEC = {
    "vol_21": 21,
    "vol_63": 63,
    "trend_21": 21,
    "trend_63": 63,
    "dd_252": 252,
    "xdisp_21": 21,
}
REGIME_FEATURE_ORDER = ["vol_21", "vol_63", "trend_21", "trend_63", "dd_252", "xdisp_21"]

# Self-tuning controls
TUNE_DEFAULT_TRIALS = 20
TUNE_EPOCHS = 30
TUNE_HIDDEN_CANDIDATES = [
    [256, 128],
    [512, 256, 128],
    [512, 256],
    [256, 256, 128],
    [768, 384, 192],
]

SEED = 42
MISSING_CODES = {-99.99, -999.0}


# Signature Helpers
def make_signature(
    assets: List[str],
    lags: int,
    input_dim: int,
    output_dim: int,
    hidden_sizes: List[int],
    dropout: float,
    lr: float,
    ensemble_k: int,
    regime_feature_order: List[str],
    n_vol_buckets: int
) -> Dict:
    return {
        "assets": list(assets),
        "lags": int(lags),
        "input_dim": int(input_dim),
        "output_dim": int(output_dim),
        "hidden_sizes": list(map(int, hidden_sizes)),
        "dropout": float(dropout),
        "learning_rate": float(lr),
        "ensemble_k": int(ensemble_k),
        "regime_feature_order": list(regime_feature_order),
        "n_vol_buckets": int(n_vol_buckets),
    }

def signature_matches(saved: Dict, current: Dict) -> bool:
    keys = [
        "assets", "lags", "input_dim", "output_dim",
        "hidden_sizes", "dropout", "learning_rate", "ensemble_k",
        "regime_feature_order", "n_vol_buckets"
    ]
    return all(saved.get(k) == current.get(k) for k in keys)


# Utilities
def ensure_dirs():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

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
    mode = mode.lower().strip()
    if mode not in {"factors", "industries"}:
        raise ValueError("mode must be 'factors' or 'industries'")

    if mode == "factors":
        header_tokens = ["rf", "mkt-rf", "mktrf", "smb", "hml"]
    else:
        header_tokens = [
            "agric", "food", "soda", "beer", "smoke", "toys", "fun",
            "books", "hshld", "clths", "hlth", "medeq", "drugs",
            "chems", "rubbr", "txtls", "bldmt", "cnstr", "steel",
            "mach", "elceq", "autos", "aero", "ships", "guns",
            "gold", "mines", "coal", "oil", "util", "telcm",
            "persv", "bussv", "hardw", "softw",
            "chips", "labeq", "paper", "boxes", "trans",
            "whlsl", "rtail", "meals", "banks",
            "insur", "rlest", "fin", "other"
        ]

    def header_looks_right(header_line: str) -> bool:
        s = header_line.strip().lower()
        if s.count(",") < 5:
            return False
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


# Parsers / Data Load
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

    # SAFETY: if header was not detected, stop
    v_cols = [c for c in df.columns if c.startswith("V")]
    if len(v_cols) > 0:
        raise RuntimeError(
            "Industry file header was not detected; columns are V1..Vn. "
            "Stop here to avoid ambiguous allocations."
        )

    for c in df.columns:
        if c != "Date":
            df[c] = df[c] / 100.0

    med = df.drop(columns=["Date"]).median(numeric_only=True)
    df.loc[:, med.index] = df.loc[:, med.index].fillna(med)
    return df

def load_ff_factors_daily_rf() -> pd.DataFrame:
    z = download_zip(FF_FACTORS_DAILY_ZIP_URL)
    df = parse_french_daily_table(z, mode="factors")

    if "RF" in df.columns:
        rf_col = "RF"
    else:
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

    if "RF" not in df.columns:
        raise RuntimeError("RF missing after merge; cannot proceed.")

    # Deterministic ordering: RF last
    cols = [c for c in df.columns if c not in ("Date", "RF")]
    df = df[["Date"] + cols + ["RF"]]

    assets = [c for c in df.columns if c != "Date"]
    print(f"Assets: {len(assets)} (includes RF)")
    print(f"Date range: {df['Date'].min().date()} → {df['Date'].max().date()}")
    return df, assets


# Regime Features
def compute_regime_features(df: pd.DataFrame, assets: List[str]) -> pd.DataFrame:
    """
    Computes a regime feature DataFrame indexed like df (same row count), including:
      vol_21, vol_63, trend_21, trend_63, dd_252, xdisp_21
    Uses only contemporaneous/past data (rolling windows).
    """
    industry_cols = [c for c in assets if c != "RF"]
    if len(industry_cols) == 0:
        raise ValueError("No industry columns found to compute regime features.")

    ind = df[industry_cols]
    mkt = ind.mean(axis=1)  # equal-weight market proxy

    # Rolling vol and trend
    vol_21 = mkt.rolling(REGIME_FEATURES_SPEC["vol_21"]).std()
    vol_63 = mkt.rolling(REGIME_FEATURES_SPEC["vol_63"]).std()
    trend_21 = mkt.rolling(REGIME_FEATURES_SPEC["trend_21"]).mean()
    trend_63 = mkt.rolling(REGIME_FEATURES_SPEC["trend_63"]).mean()

    # Drawdown on market proxy index
    mkt_index = (1.0 + mkt.fillna(0.0)).cumprod()
    roll_max = mkt_index.rolling(REGIME_FEATURES_SPEC["dd_252"], min_periods=1).max()
    dd_252 = (mkt_index / roll_max) - 1.0

    # Cross-sectional dispersion (rolling mean of daily dispersion)
    xdisp_daily = ind.std(axis=1)
    xdisp_21 = xdisp_daily.rolling(REGIME_FEATURES_SPEC["xdisp_21"]).mean()

    feats = pd.DataFrame({
        "vol_21": vol_21,
        "vol_63": vol_63,
        "trend_21": trend_21,
        "trend_63": trend_63,
        "dd_252": dd_252,
        "xdisp_21": xdisp_21,
    })
    return feats

def build_vol_bucket_edges(vol63_series: pd.Series, train_mask: np.ndarray) -> np.ndarray:
    """
    Returns edges for 3 buckets based on training-sample vol_63 quantiles.
    For 3 buckets, we need two cutoffs: q(1/3), q(2/3).
    """
    v = vol63_series.to_numpy()
    v_train = v[train_mask]
    v_train = v_train[np.isfinite(v_train)]
    if len(v_train) < 100:
        raise RuntimeError("Not enough training vol_63 values to build regime buckets.")
    q1 = np.quantile(v_train, 1/3)
    q2 = np.quantile(v_train, 2/3)
    return np.array([q1, q2], dtype=np.float64)

def assign_vol_bucket(vol63_value: float, edges: np.ndarray) -> int:
    """
    edges length 2 => 3 buckets:
      bucket 0: <= edges[0]
      bucket 1: (edges[0], edges[1]]
      bucket 2: > edges[1]
    """
    if not np.isfinite(vol63_value):
        # conservative: unknown vol treated as middle bucket
        return 1
    if vol63_value <= edges[0]:
        return 0
    if vol63_value <= edges[1]:
        return 1
    return 2


# Supervised Dataset 
def make_supervised_with_regime(
    df: pd.DataFrame,
    assets: List[str],
    lags: int,
    regime_feats: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      X: (n_samples, lags*n_assets + n_regime_feats)
      y: (n_samples, n_assets)
      vol63_at_t: (n_samples,) vol_63 value aligned to sample time t (the last day in the lag window)
    """
    R = df[assets].to_numpy(dtype=np.float32)
    Z = regime_feats[REGIME_FEATURE_ORDER].to_numpy(dtype=np.float32)

    X, y, vol63_t = [], [], []
    # sample index i corresponds to:
    #   input window ends at i-1 (inclusive), and we predict y at i+1 (as in original code)
    # original: for i in range(LAGS_DAYS, len(R) - 1):
    for i in range(lags, len(R) - 1):
        # returns window ends at i-1
        ret_win = R[i - lags:i].reshape(-1)
        # regime features at time i-1 (last observed day)
        z_t = Z[i - 1]
        X.append(np.concatenate([ret_win, z_t], axis=0))
        y.append(R[i + 1])

        # volatility regime assignment should use vol_63 at time i-1 as well
        vol63_t.append(float(regime_feats["vol_63"].iloc[i - 1]))

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
        np.asarray(vol63_t, dtype=np.float64),
    )


# Model + Training
def build_model(
    input_dim: int,
    output_dim: int,
    hidden_sizes: List[int],
    dropout: float,
    lr: float,
    seed: int
) -> tf.keras.Model:
    tf.keras.utils.set_random_seed(seed)

    layers = [tf.keras.layers.Input(shape=(input_dim,))]
    for h in hidden_sizes:
        layers.append(tf.keras.layers.Dense(int(h), activation="relu"))
        layers.append(tf.keras.layers.Dropout(float(dropout)))
    layers.append(tf.keras.layers.Dense(output_dim))

    model = tf.keras.Sequential(layers)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(lr)),
        loss="mse",
        metrics=["mae"]
    )
    return model

def try_import_optuna():
    try:
        import optuna  # type: ignore
        return optuna
    except Exception:
        return None

def tune_hyperparams(
    X: np.ndarray,
    y: np.ndarray,
    base_seed: int,
    n_trials: int
) -> Dict:
    """
    Self-tuning over: learning rate, dropout, hidden layer layout.
    Objective: validation loss (fast, robust baseline).
    """
    n = len(X)
    train_end = int(TRAIN_PCT * n)
    val_end = int((TRAIN_PCT + VAL_PCT) * n)

    scaler = StandardScaler().fit(X[:train_end])
    Xs = scaler.transform(X).astype(np.float32)

    optuna = try_import_optuna()

    def eval_config(hidden_sizes, dropout, lr, seed) -> float:
        model = build_model(X.shape[1], y.shape[1], hidden_sizes, dropout, lr, seed)
        es = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=max(3, PATIENCE // 2),
            restore_best_weights=True
        )
        hist = model.fit(
            Xs[:train_end], y[:train_end],
            validation_data=(Xs[train_end:val_end], y[train_end:val_end]),
            epochs=TUNE_EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[es],
            verbose=0
        )
        return float(np.min(hist.history["val_loss"]))

    if optuna is not None:
        def objective(trial):
            lr = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
            dropout = trial.suggest_float("dropout", 0.0, 0.30)
            hs = trial.suggest_categorical("hidden_sizes", TUNE_HIDDEN_CANDIDATES)
            seed = base_seed + trial.number + 1000
            return eval_config(hs, dropout, lr, seed)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        best = study.best_params
        return {
            "learning_rate": float(best["learning_rate"]),
            "dropout": float(best["dropout"]),
            "hidden_sizes": list(map(int, best["hidden_sizes"])),
        }

    # Fallback deterministic random search
    rng = np.random.default_rng(base_seed + 777)
    best_loss = np.inf
    best_cfg = None

    for t in range(n_trials):
        lr = float(10 ** rng.uniform(np.log10(1e-4), np.log10(5e-3)))
        dropout = float(rng.uniform(0.0, 0.30))
        hs = TUNE_HIDDEN_CANDIDATES[int(rng.integers(0, len(TUNE_HIDDEN_CANDIDATES)))]
        loss = eval_config(hs, dropout, lr, base_seed + t + 1000)

        if loss < best_loss:
            best_loss = loss
            best_cfg = {"learning_rate": lr, "dropout": dropout, "hidden_sizes": list(map(int, hs))}

    assert best_cfg is not None
    return best_cfg

def save_model_meta(meta: Dict):
    with open(MODEL_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

def load_model_meta() -> Optional[Dict]:
    if not os.path.exists(MODEL_META_PATH):
        return None
    try:
        with open(MODEL_META_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def bootstrap_indices(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=n, endpoint=False)

def train_or_load_ensemble(
    X: np.ndarray,
    y: np.ndarray,
    assets: List[str],
    retrain: bool,
    ensemble_k: int,
    hidden_sizes: List[int],
    dropout: float,
    lr: float,
    base_seed: int
) -> Tuple[List[tf.keras.Model], StandardScaler, Dict]:
    """
    Trains/loads:
      - K models (voting ensemble)
      - a scaler for X (includes regime features)
      - signature/meta dict
    """
    n = len(X)
    train_end = int(TRAIN_PCT * n)
    val_end = int((TRAIN_PCT + VAL_PCT) * n)

    current_sig = make_signature(
        assets=assets,
        lags=LAGS_DAYS,
        input_dim=X.shape[1],
        output_dim=y.shape[1],
        hidden_sizes=hidden_sizes,
        dropout=dropout,
        lr=lr,
        ensemble_k=ensemble_k,
        regime_feature_order=REGIME_FEATURE_ORDER,
        n_vol_buckets=N_VOL_BUCKETS
    )

    scaler = StandardScaler().fit(X[:train_end])
    Xs = scaler.transform(X).astype(np.float32)

    # Attempt load
    if (not retrain) and os.path.exists(SCALER_PATH):
        meta = load_model_meta()
        if meta is not None and signature_matches(meta.get("signature", {}), current_sig):
            try:
                saved = np.load(SCALER_PATH, allow_pickle=True).item()
                if not signature_matches(saved.get("signature", {}), current_sig):
                    raise RuntimeError("Scaler signature mismatch.")

                scaler = StandardScaler()
                scaler.mean_ = saved["mean_"]
                scaler.scale_ = saved["scale_"]
                scaler.var_ = scaler.scale_ ** 2
                scaler.n_features_in_ = len(scaler.mean_)

                models = []
                for k in range(ensemble_k):
                    path = os.path.join(MODELS_DIR, f"model_{k}.keras")
                    if not os.path.exists(path):
                        raise RuntimeError(f"Missing model file: {path}")
                    models.append(tf.keras.models.load_model(path))

                print("\nLoaded saved ensemble + scaler (signature match).")
                return models, scaler, current_sig
            except Exception as e:
                print(f"\nFailed to load ensemble/scaler ({e}) — retraining.")

    # Train
    models: List[tf.keras.Model] = []
    hist_agg = []

    for k in range(ensemble_k):
        seed_k = base_seed + 100 * (k + 1)

        model = build_model(
            input_dim=X.shape[1],
            output_dim=y.shape[1],
            hidden_sizes=hidden_sizes,
            dropout=dropout,
            lr=lr,
            seed=seed_k
        )

        es = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True
        )

        if BOOTSTRAP_TRAIN_FOR_ENSEMBLE:
            idx = bootstrap_indices(train_end, seed=seed_k + 555)
            X_train = Xs[idx]
            y_train = y[idx]
        else:
            X_train = Xs[:train_end]
            y_train = y[:train_end]

        history = model.fit(
            X_train, y_train,
            validation_data=(Xs[train_end:val_end], y[train_end:val_end]),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[es],
            verbose=1
        )

        models.append(model)
        hist_agg.append(pd.DataFrame(history.history).assign(model=k))
        model.save(os.path.join(MODELS_DIR, f"model_{k}.keras"))

    # Plot per-model curves
    hist_all = pd.concat(hist_agg, ignore_index=True)
    plt.figure()
    for k in range(ensemble_k):
        h = hist_all[hist_all["model"] == k]
        plt.plot(h["loss"].to_numpy(), alpha=0.7, label=f"train_{k}")
        plt.plot(h["val_loss"].to_numpy(), alpha=0.7, linestyle="--", label=f"val_{k}")
    plt.title("Ensemble training vs validation loss")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "tf_ensemble_loss.png"), dpi=150)
    plt.close()

    np.save(
        SCALER_PATH,
        {"mean_": scaler.mean_, "scale_": scaler.scale_, "signature": current_sig},
        allow_pickle=True
    )
    save_model_meta({
        "signature": current_sig,
        "ensemble_k": ensemble_k,
        "hidden_sizes": hidden_sizes,
        "dropout": dropout,
        "learning_rate": lr,
    })

    print("\nTrained and saved ensemble + scaler (new signature).")
    return models, scaler, current_sig


# Regime-bucketed Residual Pools
def build_or_load_residual_buckets(
    models: List[tf.keras.Model],
    scaler: StandardScaler,
    X: np.ndarray,
    y: np.ndarray,
    vol63_at_t: np.ndarray,
    retrain: bool,
    signature: Dict
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Returns:
      resid_buckets: list of length 3, each array shape (n_resid_bucket, n_assets)
      vol_edges: array([q33, q66]) used to assign buckets
    """
    n = len(X)
    train_end = int(TRAIN_PCT * n)

    if (not retrain) and os.path.exists(RESID_PATH):
        try:
            saved = np.load(RESID_PATH, allow_pickle=True).item()
            if not signature_matches(saved.get("signature", {}), signature):
                raise RuntimeError("Residual-buckets signature mismatch.")
            vol_edges = saved["vol_edges"].astype(np.float64)
            resid_buckets = [rb.astype(np.float32, copy=False) for rb in saved["resid_buckets"]]
            if len(resid_buckets) != N_VOL_BUCKETS:
                raise RuntimeError("Saved residual bucket count mismatch.")
            print("\nLoaded saved residual buckets (signature match).")
            return resid_buckets, vol_edges
        except Exception as e:
            print(f"\nFailed to load residual buckets ({e}) — rebuilding.")

    # Build vol edges using TRAIN portion only
    train_mask = np.zeros(n, dtype=bool)
    train_mask[:train_end] = True
    vol_edges = build_vol_bucket_edges(pd.Series(vol63_at_t), train_mask=train_mask)

    # Assign buckets
    bucket_ids = np.array([assign_vol_bucket(v, vol_edges) for v in vol63_at_t[:train_end]], dtype=np.int64)

    # Compute residuals for each model
    Xs = scaler.transform(X[:train_end]).astype(np.float32)
    y_train = y[:train_end].astype(np.float32)

    buckets = [[] for _ in range(N_VOL_BUCKETS)]  

    for k, m in enumerate(models):
        yhat = m.predict(Xs, verbose=0).astype(np.float32)
        resid = (y_train - yhat).astype(np.float32)

        # de-mean residuals per asset for stability
        resid = resid - resid.mean(axis=0, keepdims=True)

        for b in range(N_VOL_BUCKETS):
            idx_b = np.where(bucket_ids == b)[0]
            if idx_b.size > 0:
                buckets[b].append(resid[idx_b])

    resid_buckets = []
    for b in range(N_VOL_BUCKETS):
        if len(buckets[b]) == 0:
            resid_buckets.append(np.zeros((0, y.shape[1]), dtype=np.float32))
        else:
            resid_buckets.append(np.concatenate(buckets[b], axis=0).astype(np.float32, copy=False))

    np.save(
        RESID_PATH,
        {"resid_buckets": resid_buckets, "vol_edges": vol_edges, "signature": signature},
        allow_pickle=True
    )
    print("\nBuilt and saved residual buckets (new signature).")
    return resid_buckets, vol_edges


# Portfolio Selection
def sample_weights_with_rf(assets: List[str], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
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

def forecast_mean_daily_ensemble(models: List[tf.keras.Model], scaler: StandardScaler, X_last: np.ndarray) -> np.ndarray:
    Xs = scaler.transform(X_last).astype(np.float32)
    preds = []
    for m in models:
        preds.append(m.predict(Xs, verbose=0)[0].astype(np.float64))
    pred = np.mean(np.stack(preds, axis=0), axis=0)
    pred = np.clip(pred, CLIP_MIN_DAILY, CLIP_MAX_DAILY)
    return pred.astype(np.float32)

def simulate_daily_paths_regime_bucket(
    mean_daily: np.ndarray,
    resid_buckets: List[np.ndarray],
    bucket_id: int,
    n_sims: int,
    n_days: int,
    seed: int
) -> np.ndarray:
    """
    Simulate daily asset returns using mean_daily plus bootstrapped residual vectors
    sampled ONLY from the current regime bucket.

    If the chosen bucket is empty, falls back to pooling all buckets.
    """
    rng = np.random.default_rng(seed)
    n_assets = mean_daily.shape[0]

    bucket = resid_buckets[bucket_id]
    if bucket.shape[0] < 100:
        pooled = [b for b in resid_buckets if b.shape[0] > 0]
        if len(pooled) == 0:
            raise RuntimeError("All residual buckets are empty; cannot simulate.")
        bucket = np.concatenate(pooled, axis=0)

    n_resid = bucket.shape[0]
    idx = rng.integers(0, n_resid, size=(n_sims, n_days), endpoint=False)
    resid_samples = bucket[idx]  # (n_sims, n_days, n_assets)

    mean = mean_daily.reshape(1, 1, n_assets).astype(np.float32)
    sim = mean + resid_samples.astype(np.float32)
    sim = np.clip(sim, CLIP_MIN_DAILY, CLIP_MAX_DAILY)
    return sim.astype(np.float32, copy=False)

def compound_over_days(port_daily: np.ndarray) -> np.ndarray:
    if port_daily.ndim != 3:
        raise ValueError(f"compound_over_days expects 3D array, got shape {port_daily.shape}")
    x = np.clip(port_daily, -0.999999, None)
    return np.expm1(np.sum(np.log1p(x), axis=1))

def var_k_index(n_sims: int, alpha: float) -> int:
    k = int(np.floor(alpha * (n_sims - 1)))
    return max(0, min(k, n_sims - 1))

def fast_var_alpha(r: np.ndarray, alpha: float) -> np.ndarray:
    n_sims = r.shape[0]
    k = var_k_index(n_sims, alpha)
    part = np.partition(r, kth=k, axis=0)
    return part[k, :]

def evaluate_and_select_topk(
    max_loss_1m: float,
    W: np.ndarray,
    sim_daily_assets: np.ndarray
) -> Tuple[pd.DataFrame, np.ndarray]:
    n_sims, n_days, n_assets = sim_daily_assets.shape
    n_ports = W.shape[0]

    if n_days != MONTH_DAYS:
        raise ValueError(f"Simulation days mismatch: got {n_days}, expected {MONTH_DAYS}")

    W64 = W.astype(np.float64, copy=False)
    hhi_all = np.sum(W64 * W64, axis=1)

    lam_v = REWARD_WEIGHTS["lambda_vol"]
    lam_c = REWARD_WEIGHTS["lambda_conc"]

    records = []
    B = 500

    for start in range(0, n_ports, B):
        end = min(start + B, n_ports)
        Wb = W64[start:end]

        rp = np.tensordot(sim_daily_assets, Wb.T, axes=([2], [0]))
        if rp.shape != (n_sims, n_days, end - start):
            raise ValueError(f"Unexpected rp shape {rp.shape}, expected {(n_sims, n_days, end-start)}")

        r1m = compound_over_days(rp)

        p5 = fast_var_alpha(r1m, VAR_ALPHA)
        std = np.std(r1m, axis=0, ddof=0)
        mean = np.mean(r1m, axis=0)

        # reporting quantiles 
        med = np.quantile(r1m, 0.50, axis=0)
        p20 = np.quantile(r1m, 0.20, axis=0)
        p80 = np.quantile(r1m, 0.80, axis=0)

        for j in range(end - start):
            idx = start + j
            if float(p5[j]) < max_loss_1m:
                continue

            reward = float(mean[j]) - lam_v * float(std[j]) - lam_c * float(hhi_all[idx])
            records.append({
                "idx": idx,
                "reward": reward,
                "mean_1m": float(mean[j]),
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
            f"Only {len(feasible)} portfolios satisfy VaR constraint "
            f"(alpha={VAR_ALPHA:.0%}, threshold={max_loss_1m:.2%}). "
            f"Loosen max loss, increase RF_MAX_WEIGHT, increase N_PORTFOLIOS, or increase N_SIMS."
        )

    feasible = feasible.sort_values(["reward", "p5_1m", "std_1m"], ascending=[False, False, True])
    top = feasible.head(TOP_K_TO_SHOW).copy()
    topW = W[top["idx"].astype(int).to_numpy()].astype(np.float64, copy=False)
    return top, topW


# Main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", action="store_true", help="Force retrain ensemble")
    parser.add_argument("--tune", action="store_true", help="Run self-tuning before training ensemble")
    parser.add_argument("--n_trials", type=int, default=TUNE_DEFAULT_TRIALS, help="Tuning trials")
    parser.add_argument("--ensemble_k", type=int, default=ENSEMBLE_K, help="Ensemble size K")
    parser.add_argument("--seed", type=int, default=SEED, help="Global seed")
    args, _unknown = parser.parse_known_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    ensure_dirs()

    max_loss_1m = prompt_max_monthly_loss()
    print(f"\nConstraint: VaR_{int(VAR_ALPHA*100)}% (simulated 1-month return) >= {max_loss_1m:.2%}")

    df, assets = load_data_with_rf()

    # Regime features computed on the full historical panel
    regime_feats = compute_regime_features(df, assets)

    # Build supervised dataset with regime conditioning
    X, y, vol63_at_t = make_supervised_with_regime(df, assets, lags=LAGS_DAYS, regime_feats=regime_feats)

    # Self-tuning
    global LEARNING_RATE, DROPOUT_RATE, HIDDEN_SIZES
    if args.tune:
        print(f"\nTuning hyperparameters ({args.n_trials} trials)...")
        best = tune_hyperparams(X, y, base_seed=args.seed, n_trials=args.n_trials)
        LEARNING_RATE = best["learning_rate"]
        DROPOUT_RATE = best["dropout"]
        HIDDEN_SIZES = best["hidden_sizes"]
        print("\nBest tuned hyperparameters:")
        print(f"  learning_rate: {LEARNING_RATE}")
        print(f"  dropout:       {DROPOUT_RATE}")
        print(f"  hidden_sizes:  {HIDDEN_SIZES}")

    # Train or load ensemble + scaler
    models, scaler, signature = train_or_load_ensemble(
        X=X,
        y=y,
        assets=assets,
        retrain=args.retrain,
        ensemble_k=int(args.ensemble_k),
        hidden_sizes=HIDDEN_SIZES,
        dropout=DROPOUT_RATE,
        lr=LEARNING_RATE,
        base_seed=args.seed
    )

    # Build/load 3 regime-bucketed residual pools 
    resid_buckets, vol_edges = build_or_load_residual_buckets(
        models=models,
        scaler=scaler,
        X=X,
        y=y,
        vol63_at_t=vol63_at_t,
        retrain=args.retrain,
        signature=signature
    )

    
    vol63_last = float(regime_feats["vol_63"].iloc[-1])
    bucket_now = assign_vol_bucket(vol63_last, vol_edges)
    print(f"\nRegime (vol_63) edges: q33={vol_edges[0]:.6f}, q66={vol_edges[1]:.6f}")
    print(f"Current vol_63={vol63_last:.6f} => bucket {bucket_now} of {N_VOL_BUCKETS} (0=low,1=mid,2=high)")

    # Forecast next-day mean 
    X_last = X[-1:].astype(np.float32)
    mean_daily = forecast_mean_daily_ensemble(models, scaler, X_last)

    # Simulate forward 1-month 
    print(f"\nSimulating {N_SIMS} forward paths of {MONTH_DAYS} trading days using regime-bucketed residuals...")
    sim_daily_assets = simulate_daily_paths_regime_bucket(
        mean_daily=mean_daily,
        resid_buckets=resid_buckets,
        bucket_id=bucket_now,
        n_sims=N_SIMS,
        n_days=MONTH_DAYS,
        seed=args.seed + 123
    )

    # Sample candidate portfolios
    print(f"\nSampling {N_PORTFOLIOS} long-only portfolios (MAX_WEIGHT={MAX_WEIGHT:.0%}, RF_MAX_WEIGHT={RF_MAX_WEIGHT:.0%})...")
    W = sample_weights_with_rf(assets, seed=args.seed + 999)
    print(f"Generated W: {W.shape}, {W.nbytes/1e6:.2f} MB")

    # Evaluate and select top-k
    top_df, top_W = evaluate_and_select_topk(
        max_loss_1m=max_loss_1m,
        W=W,
        sim_daily_assets=sim_daily_assets
    )

    print(f"\nTOP {TOP_K_TO_SHOW} feasible portfolios (sorted by Reward):")
    print(top_df[[
        "idx", "reward", "mean_1m",
        "p5_1m", "median_1m", "p20_1m", "p80_1m",
        "std_1m", "hhi"
    ]].to_string(index=False))

    best_weights = pd.Series(top_W[0], index=assets).sort_values(ascending=False)
    best_weights_pct = (best_weights * 100.0)
    out = best_weights_pct.head(15).map(lambda x: f"{x:,.2f}%")
    print("\nBEST PORTFOLIO allocation (top 15 weights, %):")
    print(out.to_string())

    print("\nDone.")


if __name__ == "__main__":
    main()
