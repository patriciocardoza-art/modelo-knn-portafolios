"""
PROYECTO FINAL INTEGRADOR - Opción 5: Modelo KNN para optimización de riesgo en portafolios
Finanzas computacionales (CO-2067-10) · Profesor: Ángel Roberto Nava Solis
"""
import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

try:
    _HERE = Path(__file__).resolve().parent
except NameError:            # si se ejecuta pegando el código en la consola
    _HERE = Path.cwd()


# ==============================================================================
# 1. CONFIGURACIÓN
# ==============================================================================
class C:
    """Parámetros del experimento. EDITA AQUÍ nombres, fechas, activos, k, costos, etc."""
    ROOT = _HERE
    DATA_DIR = ROOT / "data"
    RESULTS_DIR = ROOT / "results"
    FIG_DIR = RESULTS_DIR / "figures"

    # ---------------------------------------------------------------- Universo de activos
    # 20 acciones de EE. UU. de distintos sectores + 3 ETFs (oro, bonos largos, emergentes).
    TICKERS = [
        "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",   # tecnología / consumo
        "JPM", "BAC", "GS",                         # financiero
        "XOM", "CVX",                               # energía
        "JNJ", "PFE", "UNH",                        # salud
        "PG", "KO", "WMT",                          # consumo defensivo
        "HD", "CAT", "NEE", "LIN",                  # industrial / utilities / materiales
        "GLD", "TLT", "EEM",                        # oro, bonos del Tesoro 20+, emergentes
    ]
    BENCHMARK = "SPY"
    START_DATE = "2017-01-01"
    END_DATE = None          # None -> hasta hoy. Formato 'YYYY-MM-DD' (yfinance excluye el último día).

    # ---------------------------------------------------------------- Ventanas temporales
    VALID_START = "2022-01-01"   # selección de hiperparámetros (NO se usa para evaluar)
    VALID_END = "2022-12-31"
    TEST_START = "2023-01-01"    # periodo fuera de muestra (walk-forward + backtest)

    HORIZON = 21        # horizonte de predicción en días hábiles (~1 mes)
    STEP = 7            # se predice cada 7 días hábiles
    REBAL_EVERY = 3     # se rebalancea cada 3 pasos = 21 días (= HORIZON → objetivos sin traslape)

    # ---------------------------------------------------------------- Hiperparámetros KNN a simular
    K_GRID = [5, 10, 20, 50, 100, 200]
    WEIGHTS_GRID = ["uniform", "distance"]

    # ---------------------------------------------------------------- Portafolio
    N_SELECT = 8              # activos que se seleccionan en cada rebalanceo
    MAX_WEIGHT = 0.25         # tope por activo
    LOOKBACK = 252            # ventana (días) para media/volatilidad/covarianza
    IDEAL_Q_RETURN = 0.90     # punto ideal: cuantil 90% de retorno ...
    IDEAL_Q_VOL = 0.10        # ... y cuantil 10% de volatilidad (en espacio estandarizado)
    RF_ANNUAL = 0.04          # tasa libre de riesgo anual (aprox. T-Bill)
    TC_BPS = 10               # costo de transacción por unidad de rotación (puntos base)
    VOL_TARGET = 0.10         # volatilidad anual objetivo para el overlay de riesgo dinámico
    VOL_WINDOW = 63           # ventana para estimar volatilidad del portafolio en el overlay

    AUTORES = "Diego Valdez Hernandez - 654157; Patricio Cardoza Garcia - 629812"   # "Nombre - matrícula; Nombre - matrícula"
    PROFESOR = "Ángel Roberto Nava Solis"
    MATERIA = "Finanzas computacionales"
    FEATURES = ["ret_5", "ret_21", "ret_63", "vol_21", "vol_63", "dd_63"]
    SEED = 42
    TRADING_DAYS = 252


# ==============================================================================
# 2. DATOS (yfinance)
# ==============================================================================
def _download_yfinance(tickers, start, end):
    import yfinance as yf

    df = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if df is None or len(df) == 0:
        raise RuntimeError("yfinance no devolvió datos (¿sin internet o ticker inválido?).")
    close = df["Close"] if isinstance(df.columns, pd.MultiIndex) else df
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    return close


def _synthetic_prices(tickers, start, end, seed):
    """Precios sintéticos (modelo de un factor + ruido). NO son datos reales."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, end or pd.Timestamp.today().normalize())
    n, m = len(idx), len(tickers)
    market = rng.normal(0.0004, 0.010, n)
    beta = rng.uniform(0.4, 1.4, m)
    alpha = rng.normal(0.0002, 0.0002, m)
    idio = rng.uniform(0.008, 0.02, m)
    rets = market[:, None] * beta[None, :] + alpha + rng.normal(0, 1, (n, m)) * idio
    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(prices, index=idx, columns=tickers)


def load_prices(refresh=False, synthetic=False):
    """Devuelve (precios, fuente). Usa caché en data/prices.csv."""
    tickers = list(dict.fromkeys(C.TICKERS + [C.BENCHMARK]))
    cache = C.DATA_DIR / ("prices_synthetic.csv" if synthetic else "prices.csv")
    C.DATA_DIR.mkdir(parents=True, exist_ok=True)

    if cache.exists() and not refresh:
        prices = pd.read_csv(cache, index_col=0, parse_dates=True)
        return prices, ("SINTÉTICO (solo prueba)" if synthetic else "yfinance (caché)")

    if synthetic:
        prices = _synthetic_prices(tickers, C.START_DATE, C.END_DATE, C.SEED)
        source = "SINTÉTICO (solo prueba)"
    else:
        prices = _download_yfinance(tickers, C.START_DATE, C.END_DATE)
        source = "yfinance"

    prices = prices.dropna(axis=1, how="all").sort_index()
    prices = prices.ffill(limit=3).dropna(how="any")
    missing = set(tickers) - set(prices.columns)
    if missing:
        print(f"[aviso] tickers sin datos y omitidos: {sorted(missing)}")
    prices.to_csv(cache)
    return prices, source


# ==============================================================================
# 3. FEATURES Y TARGET
# ==============================================================================
def _long(df: pd.DataFrame, name: str) -> pd.Series:
    d = df.copy()
    d.index.name = "date"
    d.columns.name = "ticker"
    out = d.reset_index().melt(id_vars="date", value_name=name)
    return out.set_index(["date", "ticker"])[name]


def build_panel(prices: pd.DataFrame, horizon: int = C.HORIZON) -> pd.DataFrame:
    """Panel con features calculadas SOLO con información hasta t y target = log-retorno t -> t+horizon."""
    logp = np.log(prices)
    r1 = logp.diff()
    ann = np.sqrt(C.TRADING_DAYS)
    feats = {
        "ret_5": logp.diff(5),
        "ret_21": logp.diff(21),
        "ret_63": logp.diff(63),
        "vol_21": r1.rolling(21).std() * ann,
        "vol_63": r1.rolling(63).std() * ann,
        "dd_63": prices / prices.rolling(63).max() - 1.0,
        "target": logp.shift(-horizon) - logp,
    }
    panel = pd.concat([_long(v, k) for k, v in feats.items()], axis=1)
    panel = panel.dropna(subset=C.FEATURES).sort_index()
    return panel


# ==============================================================================
# 4. TEST DIEBOLD-MARIANO
# ==============================================================================
def dm_from_loss_diff(d, h: int = 1, hln: bool = True) -> dict:
    """DM a partir de la serie del diferencial de pérdida d_t. `h` = horizonte en pasos de la serie."""
    d = np.asarray(pd.Series(d).dropna(), dtype=float)
    T = len(d)
    if T < 5:
        return {"dm_stat": np.nan, "p_value": np.nan, "T": T, "mean_diff": np.nan}
    dbar = d.mean()
    dc = d - dbar
    # varianza de largo plazo con pesos de Bartlett (garantiza varianza > 0)
    lrv = np.dot(dc, dc) / T
    for k in range(1, h):
        gk = np.dot(dc[k:], dc[:-k]) / T
        lrv += 2 * (1 - k / h) * gk
    lrv = max(lrv, 1e-18)
    dm = dbar / np.sqrt(lrv / T)
    if hln:
        corr = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
        dm *= corr
        p = 2 * (1 - stats.t.cdf(abs(dm), df=T - 1))
    else:
        p = 2 * (1 - stats.norm.cdf(abs(dm)))
    return {"dm_stat": float(dm), "p_value": float(p), "T": int(T), "mean_diff": float(dbar)}


def dm_test(e1, e2, h: int = 1, power: int = 2, hln: bool = True) -> dict:
    """DM directo con errores de pronóstico e1, e2 (pérdida cuadrática power=2 o absoluta power=1)."""
    e1, e2 = np.asarray(e1, float), np.asarray(e2, float)
    return dm_from_loss_diff(np.abs(e1) ** power - np.abs(e2) ** power, h=h, hln=hln)


def interpret(res: dict, name1: str, name2: str, alpha: float = 0.05) -> str:
    """Interpretación en español lista para el informe."""
    if np.isnan(res["dm_stat"]):
        return "Muestra insuficiente para el test DM."
    if res["p_value"] < alpha:
        better = name1 if res["dm_stat"] < 0 else name2
        return (f"Se rechaza H0 (p={res['p_value']:.3f} < {alpha}): {better} es "
                f"estadísticamente más preciso.")
    return (f"No se rechaza H0 (p={res['p_value']:.3f} ≥ {alpha}): no hay evidencia de que "
            f"{name1} y {name2} difieran en precisión predictiva.")


# ==============================================================================
# 5. MODELOS KNN (walk-forward, métricas, hiperparámetros)
# ==============================================================================
def make_positions(dates: pd.DatetimeIndex, start, end=None, step=C.STEP) -> list:
    """Posiciones (en `dates`) de la malla de decisión: una cada `step` días hábiles."""
    pos = np.where(dates >= pd.Timestamp(start))[0]
    if end is not None:
        pos = pos[dates[pos] <= pd.Timestamp(end)]
    return list(pos[::step])


def _model(task: str, k: int, weights: str):
    knn = (KNeighborsRegressor if task == "reg" else KNeighborsClassifier)(
        n_neighbors=k, weights=weights, metric="euclidean")
    return Pipeline([("sc", StandardScaler()), ("knn", knn)])


def walk_forward(panel: pd.DataFrame, dates: pd.DatetimeIndex, positions: list, task: str,
                 k: int, weights: str, horizon: int = C.HORIZON) -> pd.DataFrame:
    """Predice en cada posición re-entrenando solo con datos disponibles a esa fecha."""
    d = panel.index.get_level_values("date")
    rows = []
    for p in positions:
        t, cut = dates[p], dates[p - horizon]
        train = panel.iloc[: d.searchsorted(cut, side="right")].dropna(subset=["target"])
        now = panel.iloc[d.searchsorted(t, side="left"): d.searchsorted(t, side="right")]
        if len(train) < 500 or len(now) == 0:
            continue
        X, Xn = train[C.FEATURES].values, now[C.FEATURES].values
        out = pd.DataFrame(index=now.index)
        out["actual"] = now["target"].values
        if task == "reg":
            m = _model("reg", min(k, len(train)), weights).fit(X, train["target"].values)
            out["pred"] = m.predict(Xn)
            out["base_mean"] = train.groupby("ticker")["target"].mean().reindex(
                now.index.get_level_values("ticker")).fillna(train["target"].mean()).values
            out["base_zero"] = 0.0
        else:
            y = (train["target"].values > 0).astype(int)
            m = _model("clf", min(k, len(train)), weights).fit(X, y)
            proba = m.predict_proba(Xn)
            classes = list(m.named_steps["knn"].classes_)
            out["prob_up"] = proba[:, classes.index(1)] if 1 in classes else 0.0
            out["pred"] = (out["prob_up"] >= 0.5).astype(int)
            out["base_pred"] = int(y.mean() >= 0.5)       # clase mayoritaria del entrenamiento
            out["actual"] = np.where(np.isnan(now["target"].values), np.nan,
                                     (now["target"].values > 0).astype(float))
        rows.append(out)
    return pd.concat(rows) if rows else pd.DataFrame()


# ----------------------------------------------------------------------------- métricas
def reg_metrics(df: pd.DataFrame) -> dict:
    d = df.dropna(subset=["actual"])
    res = {}
    for name, col in [("KNN", "pred"), ("Media histórica", "base_mean"), ("Cero (naive)", "base_zero")]:
        e = d["actual"] - d[col]
        res[name] = {"MSE": float((e ** 2).mean()), "RMSE": float(np.sqrt((e ** 2).mean())),
                     "MAE": float(e.abs().mean())}
    sse_m = ((d["actual"] - d["pred"]) ** 2).sum()
    sse_b = ((d["actual"] - d["base_mean"]) ** 2).sum()
    res["R2_OOS_vs_media"] = float(1 - sse_m / sse_b)
    res["n_obs"] = int(len(d))
    return res


def clf_metrics(df: pd.DataFrame) -> dict:
    d = df.dropna(subset=["actual"])
    y, yh, yb = d["actual"].astype(int), d["pred"].astype(int), d["base_pred"].astype(int)
    cm = confusion_matrix(y, yh, labels=[0, 1])
    return {
        "confusion": cm.tolist(),
        "accuracy": float(accuracy_score(y, yh)),
        "precision": float(precision_score(y, yh, zero_division=0)),
        "recall": float(recall_score(y, yh, zero_division=0)),
        "f1": float(f1_score(y, yh, zero_division=0)),
        "baseline_accuracy": float(accuracy_score(y, yb)),
        "baseline_f1": float(f1_score(y, yb, zero_division=0)),
        "n_obs": int(len(d)),
    }


def dm_regression(df: pd.DataFrame, baseline: str, h: int, power: int = 2) -> dict:
    """DM sobre el diferencial de pérdida promedio por fecha (promedia el corte transversal)."""
    d = df.dropna(subset=["actual"])
    l1 = ((d["actual"] - d["pred"]).abs() ** power).groupby("date").mean()
    l2 = ((d["actual"] - d[baseline]).abs() ** power).groupby("date").mean()
    return dm_from_loss_diff(l1 - l2, h=h)


def dm_classification(df: pd.DataFrame, h: int) -> dict:
    """DM con pérdida 0-1 (error de clasificación) de KNN vs. clase mayoritaria."""
    d = df.dropna(subset=["actual"])
    l1 = (d["actual"] != d["pred"]).astype(float).groupby("date").mean()
    l2 = (d["actual"] != d["base_pred"]).astype(float).groupby("date").mean()
    return dm_from_loss_diff(l1 - l2, h=h)


# ----------------------------------------------------------------------------- simulación de hiperparámetros
def grid_search(panel, dates, positions, task: str) -> pd.DataFrame:
    rows = []
    for w in C.WEIGHTS_GRID:
        for k in C.K_GRID:
            wf = walk_forward(panel, dates, positions, task, k, w)
            if wf.empty:
                continue
            if task == "reg":
                m = reg_metrics(wf)["KNN"]
                rows.append({"k": k, "weights": w, **m})
            else:
                m = clf_metrics(wf)
                rows.append({"k": k, "weights": w, "accuracy": m["accuracy"], "f1": m["f1"]})
    return pd.DataFrame(rows)


# ==============================================================================
# 6. CARTERAS (KNN espacial, KNN predictivo, Markowitz, 1/N)
# ==============================================================================
def cap_weights(w: pd.Series, cap: float) -> pd.Series:
    """Aplica tope por activo y redistribuye el exceso entre los demás (iterativo)."""
    w = w / w.sum()
    for _ in range(50):
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        free = ~over & (w < cap - 1e-12)
        if not free.any():
            break
        w[free] += excess * w[free] / w[free].sum()
    return w / w.sum() if w.sum() > 1 else w


# ------------------------------------------------------------------ coordenadas riesgo-retorno
def risk_return_coords(prices: pd.DataFrame, lookback: int = C.LOOKBACK) -> pd.DataFrame:
    """Coordenadas (media anualizada, volatilidad anualizada) de cada activo en la ventana."""
    r = prices.pct_change().dropna().iloc[-lookback:]
    return pd.DataFrame({"mu": r.mean() * C.TRADING_DAYS,
                         "sigma": r.std() * np.sqrt(C.TRADING_DAYS)})


def _zscore(coords: pd.DataFrame) -> pd.DataFrame:
    return (coords - coords.mean()) / coords.std(ddof=0).replace(0, 1)


def nearest_neighbor_table(coords: pd.DataFrame) -> pd.DataFrame:
    """Vecino más cercano de cada activo (distancia euclidiana en el espacio estandarizado)."""
    z = _zscore(coords)
    nn = NearestNeighbors(n_neighbors=2, metric="euclidean").fit(z.values)
    dist, idx = nn.kneighbors(z.values)
    out = coords.copy()
    out["vecino_mas_cercano"] = z.index[idx[:, 1]]
    out["distancia"] = dist[:, 1]
    return out.sort_values("distancia")


def ideal_point(z: pd.DataFrame) -> np.ndarray:
    return np.array([z["mu"].quantile(C.IDEAL_Q_RETURN), z["sigma"].quantile(C.IDEAL_Q_VOL)])


# ------------------------------------------------------------------ estrategias
def knn_spatial_weights(coords: pd.DataFrame, n_select: int = C.N_SELECT,
                        cap: float = C.MAX_WEIGHT) -> pd.Series:
    """Los N activos más cercanos (Euclidiana) al punto ideal (alto retorno, baja volatilidad).

    Peso proporcional al inverso de la distancia al punto ideal.
    """
    z = _zscore(coords)
    nn = NearestNeighbors(n_neighbors=min(n_select, len(z)), metric="euclidean").fit(z.values)
    dist, idx = nn.kneighbors(ideal_point(z).reshape(1, -1))
    inv = 1.0 / (dist[0] + 1e-6)
    w = pd.Series(inv, index=z.index[idx[0]])
    return cap_weights(w, cap).reindex(coords.index).fillna(0.0)


def knn_predictive_weights(pred: pd.Series, vol: pd.Series, n_select: int = C.N_SELECT,
                           cap: float = C.MAX_WEIGHT) -> pd.Series:
    """Top-N por retorno predicho (KNN) ajustado por volatilidad; solo con predicción > 0.

    Si ningún activo tiene predicción positiva, la cartera queda 100% en efectivo.
    """
    score = (pred / vol.reindex(pred.index)).dropna()
    score = score[pred.reindex(score.index) > 0].nlargest(n_select)
    w = pd.Series(0.0, index=pred.index)
    if score.empty:
        return w
    inv_vol = 1.0 / vol.reindex(score.index)
    w[score.index] = cap_weights(inv_vol, max(cap, 1.0 / len(score)))
    return w


def markowitz_max_sharpe(prices: pd.DataFrame, lookback: int = C.LOOKBACK,
                         cap: float = C.MAX_WEIGHT, rf: float = C.RF_ANNUAL) -> pd.Series:
    """Frontera eficiente de Markowitz: cartera de máximo Sharpe long-only (covarianza Ledoit-Wolf)."""
    r = prices.pct_change().dropna().iloc[-lookback:]
    mu = r.mean().values * C.TRADING_DAYS
    cov = LedoitWolf().fit(r.values).covariance_ * C.TRADING_DAYS
    n = len(mu)
    cap = max(cap, 1.0 / n)

    def neg_sharpe(w):
        return -(w @ mu - rf) / np.sqrt(w @ cov @ w)

    res = minimize(neg_sharpe, np.full(n, 1.0 / n), method="SLSQP", bounds=[(0, cap)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                   options={"maxiter": 300})
    w = res.x if res.success else np.full(n, 1.0 / n)
    return pd.Series(np.clip(w, 0, None), index=r.columns).pipe(lambda s: s / s.sum())


def equal_weight(tickers) -> pd.Series:
    return pd.Series(1.0 / len(tickers), index=list(tickers))


def vol_target_scale(port_returns: pd.Series, target: float = C.VOL_TARGET,
                     window: int = C.VOL_WINDOW) -> float:
    """Gestión de riesgo dinámica: lambda = min(1, sigma_objetivo / sigma_estimada)."""
    if len(port_returns) < 20:
        return 1.0
    sig = port_returns.iloc[-window:].std() * np.sqrt(C.TRADING_DAYS)
    return float(min(1.0, target / sig)) if sig > 0 else 1.0


# ==============================================================================
# 7. BACKTEST
# ==============================================================================
def run_backtest(prices: pd.DataFrame, weights: pd.DataFrame, rf: float = C.RF_ANNUAL,
                 tc_bps: float = C.TC_BPS) -> pd.Series:
    """`weights`: índice = fechas de rebalanceo (decisión al cierre), columnas = tickers (suma <= 1).

    Los retornos se capturan desde el día siguiente a la decisión (sin look-ahead).
    """
    rets = prices.pct_change()
    rf_d = (1 + rf) ** (1 / C.TRADING_DAYS) - 1
    dates = list(weights.index)
    out, prev_w = [], pd.Series(0.0, index=weights.columns)
    for i, t in enumerate(dates):
        t_next = dates[i + 1] if i + 1 < len(dates) else prices.index[-1]
        seg = rets.loc[(rets.index > t) & (rets.index <= t_next), weights.columns].fillna(0.0)
        if seg.empty:
            continue
        w = weights.loc[t].fillna(0.0)
        cash = 1.0 - w.sum()
        turnover = (w - prev_w).abs().sum()
        growth_assets = (1 + seg).cumprod()
        growth_cash = (1 + rf_d) ** np.arange(1, len(seg) + 1)
        value = growth_assets.values @ w.values + cash * growth_cash
        value = pd.Series(value, index=seg.index)
        daily = value.pct_change()
        daily.iloc[0] = value.iloc[0] - 1.0
        daily.iloc[0] -= turnover * tc_bps / 1e4
        out.append(daily)
        prev_w = pd.Series(growth_assets.iloc[-1].values * w.values, index=w.index) / value.iloc[-1]
    return pd.concat(out)


def performance(r: pd.Series, rf: float = C.RF_ANNUAL) -> dict:
    r = r.dropna()
    n = len(r)
    rf_d = (1 + rf) ** (1 / C.TRADING_DAYS) - 1
    wealth = (1 + r).cumprod()
    ann_ret = wealth.iloc[-1] ** (C.TRADING_DAYS / n) - 1
    vol = r.std() * np.sqrt(C.TRADING_DAYS)
    ex = r - rf_d
    sharpe = ex.mean() / r.std() * np.sqrt(C.TRADING_DAYS) if r.std() > 0 else np.nan
    dd_dev = np.sqrt((np.minimum(ex, 0) ** 2).mean())
    sortino = ex.mean() / dd_dev * np.sqrt(C.TRADING_DAYS) if dd_dev > 0 else np.nan
    mdd = (wealth / wealth.cummax() - 1).min()
    return {"ROI acumulado": float(wealth.iloc[-1] - 1), "Retorno anual": float(ann_ret),
            "Volatilidad anual": float(vol), "Sharpe": float(sharpe), "Sortino": float(sortino),
            "Max Drawdown": float(mdd),
            "Calmar": float(ann_ret / abs(mdd)) if mdd < 0 else np.nan}


def drawdown(r: pd.Series) -> pd.Series:
    w = (1 + r.dropna()).cumprod()
    return w / w.cummax() - 1


# ==============================================================================
# 8. GRÁFICAS
# ==============================================================================
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
PALETTE = ["#0b3a6f", "#d1495b", "#2a9d8f", "#e9a23b", "#6c757d", "#7b4fa3", "#3d8bd9"]


def _save(fig, name):
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = C.FIG_DIR / name
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_knn_space(coords, weights, nn_table):
    """Plano riesgo-retorno estandarizado: vecinos más cercanos, punto ideal y activos elegidos."""
    z = _zscore(coords)
    ideal = ideal_point(z)
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for t, row in nn_table.iterrows():
        v = row["vecino_mas_cercano"]
        ax.plot([z.loc[t, "sigma"], z.loc[v, "sigma"]], [z.loc[t, "mu"], z.loc[v, "mu"]],
                color="#bbbbbb", lw=0.8, zorder=1)
    sel = weights[weights > 0].index
    ax.scatter(z["sigma"], z["mu"], s=40, color=PALETTE[4], zorder=2, label="Activos")
    ax.scatter(z.loc[sel, "sigma"], z.loc[sel, "mu"], s=40 + 600 * weights[sel], color=PALETTE[0],
               alpha=0.85, zorder=3, label="Seleccionados (tamaño = peso)")
    ax.scatter(ideal[1], ideal[0], marker="*", s=260, color=PALETTE[1], zorder=4, label="Punto ideal")
    for t in z.index:
        ax.annotate(t, (z.loc[t, "sigma"], z.loc[t, "mu"]), fontsize=7, xytext=(3, 3),
                    textcoords="offset points")
    ax.set_xlabel("Volatilidad anualizada (z-score)")
    ax.set_ylabel("Retorno medio anualizado (z-score)")
    ax.set_title("KNN espacial: vecinos más cercanos y selección de cartera")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
    return _save(fig, "01_espacio_knn.png")


def plot_hyper_heatmap(grid, metric="RMSE", name="02_hiperparametros_rmse.png", title=None):
    piv = grid.pivot(index="weights", columns="k", values=metric)
    fig, ax = plt.subplots(figsize=(7, 2.6))
    im = ax.imshow(piv.values, aspect="auto", cmap="viridis_r" if metric == "RMSE" else "viridis")
    ax.set_xticks(range(len(piv.columns)), piv.columns)
    ax.set_yticks(range(len(piv.index)), piv.index)
    ax.grid(False)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:.4f}", ha="center", va="center", color="w", fontsize=8)
    ax.set_xlabel("k (vecinos)")
    ax.set_title(title or f"Simulación de hiperparámetros KNN — {metric} (validación 2022)")
    fig.colorbar(im, ax=ax)
    return _save(fig, name)


def plot_confusion(cm, name="03_matriz_confusion.png"):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    ax.imshow(cm, cmap="Blues")
    ax.grid(False)
    ax.set_xticks([0, 1], ["Baja", "Sube"])
    ax.set_yticks([0, 1], ["Baja", "Sube"])
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="w" if cm[i, j] > cm.max() / 2 else "k", fontsize=12)
    ax.set_title("Matriz de confusión (KNN clasificador)")
    return _save(fig, name)


def plot_pred_vs_actual(df):
    d = df.dropna(subset=["actual"])
    fig, ax = plt.subplots(figsize=(5, 4.4))
    ax.scatter(d["pred"] * 100, d["actual"] * 100, s=6, alpha=0.25, color=PALETTE[0])
    lim = np.nanpercentile(np.abs(np.r_[d["pred"], d["actual"]]), 99) * 100
    ax.plot([-lim, lim], [-lim, lim], color=PALETTE[1], lw=1, label="Predicción perfecta")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Retorno predicho a 21d (%)")
    ax.set_ylabel("Retorno real a 21d (%)")
    ax.set_title("KNN regresor: predicho vs. real (fuera de muestra)")
    ax.legend()
    return _save(fig, "04_predicho_vs_real.png")


def plot_dm_cumulative(df, h_baseline="base_mean"):
    d = df.dropna(subset=["actual"])
    l1 = ((d["actual"] - d["pred"]) ** 2).groupby("date").mean()
    l2 = ((d["actual"] - d[h_baseline]) ** 2).groupby("date").mean()
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.plot((l1 - l2).cumsum(), color=PALETTE[0])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title("Diferencial de pérdida acumulado (KNN − media histórica)\n"
                 "por debajo de 0 = KNN acumula menos error")
    ax.set_ylabel("Σ (error² KNN − error² base)")
    return _save(fig, "05_dm_diferencial_acumulado.png")


def plot_equity(returns: dict):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for i, (k, r) in enumerate(returns.items()):
        ax.plot((1 + r).cumprod(), label=k, color=PALETTE[i % len(PALETTE)],
                lw=2 if "KNN" in k else 1.2)
    ax.set_title("Valor acumulado de $1 invertido (fuera de muestra)")
    ax.legend(fontsize=7, ncol=2)
    return _save(fig, "06_curva_capital.png")


def plot_drawdowns(returns: dict):
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for i, (k, r) in enumerate(returns.items()):
        ax.plot(drawdown(r) * 100, label=k, color=PALETTE[i % len(PALETTE)], lw=1.2)
    ax.set_title("Drawdown (%)")
    ax.legend(fontsize=7, ncol=2)
    return _save(fig, "07_drawdown.png")


def plot_weights(weights: pd.DataFrame, title="Pesos de la cartera KNN espacial en el tiempo"):
    w = weights.loc[:, (weights > 0).any()]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.stackplot(w.index, w.T.values, labels=w.columns)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(fontsize=6, ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    return _save(fig, "08_pesos_tiempo.png")


# ==============================================================================
# 9. INFORME PDF
# ==============================================================================
NAVY = colors.HexColor("#0b3a6f")


def _register_arial():
    """Registra Arial (Windows/Mac); si no existe usa Liberation Sans (idéntica en métricas) o Helvetica."""
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    cands = [
        ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/ariali.ttf", "C:/Windows/Fonts/arialbi.ttf"),
        ("/Library/Fonts/Arial.ttf", "/Library/Fonts/Arial Bold.ttf", "/Library/Fonts/Arial Italic.ttf", "/Library/Fonts/Arial Bold Italic.ttf"),
        ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
         "/System/Library/Fonts/Supplemental/Arial Italic.ttf", "/System/Library/Fonts/Supplemental/Arial Bold Italic.ttf"),
        ("/usr/share/fonts/truetype/msttcorefonts/Arial.ttf", "/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf",
         "/usr/share/fonts/truetype/msttcorefonts/Arial_Italic.ttf", "/usr/share/fonts/truetype/msttcorefonts/Arial_Bold_Italic.ttf"),
        ("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
         "/usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-BoldItalic.ttf"),
    ]
    for reg, bold, ita, bi in cands:
        if all(Path(x).exists() for x in (reg, bold, ita, bi)):
            for name, f in (("Arial", reg), ("Arial-Bold", bold), ("Arial-Italic", ita), ("Arial-BoldItalic", bi)):
                pdfmetrics.registerFont(TTFont(name, f))
            pdfmetrics.registerFontFamily("Arial", normal="Arial", bold="Arial-Bold",
                                          italic="Arial-Italic", boldItalic="Arial-BoldItalic")
            return "Arial", "Arial-Bold"
    return "Helvetica", "Helvetica-Bold"      # Helvetica es el equivalente estándar de Arial en PDF


FONT, FONT_B = _register_arial()
SS = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=SS["Heading1"], fontName=FONT_B, textColor=NAVY, fontSize=15, spaceBefore=14, spaceAfter=6)
H2 = ParagraphStyle("H2", parent=SS["Heading2"], fontName=FONT_B, textColor=NAVY, fontSize=12, spaceBefore=8, spaceAfter=4)
BODY = ParagraphStyle("B", parent=SS["BodyText"], fontName=FONT, fontSize=11, leading=15, alignment=4)
SMALL = ParagraphStyle("S", parent=BODY, fontSize=9, leading=11.5, textColor=colors.HexColor("#555555"))
WARN = ParagraphStyle("W", parent=BODY, textColor=colors.HexColor("#9b1c1c"),
                      backColor=colors.HexColor("#fdeaea"), borderPadding=6, spaceAfter=8)


# ------------------------------------------------------------------ utilidades
def _eq(latex: str, name: str, size=15) -> Image:
    path = C.FIG_DIR / f"eq_{name}.png"
    fig = plt.figure(figsize=(0.1, 0.1))
    fig.text(0, 0, f"${latex}$", fontsize=size)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.08, transparent=False)
    plt.close(fig)
    from PIL import Image as PILImage
    w, h = PILImage.open(path).size
    scale = min(1.0, (15.5 * cm) / (w / 220 * 72))
    return Image(str(path), width=w / 220 * 72 * scale, height=h / 220 * 72 * scale)


def _img(name: str, width_cm: float) -> Image:
    from PIL import Image as PILImage
    p = C.FIG_DIR / name
    w, h = PILImage.open(p).size
    return Image(str(p), width=width_cm * cm, height=width_cm * cm * h / w)


def _table(data, col_widths=None, font=8):
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), FONT_B), ("FONTNAME", (0, 1), (-1, -1), FONT), ("FONTSIZE", (0, 0), (-1, -1), font),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f5f9")]),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#c9d3e0")),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return t


def _p(x, d=3):
    return f"{x * 100:.{d - 1}f}%"


# ------------------------------------------------------------------ informe
def build_report(s, grid_reg, grid_clf, nn_tab, perf, latest) -> str:
    out = str(C.RESULTS_DIR / "Informe_Tecnico_KNN.pdf")
    doc = SimpleDocTemplate(out, pagesize=letter, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                            title="Informe Técnico – KNN para optimización de riesgo en portafolios")
    E = []
    cfg, rm, cm_, dm = s["config"], s["reg_metricas"], s["clf_metricas"], s["dm"]

    # ---------------- 1. resumen
    perf_d = s["desempeno"]
    best = max(perf_d, key=lambda k: perf_d[k]["Sharpe"])
    E += [Paragraph("1. Resumen ejecutivo", H1),
          Paragraph(
              f"Se construyó un sistema de optimización de carteras basado en <b>K-Nearest Neighbors (KNN)</b> "
              f"sobre un universo de {s['n_activos']} activos (acciones de EE. UU. y ETFs de oro, bonos y "
              f"emergentes). KNN se usa de dos formas: (i) <b>espacial</b>, midiendo la distancia euclidiana de "
              f"cada activo en el plano retorno-volatilidad respecto a un punto ideal, y (ii) <b>predictiva</b>, "
              f"estimando el retorno a {cfg['HORIZON']} días y la dirección (sube/baja). La validación es "
              f"<i>walk-forward</i> fuera de muestra desde {cfg['TEST_START']}, con test Diebold-Mariano contra "
              f"líneas base. En el backtest ({s['rango_backtest'][0]} a {s['rango_backtest'][1]}), la estrategia "
              f"con mayor Sharpe fue <b>{best}</b> ({perf_d[best]['Sharpe']:.2f}).", BODY)]

    # ---------------- 2. datos y metodología
    E += [Paragraph("2. Datos y metodología", H1),
          Paragraph(
              f"<b>Datos.</b> Precios diarios ajustados (yfinance) de {', '.join(s['tickers'])}; benchmark SPY. "
              f"<b>Features</b> (solo información hasta t): {', '.join(cfg['FEATURES'])} — retornos a 5/21/63 días, "
              f"volatilidad anualizada a 21/63 días y drawdown a 63 días. <b>Target:</b> log-retorno a "
              f"{cfg['HORIZON']} días. <b>Particiones:</b> validación {cfg['VALID_START']} a {cfg['VALID_END']} "
              f"(selección de hiperparámetros) y prueba desde {cfg['TEST_START']} (jamás usada para elegir "
              f"parámetros). Cada {cfg['STEP']} días hábiles el modelo se re-entrena con ventana expansiva y una "
              f"brecha de {cfg['HORIZON']} días entre entrenamiento y decisión para evitar <i>look-ahead</i>. "
              f"Se rebalancea cada {cfg['STEP'] * cfg['REBAL_EVERY']} días con costo de {cfg['TC_BPS']} pb por "
              f"rotación, tope de {cfg['MAX_WEIGHT']:.0%} por activo, long-only y efectivo remunerado a "
              f"{cfg['RF_ANNUAL']:.0%}.", BODY)]

    # ---------------- 3. matemática
    E += [Paragraph("3. Especificación matemática", H1),
          Paragraph("<b>3.1 Espacio riesgo-retorno y distancia euclidiana.</b> Cada activo i es un punto "
                    "x<sub>i</sub> = (μ<sub>i</sub>, σ<sub>i</sub>) estandarizado:", BODY),
          _eq(r"\mu_i=\frac{252}{n}\sum_{t=1}^{n}r_{i,t}\qquad "
              r"\sigma_i=\sqrt{252}\sqrt{\frac{1}{n-1}\sum_{t=1}^{n}(r_{i,t}-\bar r_i)^2}", "coords"),
          _eq(r"d(x_i,x_j)=\sqrt{\sum_{m=1}^{p}(x_{i,m}-x_{j,m})^2}", "dist"),
          Paragraph("El <b>punto ideal</b> x* combina el cuantil 90% de retorno y el 10% de volatilidad. La cartera "
                    "espacial toma los N vecinos más cercanos a x* con pesos inversamente proporcionales a la "
                    "distancia (con tope por activo):", BODY),
          _eq(r"w_i=\frac{1/(d(x_i,x^*)+\epsilon)}{\sum_{j\in N_k(x^*)}1/(d(x_j,x^*)+\epsilon)}", "pesos"),
          Paragraph("<b>3.2 KNN predictivo.</b> Regresión (retorno a 21 días) y clasificación (dirección):", BODY),
          _eq(r"\hat y(x)=\frac{\sum_{j\in N_k(x)}\omega_j\,y_j}{\sum_{j\in N_k(x)}\omega_j}"
              r"\qquad \omega_j\in\{1,\;1/d_j\}", "knn"),
          Paragraph("<b>3.3 Métricas de error y clasificación.</b>", BODY),
          _eq(r"\mathrm{MSE}=\frac{1}{T}\sum_t e_t^2\quad \mathrm{RMSE}=\sqrt{\mathrm{MSE}}\quad "
              r"\mathrm{MAE}=\frac{1}{T}\sum_t |e_t|", "err"),
          _eq(r"R^2_{OOS}=1-\frac{\sum_t(y_t-\hat y_t)^2}{\sum_t(y_t-\bar y_{t}^{\,hist})^2}\qquad "
              r"F_1=\frac{2PR}{P+R}", "r2f1"),
          Paragraph("<b>3.4 Test de Diebold-Mariano</b> (con corrección de Harvey-Leybourne-Newbold). "
                    "Con errores e<sub>1,t</sub>, e<sub>2,t</sub> de dos modelos:", BODY),
          _eq(r"d_t=e_{1,t}^2-e_{2,t}^2\qquad "
              r"DM=\frac{\bar d}{\sqrt{\widehat{LRV}(d)/T}}\qquad "
              r"DM^{*}=DM\sqrt{\frac{T+1-2h+h(h-1)/T}{T}}", "dm"),
          Paragraph("H<sub>0</sub>: igual precisión predictiva. DM&lt;0 → el modelo 1 (KNN) es más preciso; "
                    "DM&gt;0 → el modelo 2. La varianza de largo plazo usa pesos de Bartlett; "
                    f"h = {int(-(-cfg['HORIZON'] // cfg['STEP']))} pasos de la malla de decisión. La pérdida de cada "
                    "fecha es el promedio sobre el corte transversal de activos.", BODY),
          Paragraph("<b>3.5 Métricas de cartera.</b>", BODY),
          _eq(r"S=\frac{\bar r_p-r_f}{\sigma_p}\sqrt{252}\qquad "
              r"So=\frac{\bar r_p-r_f}{DD}\sqrt{252},\; DD=\sqrt{\overline{\min(r_p-r_f,0)^2}}", "sharpe"),
          _eq(r"\mathrm{MDD}=\max_t\frac{\max_{s\leq t}V_s-V_t}{\max_{s\leq t}V_s}\qquad "
              r"\lambda_t=\min\left(1,\frac{\sigma^{*}}{\hat\sigma_{p,t}}\right)", "mdd"),
          Paragraph("La última expresión es el <i>overlay</i> de gestión de riesgo dinámica (volatilidad objetivo "
                    f"σ* = {cfg['VOL_TARGET']:.0%}): se reduce la exposición cuando la volatilidad reciente supera "
                    "el objetivo y el resto se mantiene en efectivo. Como referencia se incluye Markowitz "
                    "(máximo Sharpe, covarianza Ledoit-Wolf).", BODY)]

    # ---------------- 4. resultados
    E += [PageBreak(), Paragraph("4. Resultados", H1),
          Paragraph("4.1 Simulación de hiperparámetros (validación)", H2)]
    g = grid_reg.sort_values(["RMSE"]).copy()
    data = [["k", "Pesos", "MSE", "RMSE", "MAE"]] + [
        [int(r.k), r.weights, f"{r.MSE:.5f}", f"{r.RMSE:.5f}", f"{r.MAE:.5f}"] for r in g.itertuples()]
    E += [_table(data, [2 * cm, 3 * cm, 3 * cm, 3 * cm, 3 * cm]), Spacer(1, 6),
          _img("02_hiperparametros_rmse.png", 14),
          Paragraph(f"Configuración elegida (menor RMSE en validación): k = {s['mejor_reg']['k']}, "
                    f"pesos = {s['mejor_reg']['weights']}. Clasificador: k = {s['mejor_clf']['k']}, "
                    f"pesos = {s['mejor_clf']['weights']} (mayor F1).", BODY),
          _img("02b_hiperparametros_f1.png", 14)]

    E.append(Paragraph("4.2 Precisión predictiva fuera de muestra", H2))
    data = [["Modelo", "MSE", "RMSE", "MAE"]] + [
        [k, f"{rm[k]['MSE']:.5f}", f"{rm[k]['RMSE']:.5f}", f"{rm[k]['MAE']:.5f}"]
        for k in ["KNN", "Media histórica", "Cero (naive)"]]
    E += [_table(data, [5 * cm, 3 * cm, 3 * cm, 3 * cm]),
          Paragraph(f"R² fuera de muestra de KNN frente a la media histórica: <b>{rm['R2_OOS_vs_media']:.4f}</b> "
                    f"(positivo = KNN mejora a la media). Observaciones: {rm['n_obs']}.", BODY),
          _img("04_predicho_vs_real.png", 9)]

    E.append(Paragraph("4.3 Test de Diebold-Mariano", H2))
    data = [["Comparación", "Pérdida", "DM*", "p-valor", "T"]]
    for key, lab, loss in [("reg_vs_media", "KNN vs. media histórica", "cuadrática"),
                           ("reg_vs_cero", "KNN vs. pronóstico cero", "cuadrática"),
                           ("reg_vs_media_abs", "KNN vs. media histórica", "absoluta"),
                           ("clf_vs_mayoria", "KNN clasif. vs. clase mayoritaria", "0-1")]:
        r = dm[key]
        data.append([lab, loss, f"{r['dm_stat']:.3f}", f"{r['p_value']:.3f}", r["T"]])
    E.append(_table(data, [6.5 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm, 1.5 * cm]))
    for key in ["reg_vs_media", "reg_vs_cero", "clf_vs_mayoria"]:
        E.append(Paragraph(f"• {dm[key]['texto']}", BODY))
    E.append(_img("05_dm_diferencial_acumulado.png", 13))

    E.append(Paragraph("4.4 Clasificación de dirección: matriz de confusión", H2))
    E += [_img("03_matriz_confusion.png", 7),
          Paragraph(f"Exactitud {_p(cm_['accuracy'])} (línea base mayoritaria {_p(cm_['baseline_accuracy'])}); "
                    f"precisión {cm_['precision']:.3f}; recall {cm_['recall']:.3f}; "
                    f"F1 {cm_['f1']:.3f} (línea base {cm_['baseline_f1']:.3f}).", BODY)]

    E += [PageBreak(), Paragraph("4.5 KNN espacial: coordenadas y vecino más cercano", H2),
          _img("01_espacio_knn.png", 13)]
    nn = nn_tab.head(12)
    data = [["Activo", "Retorno anual", "Volatilidad", "Vecino más cercano", "Distancia"]] + [
        [t, _p(r.mu), _p(r.sigma), r.vecino_mas_cercano, f"{r.distancia:.3f}"] for t, r in nn.iterrows()]
    E += [_table(data, [2.5 * cm, 3 * cm, 3 * cm, 4 * cm, 3 * cm]),
          Paragraph(f"(12 pares más próximos; tabla completa en results/vecinos_mas_cercanos.csv.) "
                    f"Estimado con los últimos {cfg['LOOKBACK']} días al {s['fecha_cartera_actual']}.", SMALL)]

    E.append(Paragraph("4.6 Desempeño de las carteras (backtest fuera de muestra)", H2))
    cols = ["ROI acumulado", "Retorno anual", "Volatilidad anual", "Sharpe", "Sortino", "Max Drawdown"]
    data = [["Estrategia"] + ["ROI acum.", "Ret. anual", "Vol. anual", "Sharpe", "Sortino", "Max DD"]]
    for k, v in perf.iterrows():
        data.append([k, _p(v[cols[0]]), _p(v[cols[1]]), _p(v[cols[2]]),
                     f"{v['Sharpe']:.2f}", f"{v['Sortino']:.2f}", _p(v["Max Drawdown"])])
    E += [_table(data, [4.8 * cm, 2 * cm, 2 * cm, 2 * cm, 1.7 * cm, 1.7 * cm, 2 * cm], font=7.5),
          Paragraph(f"Periodo {s['rango_backtest'][0]} a {s['rango_backtest'][1]}; {s['n_rebalanceos']} "
                    f"rebalanceos; costos de transacción incluidos.", SMALL),
          _img("06_curva_capital.png", 15), _img("07_drawdown.png", 15), _img("08_pesos_tiempo.png", 15)]

    E.append(Paragraph(f"4.7 Cartera sugerida al {s['fecha_cartera_actual']}", H2))
    lw = latest[(latest > 0.005).any(axis=1)]
    data = [["Activo"] + list(lw.columns)] + [[t] + [_p(v) for v in row] for t, row in lw.iterrows()]
    E += [_table(data, [3 * cm] + [3.5 * cm] * len(lw.columns)),
          Paragraph("Resultado académico de un modelo; no constituye recomendación de inversión.", SMALL)]

    # ---------------- 5. riesgo
    ps = perf_d["KNN Espacial"]
    pv = perf_d["KNN Espacial + VolTarget"]
    E += [Paragraph("5. Análisis de riesgo", H1),
          Paragraph(
              f"La cartera KNN espacial obtuvo volatilidad anual de {_p(ps['Volatilidad anual'])} y un máximo "
              f"drawdown de {_p(ps['Max Drawdown'])}. Con el <i>overlay</i> de volatilidad objetivo, "
              f"la volatilidad fue {_p(pv['Volatilidad anual'])} y el drawdown {_p(pv['Max Drawdown'])} "
              f"(Sharpe {pv['Sharpe']:.2f} vs. {ps['Sharpe']:.2f}). El overlay reduce riesgo a cambio de "
              f"menor exposición; su valor depende de si el mercado del periodo tuvo episodios de estrés.", BODY)]

    # ---------------- 6. interpretación Gemini (opcional)
    if s.get("gemini"):
        from xml.sax.saxutils import escape
        E.append(Paragraph("6. Interpretación asistida por IA (Gemini API)", H1))
        for par in s["gemini"].replace("**", "").replace("#", "").split("\n\n"):
            if par.strip():
                E.append(Paragraph(escape(par.strip()).replace("\n", "<br/>"), BODY))
        E.append(Paragraph(f"Texto generado con la API de Gemini (modelo configurado en GEMINI_MODEL) a partir de "
                           f"los resultados de esta ejecución; revisado por el equipo.", SMALL))

    # ---------------- 7. conclusiones
    best_knn = max(["KNN Espacial", "KNN Espacial + VolTarget", "KNN Predictivo"],
                   key=lambda k: perf_d[k]["Sharpe"])
    ref = {k: perf_d[k]["Sharpe"] for k in ["Markowitz (máx. Sharpe)", "Equiponderada 1/N", "SPY (buy & hold)"]}
    wins = [k for k, v in ref.items() if perf_d[best_knn]["Sharpe"] > v]
    E += [Paragraph("7. Conclusiones y limitaciones", H1),
          Paragraph(
              f"• La mejor variante KNN por Sharpe fue <b>{best_knn}</b> ({perf_d[best_knn]['Sharpe']:.2f}); "
              + (f"superó en Sharpe a: {', '.join(wins)}." if wins else
                 "no superó en Sharpe a ninguna de las referencias (Markowitz, 1/N, SPY).") +
              f" Referencias: " + "; ".join(f"{k} {v:.2f}" for k, v in ref.items()) + ".", BODY),
          Paragraph(f"• Predicción: {dm['reg_vs_media']['texto']} Un RMSE menor al de la línea base no basta; "
                    f"el test DM indica si la diferencia es estadísticamente distinguible del azar.", BODY),
          Paragraph("• <b>Limitaciones:</b> (i) el universo se eligió con empresas vigentes hoy "
                    "(sesgo de supervivencia); (ii) un solo periodo de prueba con pocos rebalanceos limita la "
                    "potencia estadística; (iii) los retornos son difíciles de predecir y KNN es sensible a la "
                    "escala y a la dimensionalidad (mitigado con estandarización y un conjunto pequeño de "
                    "features); (iv) costos y deslizamiento simplificados; (v) resultados pasados no garantizan "
                    "resultados futuros.", BODY),
          Paragraph("• <b>Trabajo futuro:</b> ventana de entrenamiento móvil, más features (fundamentales, "
                    "sentimiento NLP), optimización conjunta de k por régimen y ejecución en testnet.", BODY)]

    doc.build(E)
    return out


# ==============================================================================
# 10. GEMINI (opcional)
# ==============================================================================
def _load_env():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass


def _gmodel() -> str:
    return os.getenv("GEMINI_MODEL", "gemini-2.5-flash")   # verifica el modelo vigente en AI Studio


def _client():
    _load_env()
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key or key.startswith("pega_aqui"):
        raise RuntimeError("Falta GEMINI_API_KEY en el archivo .env")
    from google import genai
    return genai.Client(api_key=key)


def has_key() -> bool:
    _load_env()
    k = os.getenv("GEMINI_API_KEY", "").strip()
    return bool(k) and not k.startswith("pega_aqui")


def check_gemini():
    """Verificación previa de la clave (el proyecto exige auditar las API keys antes de exponer)."""
    try:
        r = _client().models.generate_content(model=_gmodel(), contents="Responde solo con la palabra: OK")
        return True, (r.text or "").strip()
    except Exception as e:  # noqa: BLE001
        return False, str(e)[:300]


def _ask(client, prompt: str) -> str:
    resp = client.models.generate_content(model=_gmodel(), contents=prompt)
    return (resp.text or "").strip()


def gemini_interpretation(summary: dict) -> str:
    """Devuelve dos párrafos: interpretación de resultados y comentario de riesgo de la cartera actual."""
    client = _client()
    res = {k: summary[k] for k in ["reg_metricas", "clf_metricas", "dm", "desempeno", "rango_backtest"]
           if k in summary}
    p1 = (
        "Eres analista cuantitativo. Con los siguientes resultados de un proyecto académico (KNN para "
        "optimizar carteras; test Diebold-Mariano; backtest fuera de muestra con costos) redacta en español, "
        "en máximo 170 palabras y en texto plano (sin markdown, sin viñetas): qué estrategia tuvo mejor "
        "relación riesgo/retorno, si KNN superó estadísticamente a la línea base según el test DM (cita el "
        "p-valor) y dos limitaciones. No inventes cifras.\n\n"
        + json.dumps(res, ensure_ascii=False, default=float)[:6000])
    cur = {"fecha": summary.get("fecha_cartera_actual"), "pesos": summary.get("cartera_actual"),
           "riesgo_historico": {k: {m: v[m] for m in ("Volatilidad anual", "Max Drawdown", "Sharpe")}
                                for k, v in summary.get("desempeno", {}).items()
                                if k.startswith("KNN") or k.startswith("Markowitz")}}
    p2 = (
        "Eres gestor de riesgos. Con la cartera sugerida actual y su riesgo histórico, redacta en español, "
        "en máximo 120 palabras y en texto plano (sin markdown): concentración por activo, diferencias entre "
        "KNN espacial, KNN predictivo y Markowitz, y una recomendación de monitoreo. Es un ejercicio "
        "académico, no asesoría financiera. No inventes cifras.\n\n"
        + json.dumps(cur, ensure_ascii=False, default=float)[:4000])
    return _ask(client, p1) + "\n\n" + _ask(client, p2)


# ==============================================================================
# 11. PIPELINE PRINCIPAL
# ==============================================================================
def log(msg):
    print(f"[{pd.Timestamp.now():%H:%M:%S}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true", help="datos sintéticos (solo prueba)")
    ap.add_argument("--refresh", action="store_true", help="forzar nueva descarga")
    ap.add_argument("--no-report", action="store_true", help="no generar el PDF")
    ap.add_argument("--autores", default=C.AUTORES)
    args = ap.parse_known_args()[0]

    C.RESULTS_DIR.mkdir(exist_ok=True)
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Verificación previa de la API key de Gemini (falla rápido, antes de los cálculos largos)

    use_gemini = has_key()
    if use_gemini:
        ok, msg = check_gemini()
        log(f"Gemini API ({_gmodel()}): {'OK' if ok else 'FALLÓ → ' + msg}")
        use_gemini = ok
    else:
        log("Gemini: sin GEMINI_API_KEY en .env → el informe se generará sin la sección de IA.")

    # ------------------------------------------------------------------ 1. datos
    prices, source = load_prices(refresh=args.refresh, synthetic=args.synthetic)
    assets = prices.drop(columns=[C.BENCHMARK])
    tickers = list(assets.columns)
    dates = assets.index
    log(f"Datos: {source} | {len(tickers)} activos | {dates[0]:%Y-%m-%d} → {dates[-1]:%Y-%m-%d}")
    panel = build_panel(assets)

    # ------------------------------------------------------------------ 2. simulación de hiperparámetros (validación)
    pos_val = make_positions(dates, C.VALID_START, C.VALID_END)
    log(f"Simulando hiperparámetros KNN en validación ({len(pos_val)} fechas)…")
    grid_reg = grid_search(panel, dates, pos_val, "reg")
    grid_clf = grid_search(panel, dates, pos_val, "clf")
    best_reg = grid_reg.sort_values("RMSE").iloc[0]
    best_clf = grid_clf.sort_values("f1", ascending=False).iloc[0]
    grid_reg.to_csv(C.RESULTS_DIR / "hiperparametros_regresion.csv", index=False)
    grid_clf.to_csv(C.RESULTS_DIR / "hiperparametros_clasificacion.csv", index=False)
    log(f"Mejor regresor: k={int(best_reg['k'])}, pesos={best_reg['weights']} | "
        f"mejor clasificador: k={int(best_clf['k'])}, pesos={best_clf['weights']}")

    # ------------------------------------------------------------------ 3. walk-forward fuera de muestra
    pos_grid = make_positions(dates, C.TEST_START)
    last = len(dates) - 1
    pos_all = pos_grid + ([last] if pos_grid[-1] != last else [])
    log(f"Walk-forward fuera de muestra ({len(pos_all)} fechas de decisión)…")
    reg = walk_forward(panel, dates, pos_all, "reg", int(best_reg["k"]), best_reg["weights"])
    clf = walk_forward(panel, dates, pos_all, "clf", int(best_clf["k"]), best_clf["weights"])
    reg.to_csv(C.RESULTS_DIR / "predicciones_regresion.csv")
    clf.to_csv(C.RESULTS_DIR / "predicciones_clasificacion.csv")

    h_steps = int(np.ceil(C.HORIZON / C.STEP))   # horizonte medido en pasos de la malla
    reg_m, clf_m = reg_metrics(reg), clf_metrics(clf)
    dm = {
        "reg_vs_media": dm_regression(reg, "base_mean", h_steps),
        "reg_vs_cero": dm_regression(reg, "base_zero", h_steps),
        "reg_vs_media_abs": dm_regression(reg, "base_mean", h_steps, power=1),
        "clf_vs_mayoria": dm_classification(clf, h_steps),
    }
    dm["reg_vs_media"]["texto"] = interpret(dm["reg_vs_media"], "KNN", "la media histórica")
    dm["reg_vs_cero"]["texto"] = interpret(dm["reg_vs_cero"], "KNN", "el pronóstico cero")
    dm["reg_vs_media_abs"]["texto"] = interpret(dm["reg_vs_media_abs"], "KNN", "la media histórica")
    dm["clf_vs_mayoria"]["texto"] = interpret(dm["clf_vs_mayoria"], "KNN", "la clase mayoritaria")
    log(f"RMSE KNN={reg_m['KNN']['RMSE']:.4f} vs media={reg_m['Media histórica']['RMSE']:.4f} | "
        f"DM p={dm['reg_vs_media']['p_value']:.3f} | F1 clf={clf_m['f1']:.3f}")

    # ------------------------------------------------------------------ 4. carteras y backtest
    rebal_pos = pos_grid[::C.REBAL_EVERY]
    rebal_dates = [dates[p] for p in rebal_pos]
    W = {k: [] for k in ["KNN Espacial", "KNN Espacial + VolTarget", "KNN Predictivo",
                         "Markowitz (máx. Sharpe)", "Equiponderada 1/N", f"{C.BENCHMARK} (buy & hold)"]}
    idx_ok = []
    for p, t in zip(rebal_pos, rebal_dates):
        window = assets.iloc[: p + 1]                       # solo información <= t
        coords = risk_return_coords(window)
        w_sp = knn_spatial_weights(coords)
        pr = window.pct_change().iloc[-C.VOL_WINDOW:] @ w_sp
        lam = vol_target_scale(pr)
        try:
            now = panel.xs(t, level="date")
            pred = reg.xs(t, level="date")["pred"]
            w_pr = knn_predictive_weights(pred, now["vol_63"])
        except KeyError:
            continue
        W["KNN Espacial"].append(w_sp)
        W["KNN Espacial + VolTarget"].append(w_sp * lam)
        W["KNN Predictivo"].append(w_pr.reindex(tickers).fillna(0.0))
        W["Markowitz (máx. Sharpe)"].append(markowitz_max_sharpe(window))
        W["Equiponderada 1/N"].append(equal_weight(tickers))
        W[f"{C.BENCHMARK} (buy & hold)"].append(pd.Series({C.BENCHMARK: 1.0}))
        idx_ok.append(t)

    returns, wdfs = {}, {}
    for name, lst in W.items():
        wdf = pd.DataFrame(lst, index=idx_ok).fillna(0.0)
        wdfs[name] = wdf
        returns[name] = run_backtest(prices, wdf)
        wdf.to_csv(C.RESULTS_DIR / f"pesos_{name.split(' ')[0].lower()}_{len(wdfs)}.csv")
    common = None
    for r in returns.values():
        common = r.index if common is None else common.intersection(r.index)
    returns = {k: v.loc[common] for k, v in returns.items()}
    perf = pd.DataFrame({k: performance(v) for k, v in returns.items()}).T
    perf.to_csv(C.RESULTS_DIR / "metricas_carteras.csv")
    pd.DataFrame(returns).to_csv(C.RESULTS_DIR / "retornos_diarios.csv")
    log("Backtest terminado:\n" + perf[["ROI acumulado", "Retorno anual", "Volatilidad anual",
                                         "Sharpe", "Sortino", "Max Drawdown"]].round(3).to_string())

    # ------------------------------------------------------------------ 5. cartera recomendada HOY
    coords_now = risk_return_coords(assets)
    nn_tab = nearest_neighbor_table(coords_now)
    w_sp_now = knn_spatial_weights(coords_now)
    pred_now = reg.xs(dates[last], level="date")["pred"]
    w_pr_now = knn_predictive_weights(pred_now, panel.xs(dates[last], level="date")["vol_63"])
    latest = pd.DataFrame({"KNN Espacial": w_sp_now, "KNN Predictivo": w_pr_now.reindex(tickers),
                           "Markowitz": markowitz_max_sharpe(assets)}).fillna(0.0)
    latest.to_csv(C.RESULTS_DIR / "cartera_actual.csv")
    nn_tab.to_csv(C.RESULTS_DIR / "vecinos_mas_cercanos.csv")

    # ------------------------------------------------------------------ 6. gráficas
    log("Generando gráficas…")
    plot_knn_space(coords_now, w_sp_now, nn_tab)
    plot_hyper_heatmap(grid_reg, "RMSE")
    plot_hyper_heatmap(grid_clf, "f1", "02b_hiperparametros_f1.png",
                          "Simulación de hiperparámetros KNN — F1 (validación 2022)")
    plot_confusion(clf_m["confusion"])
    plot_pred_vs_actual(reg)
    plot_dm_cumulative(reg)
    plot_equity(returns)
    plot_drawdowns(returns)
    plot_weights(wdfs["KNN Espacial"])

    # ------------------------------------------------------------------ 7. resumen
    summary = {
        "fuente_datos": source,
        "sintetico": bool(args.synthetic),
        "autores": args.autores,
        "profesor": C.PROFESOR,
        "materia": C.MATERIA,
        "fecha_ejecucion": f"{pd.Timestamp.now():%Y-%m-%d %H:%M}",
        "n_activos": len(tickers), "tickers": tickers,
        "rango_datos": [f"{dates[0]:%Y-%m-%d}", f"{dates[-1]:%Y-%m-%d}"],
        "rango_backtest": [f"{common[0]:%Y-%m-%d}", f"{common[-1]:%Y-%m-%d}"],
        "n_rebalanceos": len(idx_ok),
        "mejor_reg": {"k": int(best_reg["k"]), "weights": best_reg["weights"],
                      "RMSE_val": float(best_reg["RMSE"])},
        "mejor_clf": {"k": int(best_clf["k"]), "weights": best_clf["weights"],
                      "F1_val": float(best_clf["f1"])},
        "reg_metricas": reg_m, "clf_metricas": clf_m, "dm": dm,
        "desempeno": perf.to_dict(orient="index"),
        "cartera_actual": latest.round(4).to_dict(),
        "fecha_cartera_actual": f"{dates[last]:%Y-%m-%d}",
        "config": {k: getattr(C, k) for k in [
            "HORIZON", "STEP", "REBAL_EVERY", "N_SELECT", "MAX_WEIGHT", "LOOKBACK", "RF_ANNUAL",
            "TC_BPS", "VOL_TARGET", "VALID_START", "VALID_END", "TEST_START", "FEATURES",
            "K_GRID", "WEIGHTS_GRID"]},
    }
    (C.RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False,
                                                           default=float))

    # ------------------------------------------------------------------ 8. informe + (opcional) Gemini
    if use_gemini:
        try:
            summary["gemini"] = gemini_interpretation(summary)
            log("Interpretación de Gemini incluida en el informe.")
        except Exception as e:  # noqa: BLE001
            log(f"[aviso] Gemini falló al generar el texto: {str(e)[:200]}")

    if not args.no_report:

        out = build_report(summary, grid_reg, grid_clf, nn_tab, perf, latest)
        log(f"Informe PDF: {out}")
        try:   # copia también a la carpeta Descargas del usuario
            dl = Path.home() / "Downloads"
            dl.mkdir(exist_ok=True)
            copia = Path(shutil.copy(out, dl / Path(out).name))
            log(f"Copia del PDF en Descargas: {copia}")
        except Exception as e:  # noqa: BLE001
            log(f"[aviso] No se pudo copiar a Descargas: {e}")
    log("Listo.")


if __name__ == "__main__":
    main()
