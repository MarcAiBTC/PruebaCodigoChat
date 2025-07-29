import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Tuple, Dict, Optional
import time
import logging

try:
    import yfinance as yf
except Exception:
    yf = None

try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:
    px = None
    go = None

# ✅ Reemplazo seguro de __file__
BASE_DIR = os.path.abspath("user_data")
PORTFOLIO_DIR = os.path.join(BASE_DIR, "portfolios")

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache precios
_price_cache = {}
_cache_timestamps = {}
CACHE_DURATION_MINUTES = 5


def _ensure_portfolio_dir() -> None:
    if not os.path.exists(PORTFOLIO_DIR):
        os.makedirs(PORTFOLIO_DIR, exist_ok=True)


def list_portfolios(username: str) -> List[str]:
    _ensure_portfolio_dir()
    files = []
    prefix = f"{username}_"
    for fname in os.listdir(PORTFOLIO_DIR):
        if fname.startswith(prefix) and (fname.endswith(".csv") or fname.endswith(".json")):
            files.append(fname)

    def extract_time(f: str) -> float:
        try:
            base = os.path.splitext(f)[0]
            if "_current" in base:
                return float('inf')
            ts = base[len(prefix):]
            dt = datetime.strptime(ts, "%Y%m%d_%H%M%S")
            return dt.timestamp()
        except Exception:
            return 0.0

    files.sort(key=extract_time, reverse=True)
    return files


def save_portfolio(username: str, df: pd.DataFrame, fmt: str = "csv", overwrite: bool = False) -> str:
    _ensure_portfolio_dir()
    if df is None or df.empty:
        raise ValueError("No se puede guardar una cartera vacía.")

    required_cols = {'Ticker', 'Purchase Price', 'Quantity', 'Asset Type'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Faltan columnas necesarias: {required_cols}")

    if overwrite:
        fname = f"{username}_current.{fmt}"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"{username}_{timestamp}.{fmt}"

    fpath = os.path.join(PORTFOLIO_DIR, fname)
    df = df.copy()
    df = df.dropna(subset=['Purchase Price', 'Quantity'])
    df['Ticker'] = df['Ticker'].astype(str)
    df['Purchase Price'] = pd.to_numeric(df['Purchase Price'], errors='coerce')
    df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
    df['Asset Type'] = df['Asset Type'].astype(str)

    if fmt == "csv":
        df.to_csv(fpath, index=False, encoding='utf-8')
    elif fmt == "json":
        df.to_json(fpath, orient="records", indent=2)
    else:
        raise ValueError("Formato no soportado")

    return fpath


def load_portfolio(username: str, filename: Optional[str] = None) -> Optional[pd.DataFrame]:
    _ensure_portfolio_dir()

    if filename is None:
        current_file = f"{username}_current.csv"
        if os.path.exists(os.path.join(PORTFOLIO_DIR, current_file)):
            filename = current_file
        else:
            files = list_portfolios(username)
            if not files:
                return None
            filename = files[0]

    fpath = os.path.join(PORTFOLIO_DIR, filename)
    if not os.path.isfile(fpath):
        return None

    _, ext = os.path.splitext(fpath)
    df = pd.read_csv(fpath) if ext == ".csv" else pd.read_json(fpath)
    if df.empty:
        return None

    df = df.dropna(subset=['Purchase Price', 'Quantity'])
    df = df[df['Purchase Price'] > 0]
    df = df[df['Quantity'] > 0]
    df['Ticker'] = df['Ticker'].astype(str).str.upper()
    df['Purchase Price'] = pd.to_numeric(df['Purchase Price'], errors='coerce')
    df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
    df['Asset Type'] = df['Asset Type'].astype(str)

    return df


def fetch_current_prices(tickers: List[str]) -> Dict[str, float]:
    if not tickers:
        return {}

    prices = {t: np.nan for t in tickers}
    if yf is None:
        return prices

    cache_key = ','.join(sorted(tickers))
    now = time.time()
    if cache_key in _price_cache and now - _cache_timestamps.get(cache_key, 0) < CACHE_DURATION_MINUTES * 60:
        return _price_cache[cache_key]

    try:
        data = yf.download(" ".join(tickers), period="1d", interval="1m", progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            for t in tickers:
                try:
                    prices[t] = float(data['Adj Close'][t].dropna().iloc[-1])
                except:
                    continue
        elif len(tickers) == 1:
            prices[tickers[0]] = float(data['Adj Close'].dropna().iloc[-1])
    except:
        pass

    for t in tickers:
        if pd.isna(prices[t]):
            try:
                hist = yf.Ticker(t).history(period="1d")
                if not hist.empty:
                    prices[t] = float(hist['Close'].iloc[-1])
            except:
                continue

    _price_cache[cache_key] = prices
    _cache_timestamps[cache_key] = now
    return prices


def compute_metrics(df: pd.DataFrame, prices: Dict[str, float]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    df['Current Price'] = df['Ticker'].map(prices.get)
    df['Total Value'] = df['Current Price'] * df['Quantity']
    df['Cost Basis'] = df['Purchase Price'] * df['Quantity']
    df['P/L'] = df['Total Value'] - df['Cost Basis']
    df['P/L %'] = np.where(df['Cost Basis'] > 0, (df['Total Value'] / df['Cost Basis'] - 1.0) * 100, np.nan)
    total_val = df['Total Value'].sum()
    df['Weight %'] = df['Total Value'] / total_val * 100 if total_val > 0 else np.nan
    return df


def compute_technical_indicators(ticker: str, period: str = "6mo") -> Tuple[float, float]:
    if yf is None:
        return np.nan, np.nan
    try:
        hist = yf.Ticker(ticker).history(period=period)
        if hist.empty or 'Close' not in hist.columns:
            return np.nan, np.nan

        prices = hist['Close'].dropna()
        if len(prices) < 15:
            return np.nan, np.nan

        delta = prices.diff().dropna()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(alpha=1/14, min_periods=14).mean()
        avg_loss = loss.ewm(alpha=1/14, min_periods=14).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        latest_rsi = float(rsi.iloc[-1]) if not rsi.empty else np.nan

        volatility = prices.pct_change().dropna().std() * np.sqrt(252) * 100
        return latest_rsi, volatility
    except:
        return np.nan, np.nan

        return (
            f"La mayor parte de tu cartera (≈{max_weight:.1f}%) está en {max_type}. "
            "Considera diversificar hacia otras clases de activos."
        )
    return None
