import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Tuple, Dict, Optional

try:
    import yfinance as yf  # type: ignore
except Exception:
    yf = None  # yfinance may not be available in certain environments

try:
    import plotly.express as px  # type: ignore
except Exception:
    px = None

"""
portfolio_utils.py
------------------

This module provides helper functions for working with user portfolios.
It handles loading/saving portfolio data, fetching current market prices,
computing performance metrics, and preparing data for visualisation.
"""

# Base directory for storing user portfolios
BASE_DIR = os.path.join(os.path.dirname(__file__), "user_data")
PORTFOLIO_DIR = os.path.join(BASE_DIR, "portfolios")


def _ensure_portfolio_dir() -> None:
    """Ensure the portfolios directory exists."""
    if not os.path.exists(PORTFOLIO_DIR):
        os.makedirs(PORTFOLIO_DIR, exist_ok=True)


def list_portfolios(username: str) -> List[str]:
    """
    List portfolio files associated with a given user.

    Parameters
    ----------
    username : str
        The user's username.

    Returns
    -------
    list of str
        Filenames of the user's portfolios sorted by creation time (newest first).
    """
    _ensure_portfolio_dir()
    files = []
    prefix = f"{username}_"
    for fname in os.listdir(PORTFOLIO_DIR):
        if fname.startswith(prefix) and (fname.endswith(".csv") or fname.endswith(".json")):
            files.append(fname)
    # Sort by timestamp extracted from filename (assuming format username_YYYYMMDD_HHMMSS.ext)
    def extract_time(f: str) -> float:
        try:
            base = os.path.splitext(f)[0]
            ts = base[len(prefix):]
            dt = datetime.strptime(ts, "%Y%m%d_%H%M%S")
            return dt.timestamp()
        except Exception:
            return 0.0
    files.sort(key=extract_time, reverse=True)
    return files


def save_portfolio(username: str, df: pd.DataFrame, fmt: str = "csv") -> str:
    """
    Save a DataFrame representing a portfolio to disk for a given user.

    Each time a portfolio is saved a timestamped file is created.  Supported
    formats are CSV and JSON.  The DataFrame should at minimum contain the
    following columns: 'Ticker', 'Purchase Price', 'Quantity', and 'Asset Type'.

    Parameters
    ----------
    username : str
        User's username.  Filenames are prefixed with this.
    df : pandas.DataFrame
        Portfolio data.
    fmt : str, optional
        Format to save: either 'csv' or 'json'.  Default is 'csv'.

    Returns
    -------
    str
        The path to the saved file.
    """
    _ensure_portfolio_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{username}_{timestamp}.{fmt}"
    fpath = os.path.join(PORTFOLIO_DIR, fname)
    if fmt == "csv":
        df.to_csv(fpath, index=False)
    elif fmt == "json":
        df.to_json(fpath, orient="records", indent=2)
    else:
        raise ValueError("Unsupported format for portfolio saving")
    return fpath


def load_portfolio(username: str, filename: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Load a saved portfolio for a given user.

    If filename is None, the latest portfolio (based on timestamp in filename)
    is loaded.  If no portfolio exists, None is returned.

    Parameters
    ----------
    username : str
        User's username.
    filename : str, optional
        The name of a specific portfolio file to load.

    Returns
    -------
    pandas.DataFrame or None
        The loaded portfolio data.
    """
    _ensure_portfolio_dir()
    portfolio_files = list_portfolios(username)
    if not portfolio_files:
        return None
    if filename is None:
        filename = portfolio_files[0]
    fpath = os.path.join(PORTFOLIO_DIR, filename)
    if not os.path.isfile(fpath):
        return None
    _, ext = os.path.splitext(fpath)
    if ext.lower() == ".csv":
        df = pd.read_csv(fpath)
    else:
        df = pd.read_json(fpath)
    return df


def fetch_current_prices(tickers: List[str]) -> Dict[str, float]:
    """
    Retrieve current prices for a list of tickers using yfinance.

    If yfinance is not installed or network requests fail, the returned
    dictionary will have NaNs for missing values.

    Parameters
    ----------
    tickers : list of str
        The asset symbols to fetch.

    Returns
    -------
    dict
        Mapping from ticker to latest price.  Missing tickers map to np.nan.
    """
    prices = {t: np.nan for t in tickers}
    if yf is None:
        return prices
    # Use yfinance to fetch multiple tickers simultaneously for efficiency
    try:
        # yfinance accepts comma‑separated ticker strings
        data = yf.download(tickers=" ".join(tickers), period="1d", interval="1m", progress=False)
        # When multiple tickers are fetched, the DataFrame is MultiIndex with columns like ('Adj Close','AAPL'), etc.
        if isinstance(data.columns, pd.MultiIndex):
            for t in tickers:
                try:
                    # Use the 'Adj Close' column from the last row
                    prices[t] = float(data['Adj Close'][t].dropna().iloc[-1])
                except Exception:
                    prices[t] = np.nan
        else:
            # Single ticker: use Close
            try:
                prices[tickers[0]] = float(data['Adj Close'].dropna().iloc[-1])
            except Exception:
                prices[tickers[0]] = np.nan
    except Exception:
        # Fallback: attempt to use Ticker.fast_info if available
        for t in tickers:
            try:
                ticker_obj = yf.Ticker(t)
                info = ticker_obj.fast_info
                # Some tickers may not have 'lastPrice'; fallback to 'previousClose'
                if info is None:
                    raise Exception
                prices[t] = float(info.get('lastPrice') or info.get('last_price') or info.get('previousClose') or np.nan)
            except Exception:
                prices[t] = np.nan
    return prices


def compute_metrics(df: pd.DataFrame, prices: Dict[str, float]) -> pd.DataFrame:
    """
    Compute portfolio metrics given purchase information and current prices.

    The returned DataFrame includes the following additional columns:

    - Current Price
    - Total Value (Current Price * Quantity)
    - Profit/Loss (%)
    - Profit/Loss (absolute)
    - Weight (%) (percentage of total portfolio value)

    Parameters
    ----------
    df : pandas.DataFrame
        The portfolio DataFrame with columns: 'Ticker', 'Purchase Price', 'Quantity', and 'Asset Type'.
    prices : dict
        Mapping from ticker to current price.

    Returns
    -------
    pandas.DataFrame
        DataFrame with computed metrics.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    # Ensure expected columns exist
    required_cols = ['Ticker', 'Purchase Price', 'Quantity']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    # Map current prices
    df['Current Price'] = df['Ticker'].map(prices)
    # Compute total current value per asset
    df['Total Value'] = df['Current Price'] * df['Quantity']
    # Compute cost basis per asset
    df['Cost Basis'] = df['Purchase Price'] * df['Quantity']
    # Compute absolute profit/loss and percent change
    df['P/L'] = df['Total Value'] - df['Cost Basis']
    df['P/L %'] = np.where(df['Cost Basis'] > 0, (df['Total Value'] / df['Cost Basis'] - 1.0) * 100.0, np.nan)
    # Compute weight in portfolio
    total_portfolio_value = df['Total Value'].sum()
    if total_portfolio_value > 0:
        df['Weight %'] = df['Total Value'] / total_portfolio_value * 100.0
    else:
        df['Weight %'] = np.nan
    return df


def compute_rsi(prices: pd.Series, period: int = 14) -> float:
    """
    Compute the Relative Strength Index (RSI) for a series of prices.

    RSI is a momentum oscillator that measures the magnitude of recent price
    changes to evaluate overbought or oversold conditions.

    Parameters
    ----------
    prices : pandas.Series
        Series of historical prices ordered by date.
    period : int, optional
        The lookback period for RSI.  Default is 14.

    Returns
    -------
    float
        The most recent RSI value.
    """
    if prices is None or len(prices) < period + 1:
        return float('nan')
    delta = prices.diff().dropna()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    # Use exponential moving averages of gains and losses
    avg_gain = gains.ewm(alpha=1.0/period, min_periods=period).mean()
    avg_loss = losses.ewm(alpha=1.0/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])


def compute_volatility(prices: pd.Series) -> float:
    """
    Compute the annualised volatility of a series of prices.

    Volatility is defined as the standard deviation of daily returns multiplied
    by the square root of the number of trading periods in a year (~252).

    Parameters
    ----------
    prices : pandas.Series
        Series of historical prices ordered by date.

    Returns
    -------
    float
        Annualised volatility as a percentage.
    """
    if prices is None or len(prices) < 2:
        return float('nan')
    returns = prices.pct_change().dropna()
    daily_vol = returns.std()
    annual_vol = daily_vol * np.sqrt(252)
    return float(annual_vol * 100.0)


def asset_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise the portfolio by asset type.

    Returns a DataFrame with columns 'Asset Type' and 'Value', showing the
    total value invested in each asset class.

    Parameters
    ----------
    df : pandas.DataFrame
        The portfolio DataFrame after computing metrics.  Must contain
        'Asset Type' and 'Total Value'.

    Returns
    -------
    pandas.DataFrame
        Summary DataFrame by asset class.
    """
    if df is None or df.empty or 'Asset Type' not in df.columns or 'Total Value' not in df.columns:
        return pd.DataFrame()
    summary = df.groupby('Asset Type')['Total Value'].sum().reset_index()
    summary = summary.sort_values('Total Value', ascending=False)
    return summary


def top_and_worst_assets(df: pd.DataFrame, n: int = 3) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Identify the top and worst performing assets based on percentage returns.

    Parameters
    ----------
    df : pandas.DataFrame
        Portfolio DataFrame with a 'P/L %' column.
    n : int, optional
        Number of assets to return in each category.  Default is 3.

    Returns
    -------
    tuple of (pandas.DataFrame, pandas.DataFrame)
        The first DataFrame contains the top performers, the second the worst performers.
    """
    if df is None or df.empty or 'P/L %' not in df.columns:
        return (pd.DataFrame(), pd.DataFrame())
    sorted_df = df.sort_values('P/L %', ascending=False)
    top_n = sorted_df.head(n)
    worst_n = sorted_df.tail(n).iloc[::-1]
    return top_n, worst_n


def suggest_diversification(df: pd.DataFrame) -> Optional[str]:
    """
    Provide a simple diversification suggestion based on asset allocation.

    If a single asset class represents more than 70% of the portfolio value,
    recommend diversification.

    Parameters
    ----------
    df : pandas.DataFrame
        Portfolio DataFrame with computed 'Weight %' and 'Asset Type' columns.

    Returns
    -------
    str or None
        A suggestion string if diversification is needed; otherwise None.
    """
    if df is None or df.empty or 'Asset Type' not in df.columns or 'Weight %' not in df.columns:
        return None
    breakdown = df.groupby('Asset Type')['Weight %'].sum()
    max_type = breakdown.idxmax()
    max_weight = breakdown.max()
    if max_weight > 70:
        return (
            f"La mayor parte de tu cartera (≈{max_weight:.1f}%) está en {max_type}. "
            "Considera diversificar hacia otras clases de activos."
        )
    return None
    
CACHE_DURATION_MINUTES = 1

def get_cached_prices(tickers: List[str], cache_duration_minutes: int = CACHE_DURATION_MINUTES) -> Dict[str, float]:
    """
    Get current prices with intelligent caching system.

    Args:
        tickers: List of ticker symbols
        cache_duration_minutes: Cache validity period

    Returns:
        Dictionary mapping tickers to current prices
    """
    if not tickers or not YF_AVAILABLE:
        return {ticker: np.nan for ticker in tickers}

    now = time.time()
    cache_key = ','.join(sorted(tickers))

    if (cache_key in PRICE_CACHE and
        now - CACHE_TIMESTAMPS.get(cache_key, 0) < cache_duration_minutes * 60):
        return PRICE_CACHE[cache_key]

    # Fetch prices and update cache
    prices = fetch_current_prices(tickers)
    PRICE_CACHE[cache_key] = prices
    CACHE_TIMESTAMPS[cache_key] = now
    return prices
