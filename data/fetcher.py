import hashlib
import logging
import pickle
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

TICKERS = [
    "^GSPC", "^IXIC", "^VIX",
    "GC=F", "CL=F", "HG=F",
    "SPY", "QQQ", "TLT", "GLD", "XLE", "GSG", "HYG",
]

CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_TTL_HOURS = 24


def fetch_prices(
    start: str = "2004-01-01",
    end: str | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Returns wide DataFrame of adjusted Close prices.
    Columns = TICKERS, DatetimeIndex (business days).
    Uses 24h pickle cache keyed on (start, end).
    """
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _cache_path(start, end)

    if not force_refresh:
        cached = _load_from_cache(cache_path)
        if cached is not None:
            return cached

    return _download_and_cache(start, end, cache_path)


def _cache_path(start: str, end: str) -> Path:
    key = hashlib.md5(f"{start}_{end}_{'_'.join(sorted(TICKERS))}".encode()).hexdigest()
    return CACHE_DIR / f"prices_{key}.pkl"


def _load_from_cache(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
    if age > timedelta(hours=CACHE_TTL_HOURS):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _download_and_cache(start: str, end: str, path: Path) -> pd.DataFrame:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = yf.download(
                TICKERS,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=False,
            )

        # yfinance returns MultiIndex columns; extract Close prices
        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"].copy()
        else:
            prices = raw[["Close"]].copy()

        # Ensure all expected tickers are present
        missing = [t for t in TICKERS if t not in prices.columns]
        if missing:
            logger.warning("Missing tickers from download: %s", missing)
            for t in missing:
                prices[t] = float("nan")

        prices = prices[TICKERS]
        prices = prices.ffill().bfill()
        prices.index = pd.to_datetime(prices.index)

        with open(path, "wb") as f:
            pickle.dump(prices, f)

        return prices

    except Exception as exc:
        logger.error("Download failed: %s. Trying stale cache.", exc)
        stale = _load_stale_cache(path)
        if stale is not None:
            return stale
        raise RuntimeError(f"Cannot fetch data and no cache available: {exc}") from exc


def _load_stale_cache(path: Path) -> pd.DataFrame | None:
    if path.exists():
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    # Try any existing cache file for fallback
    for p in CACHE_DIR.glob("prices_*.pkl"):
        try:
            with open(p, "rb") as f:
                return pickle.load(f)
        except Exception:
            continue
    return None
