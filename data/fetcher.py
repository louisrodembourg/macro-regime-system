import hashlib
import logging
import pickle
import time
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
        prices.index = pd.to_datetime(prices.index)

        # Retry tickers that came back all-NaN (yfinance rate-limit on cold start)
        prices = _retry_failed_tickers(prices, start, end)

        prices = prices.ffill().bfill()

        with open(path, "wb") as f:
            pickle.dump(prices, f)

        return prices

    except Exception as exc:
        logger.error("Download failed: %s. Trying stale cache.", exc)
        stale = _load_stale_cache(path)
        if stale is not None:
            return stale
        raise RuntimeError(f"Cannot fetch data and no cache available: {exc}") from exc


def _retry_failed_tickers(prices: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Re-download any ticker whose column is entirely NaN (rate-limit victim)."""
    failed = [t for t in prices.columns if prices[t].isna().all()]
    if not failed:
        return prices

    logger.warning("Retrying %d rate-limited tickers individually: %s", len(failed), failed)
    for i, ticker in enumerate(failed):
        if i > 0:
            time.sleep(3)  # back off between individual requests
        for attempt in range(3):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    single = yf.download(
                        ticker, start=start, end=end,
                        auto_adjust=True, progress=False, threads=False,
                    )
                if single.empty:
                    raise ValueError("empty response")
                close = single["Close"] if "Close" in single.columns else single.iloc[:, 0]
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                close.index = pd.to_datetime(close.index)
                prices[ticker] = close.reindex(prices.index)
                logger.info("Retry succeeded for %s (attempt %d)", ticker, attempt + 1)
                break
            except Exception as exc:
                logger.warning("Retry %d failed for %s: %s", attempt + 1, ticker, exc)
                if attempt < 2:
                    time.sleep(5 * (attempt + 1))

    return prices


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
