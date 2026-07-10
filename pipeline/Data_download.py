#!/usr/bin/env python3
"""
================================================================================
 Quant Research Data Acquisition Layer
 López de Prado-style data lake builder for US equity market research.

 Downloads daily market and macroeconomic data from Yahoo Finance and FRED,
 cleans it according to a strict, non-imputing set of rules, and stores it
 as an organized collection of Snappy-compressed Parquet files.

 FRED access:
   Requires a FRED API key (free, takes ~30 seconds):
     https://fred.stlouisfed.org/docs/api/api_key.html
   Set it via the FRED_API_KEY environment variable or --fred-key. The
   script uses ONLY the official fredapi package (Fred.get_series) to talk
   to FRED -- no scraping, no CSV-export endpoints, no undocumented routes.
   If no key is available, the script fails fast before downloading
   anything, rather than silently degrading to a less reliable method.

 Usage:
     export FRED_API_KEY="your_fred_api_key"
     python download_quant_data.py --root ./data --start 2000-01-01

 Requirements:
     pip install yfinance fredapi pandas numpy tqdm pyarrow
================================================================================
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from io import StringIO
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import yfinance as yf
except ImportError:
    yf = None

try:
    from fredapi import Fred
except ImportError:
    Fred = None

try:
    import requests
except ImportError:
    requests = None


# A registry entry can be a single ticker/series string, or a list of
# candidate identifiers to try in order (first success wins). This is used
# for symbols that are inconsistently available across Yahoo Finance's
# backends (e.g. some index tickers get deprecated or region-locked).
Candidates = Union[str, list]


# =============================================================================
# Logging setup
# =============================================================================

def build_logger(log_dir: Path) -> logging.Logger:
    """Configure a logger that writes to both console and a timestamped file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"download_log_{ts}.log"

    logger = logging.getLogger("quant_data")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    logger.info("Log file initialized at %s", log_path)
    return logger


# =============================================================================
# Result tracking
# =============================================================================

@dataclass
class DownloadResult:
    name: str
    category: str
    source: str
    identifier: str
    success: bool
    shape: tuple = (0, 0)
    first_date: Optional[str] = None
    last_date: Optional[str] = None
    missing_values: int = 0
    save_path: Optional[str] = None
    error: Optional[str] = None


@dataclass
class RunSummary:
    results: list = field(default_factory=list)

    def add(self, r: DownloadResult) -> None:
        self.results.append(r)

    @property
    def successes(self) -> list:
        return [r for r in self.results if r.success]

    @property
    def failures(self) -> list:
        return [r for r in self.results if not r.success]

    def print_summary(self, logger: logging.Logger) -> None:
        logger.info("=" * 80)
        logger.info("FINAL DOWNLOAD SUMMARY")
        logger.info("=" * 80)
        logger.info("Total datasets attempted : %d", len(self.results))
        logger.info("Successful downloads      : %d", len(self.successes))
        logger.info("Failed downloads           : %d", len(self.failures))
        if self.failures:
            logger.info("-- Failed datasets --")
            for r in self.failures:
                logger.info("  [%s] %s (%s: %s) -> %s",
                             r.category, r.name, r.source, r.identifier, r.error)
        logger.info("=" * 80)


# =============================================================================
# Cleaning utilities
# =============================================================================

def clean_dataframe(df: pd.DataFrame, name: str, logger: logging.Logger) -> pd.DataFrame:
    """
    Apply the general data-cleaning rules:
      - inspect shape / dtypes
      - print date range and missing values
      - drop 100% empty columns
      - drop duplicate timestamps
      - sort index
      - keep Date as DatetimeIndex
      - NEVER forward/back fill
    """
    if df is None or df.empty:
        raise ValueError("Empty dataframe returned")

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    df.index.name = "Date"

    df = df[~df.index.isna()]

    empty_cols = [c for c in df.columns if df[c].isna().all()]
    if empty_cols:
        df = df.drop(columns=empty_cols)

    if df.shape[1] == 0:
        raise ValueError("All columns were empty after cleaning")

    dup_count = int(df.index.duplicated().sum())
    if dup_count:
        df = df[~df.index.duplicated(keep="first")]

    df = df.sort_index()

    shape = df.shape
    n_missing = int(df.isna().sum().sum())
    first_date = str(df.index.min().date()) if len(df) else None
    last_date = str(df.index.max().date()) if len(df) else None

    logger.info(
        "  [clean] %-28s shape=%s dtypes=%s missing=%d dropped_empty_cols=%d dup_rows=%d range=[%s -> %s]",
        name, shape, dict(df.dtypes.astype(str)), n_missing, len(empty_cols), dup_count,
        first_date, last_date,
    )

    return df


def save_parquet(df: pd.DataFrame, path: Path, logger: logging.Logger) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", compression="snappy")
    logger.info("  [save] -> %s", path)


def as_candidate_list(value: Candidates) -> list:
    return value if isinstance(value, list) else [value]


# =============================================================================
# Downloaders
# =============================================================================

class DataLake:
    def __init__(self, root: Path, start: str, end: Optional[str],
                 fred_api_key: Optional[str], logger: logging.Logger,
                 retries: int = 3, pause: float = 1.5):
        self.root = root
        self.raw = root / "raw"
        self.start = start
        self.end = end or date.today().isoformat()
        self.logger = logger
        self.retries = retries
        self.pause = pause
        self.summary = RunSummary()

        self.fred_api_key = fred_api_key

        if Fred is None:
            raise RuntimeError(
                "The 'fredapi' package is not installed. Install it with: "
                "pip install fredapi"
            )
        if not fred_api_key:
            raise RuntimeError(
                "FRED_API_KEY not found. Please create a free FRED API key at "
                "https://fred.stlouisfed.org/docs/api/api_key.html and set it via "
                "the FRED_API_KEY environment variable or --fred-key. "
                "This pipeline fails fast rather than falling back to an "
                "unreliable, undocumented data source."
            )

        self.fred = Fred(api_key=fred_api_key)
        self.logger.info("FRED API key detected -- using fredapi (Fred.get_series) for all FRED series.")

    # -- generic retry wrapper -----------------------------------------------
    def _with_retries(self, fn: Callable[[], pd.DataFrame], name: str) -> pd.DataFrame:
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.retries + 1):
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                self.logger.warning(
                    "  attempt %d/%d failed for %s: %s", attempt, self.retries, name, exc
                )
                time.sleep(self.pause)
        raise last_exc if last_exc else RuntimeError(f"Unknown failure for {name}")

    # -- Yahoo Finance --------------------------------------------------------
    def fetch_yahoo(self, name: str, tickers: Candidates, category: str, filename: str,
                     interval: str = "1d", period: Optional[str] = None) -> None:
        """
        Download OHLCV data from Yahoo Finance.

        For daily+ intervals, uses the configured start/end date range.
        For intraday intervals (e.g. '1h'), Yahoo does not support arbitrary
        historical date ranges -- it only serves a rolling lookback window
        (730 days for 1h bars). In that case pass `period` (e.g. '730d') and
        it will be used instead of start/end, per yfinance's documented
        intraday limits.
        """
        if yf is None:
            self._record_failure(name, category, "Yahoo Finance", str(tickers), "yfinance not installed")
            return

        candidates = as_candidate_list(tickers)
        last_error: Optional[str] = None
        used_ticker: Optional[str] = None
        df: Optional[pd.DataFrame] = None

        for ticker in candidates:
            def _dl(t=ticker) -> pd.DataFrame:
                if period:
                    d = yf.download(
                        t, period=period, interval=interval,
                        progress=False, auto_adjust=False, threads=False,
                    )
                else:
                    d = yf.download(
                        t, start=self.start, end=self.end, interval=interval,
                        progress=False, auto_adjust=False, threads=False,
                    )
                if d is None or d.empty:
                    raise ValueError("No data returned")
                if isinstance(d.columns, pd.MultiIndex):
                    d.columns = [c[0] for c in d.columns]
                return d

            try:
                df = self._with_retries(_dl, f"{name} ({ticker})")
                used_ticker = ticker
                if ticker != candidates[0]:
                    self.logger.info("  [fallback] %s succeeded using alternate ticker %s", name, ticker)
                break
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                continue

        if df is None:
            self._record_failure(name, category, "Yahoo Finance", ", ".join(candidates),
                                  last_error or "No candidate tickers succeeded")
            return

        try:
            df = clean_dataframe(df, name, self.logger)
            save_path = self.raw / category / f"{filename}.parquet"
            save_parquet(df, save_path, self.logger)
            self._record_success(name, category, "Yahoo Finance", used_ticker, df, save_path)
        except Exception as exc:  # noqa: BLE001
            self._record_failure(name, category, "Yahoo Finance", used_ticker or ", ".join(candidates), str(exc))

    def fetch_yahoo_stocks_batch(self, tickers_map: dict, category: str, combined_filename: str,
                                  interval: str = "1d", period: Optional[str] = None,
                                  chunk_size: int = 50) -> None:
        """
        Download OHLCV data for a large universe of single-name stocks (e.g. all
        S&P 500 constituents) efficiently, by requesting `chunk_size` tickers per
        Yahoo Finance call instead of one call per ticker. Unlike the other
        fetchers, results are NOT saved one-file-per-ticker; instead every
        ticker's cleaned OHLCV rows are stacked into a single long/tidy table
        (columns: Date, Ticker, Open, High, Low, Close, Adj Close, Volume) and
        written out as one combined Parquet file, per the requested layout.

        tickers_map: {display_name: yahoo_ticker}
        """
        if yf is None:
            for name, ticker in tickers_map.items():
                self._record_failure(name, category, "Yahoo Finance", ticker, "yfinance not installed")
            return

        items = list(tickers_map.items())
        chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]

        self.logger.info(
            "  Fetching %d tickers in %d batches of up to %d (interval=%s, %s)",
            len(items), len(chunks), chunk_size, interval,
            f"period={period}" if period else f"start={self.start} end={self.end}",
        )

        per_ticker_frames: list = []

        for chunk in tqdm(chunks, desc=f"{category} ({interval})", unit="batch"):
            chunk_tickers = [t for _, t in chunk]

            def _dl(tk=chunk_tickers) -> pd.DataFrame:
                kwargs = dict(
                    group_by="ticker", progress=False, auto_adjust=False,
                    threads=True, interval=interval,
                )
                if period:
                    d = yf.download(tk, period=period, **kwargs)
                else:
                    d = yf.download(tk, start=self.start, end=self.end, **kwargs)
                if d is None or d.empty:
                    raise ValueError("No data returned for batch")
                return d

            try:
                batch_df = self._with_retries(_dl, f"batch[{chunk_tickers[0]}..{chunk_tickers[-1]}]")
            except Exception as exc:  # noqa: BLE001
                for name, ticker in chunk:
                    self._record_failure(name, category, "Yahoo Finance", ticker,
                                          f"Batch request failed: {exc}")
                continue

            for name, ticker in chunk:
                try:
                    if isinstance(batch_df.columns, pd.MultiIndex):
                        if ticker not in batch_df.columns.get_level_values(0):
                            raise ValueError("Ticker missing from batch result")
                        sub = batch_df[ticker].copy()
                    else:
                        # Only happens if yfinance collapsed to flat columns
                        # (observed when a bare string, not a list, is passed).
                        sub = batch_df.copy()
                    sub = sub.dropna(how="all")
                    if sub.empty:
                        raise ValueError("No data returned")

                    sub = clean_dataframe(sub, name, self.logger)
                    first_date = str(sub.index.min().date()) if len(sub) else None
                    last_date = str(sub.index.max().date()) if len(sub) else None
                    n_missing = int(sub.isna().sum().sum())
                    shape = sub.shape

                    sub = sub.reset_index()
                    sub.insert(1, "Ticker", ticker)
                    per_ticker_frames.append(sub)

                    # Track per-ticker success in the summary/log even though no
                    # individual file is written -- save_path reflects the combined file.
                    self.summary.add(DownloadResult(
                        name=name, category=category, source="Yahoo Finance", identifier=ticker,
                        success=True, shape=shape, first_date=first_date, last_date=last_date,
                        missing_values=n_missing,
                        save_path=str(self.raw / category / f"{combined_filename}.parquet"),
                    ))
                except Exception as exc:  # noqa: BLE001
                    self._record_failure(name, category, "Yahoo Finance", ticker, str(exc))

        if not per_ticker_frames:
            self.logger.error("  [combined] no tickers succeeded for %s -- nothing to save", combined_filename)
            return

        combined = pd.concat(per_ticker_frames, axis=0, ignore_index=True)
        combined = combined.sort_values(["Ticker", "Date"]).reset_index(drop=True)
        save_path = self.raw / category / f"{combined_filename}.parquet"
        save_parquet(combined, save_path, self.logger)
        self.logger.info(
            "  [combined] %s: %d tickers, %d total rows -> %s",
            combined_filename, combined["Ticker"].nunique(), len(combined), save_path,
        )

    # -- FRED -------------------------------------------------------------
    def _fred_via_fredapi(self, series_id: str) -> pd.DataFrame:
        """
        Fetch a series using the official fredapi package, which wraps
        https://api.stlouisfed.org/fred/series/observations. This is the
        ONLY method used for FRED data -- no scraping, no CSV-export
        endpoints, no undocumented fallbacks.
        """
        s = self.fred.get_series(series_id, observation_start=self.start, observation_end=self.end)
        if s is None or s.empty:
            raise ValueError("No data returned from fredapi")
        return s.to_frame(name=series_id)

    def fetch_fred(self, name: str, series_id: str, category: str, filename: str) -> None:
        try:
            df = self._with_retries(lambda: self._fred_via_fredapi(series_id), name)
        except Exception as exc:  # noqa: BLE001
            self._record_failure(name, category, "FRED", series_id, str(exc))
            return

        try:
            df = clean_dataframe(df, name, self.logger)
            save_path = self.raw / category / f"{filename}.parquet"
            save_parquet(df, save_path, self.logger)
            self._record_success(name, category, "FRED (fredapi)", series_id, df, save_path)
        except Exception as exc:  # noqa: BLE001
            self._record_failure(name, category, "FRED", series_id, str(exc))

    # -- bookkeeping -----------------------------------------------------
    def _record_success(self, name, category, source, identifier, df, save_path) -> None:
        self.summary.add(DownloadResult(
            name=name, category=category, source=source, identifier=identifier,
            success=True, shape=df.shape,
            first_date=str(df.index.min().date()) if len(df) else None,
            last_date=str(df.index.max().date()) if len(df) else None,
            missing_values=int(df.isna().sum().sum()),
            save_path=str(save_path),
        ))

    def _record_failure(self, name, category, source, identifier, error) -> None:
        self.logger.error("  [FAIL] %-28s (%s: %s) -> %s", name, source, identifier, error)
        self.summary.add(DownloadResult(
            name=name, category=category, source=source, identifier=identifier,
            success=False, error=error,
        ))


# =============================================================================
# Dataset registries
# Each entry: display_name -> ticker/series_id, OR a list of candidate
# identifiers to try in order (first success wins). Lists are used for
# symbols known to be occasionally unavailable on Yahoo Finance.
# =============================================================================

US_INDICES: dict[str, Candidates] = {
    "S&P 500": "^GSPC",
    "Dow Jones Industrial Average": "^DJI",
    "NASDAQ Composite": "^IXIC",
    "NASDAQ-100": "^NDX",
    "Russell 2000": "^RUT",
    "Russell 1000": "^RUI",
    "Russell 3000": "^RUA",
    "Wilshire 5000": ["^W5000", "^FTW5000", "VTI"],  # VTI = total-market ETF proxy
    "NYSE Composite": "^NYA",
    "S&P MidCap 400": "^MID",
    "S&P SmallCap 600": ["^SML", "^SP600", "IJR"],  # IJR = iShares S&P SmallCap 600 ETF proxy
    "S&P 100": "^OEX",
}

US_SECTORS: dict[str, Candidates] = {
    "Technology": "XLK",
    "Financials": "XLF",
    "Healthcare": "XLV",
    "Energy": "XLE",
    "Utilities": "XLU",
    "Industrials": "XLI",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
    "Semiconductors": "SMH",
    "Biotechnology": "IBB",
    "Transportation": "IYT",
    "Software": "IGV",
    "Banks": "KBE",
    "Insurance": "KIE",
    "Homebuilders": "XHB",
    "Retail": "XRT",
    "Cloud Computing": "SKYY",
    "Cybersecurity": "HACK",
    "Clean Energy": "ICLN",
}

VOLATILITY: dict[str, Candidates] = {
    "CBOE VIX": "^VIX",
    "VXN (Nasdaq Volatility)": "^VXN",
    "VVIX (VIX of VIX)": "^VVIX",
    "MOVE Index (Bond Volatility)": ["^MOVE", "^MOVE.X"],
    "Short-Term VIX (VIX9D)": "^VIX9D",
    "3-Month VIX (VIX3M)": "^VIX3M",
    # Note: CBOE's Russell 2000 Volatility Index has no consistently reliable
    # free ticker on Yahoo Finance. ^RVX is included as a best-effort attempt;
    # if it fails, this series simply has no free equivalent available.
    "RVX (Russell 2000 Volatility)": "^RVX",
}

FOREX: dict[str, Candidates] = {
    "US Dollar Index (DXY)": ["DX-Y.NYB", "DX=F"],
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "USDJPY=X",
    "USD/CHF": "USDCHF=X",
    "USD/CAD": "USDCAD=X",
    "AUD/USD": "AUDUSD=X",
    "NZD/USD": "NZDUSD=X",
}

COMMODITIES: dict[str, Candidates] = {
    "Gold Futures": "GC=F",
    "Silver Futures": "SI=F",
    "Copper Futures": "HG=F",
    "Platinum Futures": "PL=F",
    "Palladium Futures": "PA=F",
    "WTI Crude Oil": "CL=F",
    "Brent Crude Oil": "BZ=F",
    "Natural Gas": "NG=F",
    "Corn Futures": "ZC=F",
    "Wheat Futures": "ZW=F",
    "Soybean Futures": "ZS=F",
    "Coffee Futures": "KC=F",
    "Sugar Futures": "SB=F",
    "Cotton Futures": "CT=F",
    "Lumber Futures": ["LBS=F", "LBR=F"],
}

TREASURY_YIELDS: dict[str, str] = {
    "1 Month Treasury Yield": "DGS1MO",
    "3 Month Treasury Yield": "DGS3MO",
    "6 Month Treasury Yield": "DGS6MO",
    "1 Year Treasury Yield": "DGS1",
    "2 Year Treasury Yield": "DGS2",
    "3 Year Treasury Yield": "DGS3",
    "5 Year Treasury Yield": "DGS5",
    "7 Year Treasury Yield": "DGS7",
    "10 Year Treasury Yield": "DGS10",
    "20 Year Treasury Yield": "DGS20",
    "30 Year Treasury Yield": "DGS30",
}

FRED_MACRO: dict[str, str] = {
    # Inflation
    "CPI (All Urban Consumers)": "CPIAUCSL",
    "Core CPI (ex Food & Energy)": "CPILFESL",
    "PPI (All Commodities)": "PPIACO",
    "Core PPI": "PPIFES",
    "PCE Price Index": "PCEPI",
    "Core PCE Price Index": "PCEPILFE",
    # Labor
    "Unemployment Rate": "UNRATE",
    "Initial Jobless Claims": "ICSA",
    "Continuing Claims": "CCSA",
    "Nonfarm Payrolls": "PAYEMS",
    "Labor Force Participation Rate": "CIVPART",
    # Growth
    "GDP (Nominal)": "GDP",
    "Real GDP": "GDPC1",
    "Industrial Production Index": "INDPRO",
    "Capacity Utilization": "TCU",
    "Retail Sales": "RSAFS",
    # Housing
    "Housing Starts": "HOUST",
    "Building Permits": "PERMIT",
    "Existing Home Sales": "EXHOSLUSM495S",
    "New Home Sales": "HSN1F",
    # Liquidity
    "Fed Total Assets (Balance Sheet)": "WALCL",
    "M2 Money Supply": "M2SL",
    "Reverse Repo (RRP)": "RRPONTSYD",
    "SOFR": "SOFR",
    "Effective Fed Funds Rate": "DFF",
    # Credit
    "Moody's AAA Corporate Spread": "AAA10Y",
    "Moody's BAA Corporate Spread": "BAA10Y",
    "High Yield Spread (ICE BofA)": "BAMLH0A0HYM2",
    "Corporate Bond Spread (ICE BofA IG)": "BAMLC0A0CM",
    # Yield curve
    "10Y-2Y Treasury Spread": "T10Y2Y",
    "10Y-3M Treasury Spread": "T10Y3M",
    # Recession indicator
    "NBER Recession Indicator (USREC)": "USREC",
    # Dollar
    "Trade Weighted Broad Dollar Index": "DTWEXBGS",
    # Manufacturing
    "ISM Manufacturing PMI (proxy: New Orders)": "AMTMNO",
}

GLOBAL_INDICES: dict[str, Candidates] = {
    "FTSE 100 (UK)": "^FTSE",
    "DAX (Germany)": "^GDAXI",
    "CAC 40 (France)": "^FCHI",
    "Euro Stoxx 50": "^STOXX50E",
    "Nikkei 225 (Japan)": "^N225",
    "TOPIX (Japan)": ["^TPX", "1306.T"],  # 1306.T = TOPIX-tracking ETF proxy
    "Hang Seng (Hong Kong)": "^HSI",
    "Shanghai Composite (China)": "000001.SS",
    "Shenzhen Component (China)": "399001.SZ",
    "CSI 300 (China)": "000300.SS",
    "KOSPI (South Korea)": "^KS11",
    "ASX 200 (Australia)": "^AXJO",
    "TSX Composite (Canada)": "^GSPTSE",
    "Bovespa (Brazil)": "^BVSP",
    "IPC Mexico": "^MXX",
    "MSCI Emerging Markets (ETF proxy: EEM)": "EEM",
}


def get_sp500_constituents(logger: logging.Logger) -> dict:
    """
    Fetch the current list of S&P 500 constituent tickers.

    Neither Yahoo Finance nor FRED expose an index-membership endpoint, so
    this pulls the standard, widely-used Wikipedia constituents table (the
    same source most quant tooling relies on for this purpose). This is
    metadata about which tickers to request -- the actual price data for
    every ticker still comes exclusively from Yahoo Finance, per the data
    source rules for the rest of the pipeline.

    Returns {display_name: yahoo_ticker}, e.g. {"Apple Inc. (AAPL)": "AAPL"}.
    Tickers with dots (e.g. BRK.B) are converted to Yahoo's hyphen convention
    (BRK-B).
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        # pd.read_html(url) makes a bare urllib request with no User-Agent,
        # which Wikipedia's edge now blocks with HTTP 403. Fetching the page
        # ourselves with a standard browser User-Agent avoids that, then we
        # hand the raw HTML to pd.read_html to parse the table as before.
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            )
        }
        html_text: Optional[str] = None
        if requests is not None:
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            html_text = resp.text
        else:
            import urllib.request
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as f:  # noqa: S310
                html_text = f.read().decode("utf-8", errors="replace")

        tables = pd.read_html(StringIO(html_text))
        df = tables[0]
        tickers_map: dict = {}
        for _, row in df.iterrows():
            symbol = str(row["Symbol"]).strip().replace(".", "-")
            security = str(row["Security"]).strip()
            if not symbol or symbol.lower() == "nan":
                continue
            tickers_map[f"{security} ({symbol})"] = symbol
        logger.info("Fetched %d S&P 500 constituents from Wikipedia", len(tickers_map))
        return tickers_map
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Could not fetch S&P 500 constituent list (%s). Per-stock downloads "
            "will be skipped; index-level data is unaffected.", exc
        )
        return {}


def slugify(text: str) -> str:
    """Convert a display name into a short, filesystem-safe filename."""
    keep = "".join(c.lower() if c.isalnum() else "_" for c in text)
    while "__" in keep:
        keep = keep.replace("__", "_")
    return keep.strip("_")


FILENAME_OVERRIDES: dict[str, str] = {
    "^GSPC": "sp500", "^DJI": "dow_jones", "^IXIC": "nasdaq_composite",
    "^NDX": "nasdaq_100", "^RUT": "russell_2000", "^VIX": "vix",
    "^VXN": "vxn", "^VVIX": "vvix", "^MOVE": "move_index",
    "DX-Y.NYB": "dxy", "GC=F": "gold", "SI=F": "silver", "CL=F": "wti_crude",
    "BZ=F": "brent_crude", "NG=F": "natural_gas", "DGS10": "us10y",
    "DGS2": "us2y", "DGS3MO": "us3m", "CPIAUCSL": "cpi", "UNRATE": "unemployment_rate",
}


def filename_for(identifier: Candidates, fallback_name: str) -> str:
    primary = identifier[0] if isinstance(identifier, list) else identifier
    return FILENAME_OVERRIDES.get(primary, slugify(fallback_name))


# =============================================================================
# Post-processing: derived yield-curve spreads
# =============================================================================

def compute_yield_curve_spreads(lake: DataLake) -> None:
    """Compute 10Y-2Y, 10Y-3M, 30Y-10Y from already-downloaded treasury data."""
    logger = lake.logger
    treasury_dir = lake.raw / "treasury"
    needed = {
        "10Y": "us10y.parquet", "2Y": "us2y.parquet",
        "3M": "us3m.parquet", "30Y": "us30y.parquet",
    }
    series = {}
    for label, fname in needed.items():
        p = treasury_dir / fname
        if p.exists():
            try:
                df = pd.read_parquet(p)
                series[label] = df.iloc[:, 0]
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not load %s for spread calc: %s", fname, exc)

    spreads = {
        "10y_minus_2y": ("10Y", "2Y"),
        "10y_minus_3m": ("10Y", "3M"),
        "30y_minus_10y": ("30Y", "10Y"),
    }

    for out_name, (a, b) in spreads.items():
        if a in series and b in series:
            combined = pd.concat([series[a], series[b]], axis=1, join="inner")
            combined.columns = [a, b]
            spread = (combined[a] - combined[b]).to_frame(name=out_name)
            spread.index.name = "Date"
            save_path = treasury_dir / f"{out_name}.parquet"
            save_parquet(spread, save_path, logger)
            logger.info("  [derived] computed %s from %s and %s", out_name, a, b)
        else:
            logger.warning("  [derived] skipped %s: missing inputs", out_name)


def build_combined_table(lake: DataLake, category: str, output_name: str) -> None:
    """
    Build one wide, outer-joined convenience table from all individually-saved
    Parquet files in a category (e.g. all treasury yields, all FRED macro
    series). This does NOT reduce the number of FRED API calls -- FRED's API
    has no bulk/multi-series endpoint, so each series is still fetched with
    its own request. It only affects how the already-downloaded results are
    packaged for easier downstream date-alignment and feature engineering.
    The individual per-series files are kept as-is (per the data lake spec).
    """
    logger = lake.logger
    cat_dir = lake.raw / category
    if not cat_dir.exists():
        return

    frames = []
    for p in sorted(cat_dir.glob("*.parquet")):
        try:
            df = pd.read_parquet(p)
            frames.append(df)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load %s for combined table: %s", p.name, exc)

    if not frames:
        logger.warning("No files found in %s to build combined table", cat_dir)
        return

    combined = pd.concat(frames, axis=1, join="outer").sort_index()
    combined.index.name = "Date"
    save_path = cat_dir / f"{output_name}.parquet"
    save_parquet(combined, save_path, logger)
    logger.info("  [combined] built %s with %d columns, %d rows from %d source files",
                output_name, combined.shape[1], combined.shape[0], len(frames))


# =============================================================================
# Orchestration
# =============================================================================

def run_category(lake: DataLake, registry: dict, category: str,
                  source: str, logger: logging.Logger) -> None:
    logger.info("-" * 80)
    logger.info("Downloading category: %s (%d datasets, source=%s)",
                category, len(registry), source)
    logger.info("-" * 80)

    for display_name, identifier in tqdm(registry.items(), desc=category, unit="ds"):
        filename = filename_for(identifier, display_name)
        if source == "yahoo":
            lake.fetch_yahoo(display_name, identifier, category, filename)
        else:
            lake.fetch_fred(display_name, identifier, category, filename)


def main() -> None:
    parser = argparse.ArgumentParser(description="Quant research data lake builder")
    parser.add_argument("--root", type=str, default="./data",
                         help="Root directory for the data lake (default: ./data)")
    parser.add_argument("--start", type=str, default="2000-01-01",
                         help="Start date YYYY-MM-DD (default: 2000-01-01)")
    parser.add_argument("--end", type=str, default=None,
                         help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--fred-key", type=str, default=os.environ.get("FRED_API_KEY"),
                         help="FRED API key (required). Get one free at "
                              "https://fred.stlouisfed.org/docs/api/api_key.html, "
                              "or set the FRED_API_KEY environment variable.")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    logger = build_logger(root / "logs")

    logger.info("=" * 80)
    logger.info("QUANT RESEARCH DATA LAKE — DOWNLOAD RUN STARTED")
    logger.info("Root directory : %s", root)
    logger.info("Date range     : %s -> %s", args.start, args.end or "today")
    logger.info("=" * 80)

    try:
        lake = DataLake(root=root, start=args.start, end=args.end,
                         fred_api_key=args.fred_key, logger=logger)
    except RuntimeError as exc:
        logger.error("FATAL: %s", exc)
        sys.exit(1)

    # ---- Market data (Yahoo Finance) ----
    run_category(lake, US_INDICES, "us_indices", "yahoo", logger)
    run_category(lake, US_SECTORS, "us_sectors", "yahoo", logger)
    run_category(lake, VOLATILITY, "volatility", "yahoo", logger)
    run_category(lake, FOREX, "forex", "yahoo", logger)
    run_category(lake, COMMODITIES, "commodities", "yahoo", logger)
    run_category(lake, GLOBAL_INDICES, "global_indices", "yahoo", logger)

    # S&P 500 hourly bars, last 730 days. Yahoo only serves a rolling
    # 730-day window for 1h data (no arbitrary start/end like daily bars),
    # so this uses period='730d' instead of the pipeline's --start date.
    logger.info("-" * 80)
    logger.info("Downloading intraday dataset: S&P 500 hourly (last 730 days)")
    logger.info("-" * 80)
    lake.fetch_yahoo(
        name="S&P 500 (Hourly, 730d)",
        tickers="^GSPC",
        category="us_indices_intraday",
        filename="sp500_hourly",
        interval="1h",
        period="730d",
    )

    # S&P 500 constituent (single-stock) data -- every company in the index,
    # daily since --start and hourly for the last 730 days. Requested in
    # batches to Yahoo Finance for efficiency, but written out as ONE
    # combined long-format table per granularity (not one file per stock).
    logger.info("-" * 80)
    logger.info("Downloading S&P 500 constituent list")
    logger.info("-" * 80)
    sp500_constituents = get_sp500_constituents(logger)

    if sp500_constituents:
        logger.info("-" * 80)
        logger.info("Downloading daily data for %d S&P 500 constituents", len(sp500_constituents))
        logger.info("-" * 80)
        lake.fetch_yahoo_stocks_batch(
            tickers_map=sp500_constituents,
            category="us_stocks",
            combined_filename="sp500_constituents_daily",
            interval="1d",
        )

        logger.info("-" * 80)
        logger.info("Downloading hourly data (last 730 days) for %d S&P 500 constituents",
                     len(sp500_constituents))
        logger.info("-" * 80)
        lake.fetch_yahoo_stocks_batch(
            tickers_map=sp500_constituents,
            category="us_stocks",
            combined_filename="sp500_constituents_hourly",
            interval="1h",
            period="730d",
        )
    else:
        logger.warning("Skipping S&P 500 per-stock downloads: constituent list unavailable")

    # ---- Macro / rates data (FRED) ----
    run_category(lake, TREASURY_YIELDS, "treasury", "fred", logger)
    run_category(lake, FRED_MACRO, "fred", "fred", logger)

    # ---- Derived series ----
    compute_yield_curve_spreads(lake)

    # ---- Combined convenience tables (individual files remain the source of truth) ----
    build_combined_table(lake, "treasury", "treasury_yields_combined")
    build_combined_table(lake, "fred", "fred_macro_combined")

    # ---- Final summary ----
    lake.summary.print_summary(logger)


if __name__ == "__main__":
    main()