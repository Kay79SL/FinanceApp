from shiny import App, ui, render, reactive
import pandas as pd
import yfinance as yf
from shinywidgets import render_widget
from datetime import datetime, date, timedelta
import numpy as np
import os
import json
from pathlib import Path
import plotly.graph_objects as go
from shinywidgets import output_widget, render_plotly
import plotly.express as px
import re

#==========================

# settings for colours

def color_to_rgba(c: str, alpha: float = 0.5) -> str:
    """Convert '#RRGGBB', 'rgb(r,g,b)', or 'rgba(r,g,b,a)' to rgba(..., alpha)."""
    c = str(c).strip()
    if c.startswith("#") and len(c) == 7:
        r = int(c[1:3], 16); g = int(c[3:5], 16); b = int(c[5:7], 16)
        return f"rgba({r},{g},{b},{alpha})"
    m = re.match(r"rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", c)
    if m:
        r, g, b = m.groups()
        return f"rgba({r},{g},{b},{alpha})"
    m = re.match(r"rgba\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*([0-9.]+)\s*\)", c)
    if m:
        r, g, b, _ = m.groups()
        return f"rgba({r},{g},{b},{alpha})"
    return c




# ============================
# CONFIG
# ===========================

#settings for paths to github
APP_NAME = "MyFinanceApp"

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

TOP100_CSV = str(DATA_DIR / "top_1000CSV.csv")          
TICKERS_MASTER_CSV = str(DATA_DIR / "master_tickers.csv")

DIVIDEND_YIELD_ASSUMPTION = 0.03  # for projections only 
DEFAULT_MASTER_TICKERS = ["LLY", "ABT", "ABBV"]

# Top chart - dAILY since 2020-01-01
CHART_START_DATE = date(2020, 1, 1)


# ===========================
# PATH + LOCAL STORAGE
# ===========================

#portfolio local path storage
def portfolio_path() -> Path:
    base = Path(os.environ.get("LOCALAPPDATA", str(Path.home())))
    folder = base / APP_NAME
    folder.mkdir(parents=True, exist_ok=True)
    return folder / "portfolio.json"

# loading and saving portfolio locally
def load_portfolio_local() -> list[dict]:
    p = portfolio_path()
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_portfolio_local(entries: list[dict]) -> None:
    p = portfolio_path()
    p.write_text(json.dumps(entries, indent=2), encoding="utf-8")


def ensure_parent_dir(path_str: str) -> None:
    Path(path_str).parent.mkdir(parents=True, exist_ok=True)


# ===========================
# SMALL UTILITIES
# ===========================

#definition to convert to float
def as_float(x, default=0.0) -> float:
    """Avoid pandas - Warning: float(Series) is deprecated."""
    try:
        if isinstance(x, pd.Series):
            return float(x.iloc[0]) if len(x) else float(default)
        if isinstance(x, (np.ndarray, list, tuple)):
            return float(x[0]) if len(x) else float(default)
        if pd.isna(x):
            return float(default)
        return float(x)
    except Exception:
        return float(default)

# def to clean ticker list
def normalize_ticker_list(items) -> list[str]:
    out = []
    for x in items or []:
        t = str(x).strip().upper()
        if t:
            out.append(t)
    return sorted(list(set(out)))


# ===========================
# MASTER TICKERS (ensure exists BEFORE UI)
# ===========================

# loading main ticker file
def load_master_tickers_file() -> list[str]:
    p = Path(TICKERS_MASTER_CSV)
    if not p.exists():
        return []
    try:
        df = pd.read_csv(p)
        if "Ticker" not in df.columns:
            return []
        return normalize_ticker_list(df["Ticker"].tolist())
    except Exception:
        return []

#saving main ticker file
def save_master_tickers_file(tickers: list[str]) -> None:
    ensure_parent_dir(TICKERS_MASTER_CSV)
    out = normalize_ticker_list(tickers)
    pd.DataFrame({"Ticker": out}).to_csv(TICKERS_MASTER_CSV, index=False)


def ensure_master_file_exists() -> None:
    p = Path(TICKERS_MASTER_CSV)
    if not p.exists():
        save_master_tickers_file(DEFAULT_MASTER_TICKERS)


ensure_master_file_exists()
MASTER_CHOICES = load_master_tickers_file()


# ===========================
# MUTED PALETTE (20+ series)
# ===========================
#choosing muted colours for better visibility in plots

MUTED = [
    "#636EFA",  # blue
    "#EF553B",  # red-orange
    "#00CC96",  # green
    "#AB63FA",  # purple
    "#19D3F3",  # cyan
    "#FF6692",  # pink
    "#B6E880",  # light green
    "#FF97FF",  # light pink
    "#7F7F7F",  # gray
    "#2CA02C",  # dark green
    "#1F77B4",  # deep blue
    "#9467BD",  # violet
    "#8C564B",  # brown
    "#17BECF",  # teal
    "#E377C2",  # magenta
    "#BCBD22",  # olive (not bright yellow)
    "#AEC7E8",  # pale blue
    "#C5B0D5",  # pale purple
]


def palette_for_n(n: int) -> list[str]:
    if n <= len(MUTED):
        return MUTED[:n]
    reps = int(np.ceil(n / len(MUTED)))
    return (MUTED * reps)[:n]


# ===========================
# DIVIDENDS (ACTUAL TO DATE)
# ===========================
_DIVIDENDS_CACHE: dict[str, pd.Series] = {}

#getting div payment history and caching
def get_dividends_series(ticker: str) -> pd.Series:
    t = str(ticker).strip().upper()
    if t in _DIVIDENDS_CACHE:
        return _DIVIDENDS_CACHE[t]
    try:
        s = yf.Ticker(t).dividends
        if s is None:
            s = pd.Series(dtype=float)
        if len(s):
            s.index = pd.to_datetime(s.index, errors="coerce")
            if getattr(s.index, "tz", None) is not None:
                s.index = s.index.tz_localize(None)
        _DIVIDENDS_CACHE[t] = s
        return s
    except Exception:
        _DIVIDENDS_CACHE[t] = pd.Series(dtype=float)
        return _DIVIDENDS_CACHE[t]

# calculating total dividends to date
def dividends_to_date_cash(ticker: str, purchase_date: datetime, shares: float) -> float:
    s = get_dividends_series(ticker)
    if s is None or len(s) == 0:
        return 0.0
    start = pd.Timestamp(purchase_date)
    end = pd.Timestamp(datetime.today())
    s2 = s.loc[(s.index >= start) & (s.index <= end)]
    return as_float(s2.sum(), 0.0) * as_float(shares, 0.0) if len(s2) else 0.0

# Cache price-on-date lookups to avoid repeated downloads
_PRICE_ON_DATE_CACHE: dict[tuple[str, str], float] = {}

#saving price of ticker on date with caching
def price_on_date_cached(ticker: str, dt: datetime) -> float | None:
    t = str(ticker).strip().upper()
    key = (t, dt.date().isoformat())
    if key in _PRICE_ON_DATE_CACHE:
        return _PRICE_ON_DATE_CACHE[key]
    p = get_price_on_date(t, dt)  
    if p is not None:
        _PRICE_ON_DATE_CACHE[key] = float(p)
    return p


#Actual dividends paid between purchase_date and as_of (inclusive), multiplied by shares.
def dividends_to_date_cash_asof(ticker: str, purchase_date: datetime, shares: float, as_of: datetime) -> float:
    s = get_dividends_series(ticker)
    if s is None or len(s) == 0:
        return 0.0
    start = pd.Timestamp(purchase_date)
    end = pd.Timestamp(as_of)
    s2 = s.loc[(s.index >= start) & (s.index <= end)]
    if len(s2) == 0:
        return 0.0
    return as_float(s2.sum(), 0.0) * as_float(shares, 0.0)


"""
Returns totals:
      - total_investment: sum shares * price(as_of)
      - total_dividends: sum dividends from purchase→as_of
      - total_profit: sum (capital_gain(as_of) - cgt(as_of) + tax_saved + espp + dividends)
      
      
    ****************** Income logic per entry: *****************  
    CGT logic: max(0, current_value*0.33 - 1070)
    """
    
 # main function to calculate total investment, total dividends, total profits,    
def compute_portfolio_kpis(entries: list[dict], as_of: datetime) -> dict:
   
    total_investment = 0.0
    total_dividends = 0.0
    total_profit = 0.0

    for e in entries:
        ticker = str(e.get("Ticker", "")).strip().upper()
        shares = as_float(e.get("Shares"), 0.0)
        inv_type = e.get("InvType", "Regular")
        pds = e.get("PurchaseDate")

        try:
            purchase_dt = datetime.fromisoformat(pds) if pds else as_of
        except Exception:
            purchase_dt = as_of

        purchase_price = e.get("PurchasePrice")
        if purchase_price is None or (isinstance(purchase_price, float) and np.isnan(purchase_price)):
            purchase_price = price_on_date_cached(ticker, purchase_dt)

        purchase_price = as_float(purchase_price, default=100.0)
        if purchase_price <= 0:
            purchase_price = 100.0

        # Value from date
        price_asof = price_on_date_cached(ticker, as_of)
        price_asof = as_float(price_asof, default=purchase_price)
        current_value = shares * price_asof
        total_investment += current_value

        # Dividends 
        divs = dividends_to_date_cash_asof(ticker, purchase_dt, shares, as_of)
        total_dividends += divs

        # Income logic 
        initial_value = shares * purchase_price
        capital_gain = current_value - initial_value

        # assigning tax saved and espp benefit
        tax_saved = 0.0
        espp_benefit = 0.0
        if inv_type == "Tax Reduction":
            tax_saved = initial_value * 0.40
        elif inv_type == "ESPP":
            espp_benefit = initial_value * 0.15
        # CGT calculation
        cgt_asof = max(0.0, (current_value * 0.33) - 1070.0)

        # total income calculation
        total_income = (capital_gain - cgt_asof) + tax_saved + espp_benefit + divs
        total_profit += total_income

    return {
        "total_investment": float(total_investment),
        "total_dividends": float(total_dividends),
        "total_profit": float(total_profit),
    }

# ===========================
# CAGR for KPIs + 2040 prediction
# ===========================

#cached max history close series
_HIST_CLOSE_CACHE: dict[str, pd.Series] = {}

def _get_close_series_max(ticker: str) -> pd.Series:
    """
    Download max history once and cache 
    """
    t = str(ticker).strip().upper()
    if t in _HIST_CLOSE_CACHE:
        return _HIST_CLOSE_CACHE[t]

    try:
        df = yf.download(t, period="max", auto_adjust=True, progress=False, threads=False)
    except Exception:
        _HIST_CLOSE_CACHE[t] = pd.Series(dtype=float)
        return _HIST_CLOSE_CACHE[t]

    if df is None or df.empty:
        _HIST_CLOSE_CACHE[t] = pd.Series(dtype=float)
        return _HIST_CLOSE_CACHE[t]

    close_obj = df["Close"] if "Close" in df.columns else df.iloc[:, 0]
    if isinstance(close_obj, pd.DataFrame):
        close = close_obj.iloc[:, 0]
    else:
        close = close_obj

    close = close.dropna().copy()
    close.index = pd.to_datetime(close.index, errors="coerce")
    close = close.dropna()
   
    if getattr(close.index, "tz", None) is not None:
        close.index = close.index.tz_localize(None)

    _HIST_CLOSE_CACHE[t] = close
    return close

# CAGR - compound Annual Growth Rate since 2010 up to as_of date
def get_cagr_since_2010_asof(ticker: str, as_of: datetime) -> float:
    
    t = str(ticker).strip().upper()
    close = _get_close_series_max(t)
    if close is None or close.empty:
        return 0.05

    as_of_ts = pd.Timestamp(as_of.date())
    close2 = close[close.index <= as_of_ts]
    if close2.empty:
        return 0.05

    close2 = close2[close2.index.year >= 2010]
    if len(close2) < 2:
        return 0.05

    start_price = as_float(close2.iloc[0], 0.0)
    end_price = as_float(close2.iloc[-1], 0.0)
    if start_price <= 0 or end_price <= 0:
        return 0.05

    start_date = close2.index[0]
    end_date = close2.index[-1]
    n_years = (end_date - start_date).days / 365.25
    if n_years <= 0:
        return 0.05

    cagr = (end_price / start_price) ** (1 / n_years) - 1
    return max(min(float(cagr), 0.5), -0.9)


"""    Compute total investment value per ticker as of a given date:
    sum shares * price(as_of) per ticker.
    """

def compute_ticker_investment_values(entries: list[dict], as_of: datetime) -> dict[str, float]:
   
    totals: dict[str, float] = {}
    for e in entries or []:
        t = str(e.get("Ticker", "")).strip().upper()
        if not t:
            continue
        shares = as_float(e.get("Shares"), 0.0)

        price_asof = price_on_date_cached(t, as_of)
        # fallback: if price missing, is taking purchase price or 0
        if price_asof is None:
            pp = e.get("PurchasePrice")
            price_asof = as_float(pp, 0.0)

        totals[t] = totals.get(t, 0.0) + (shares * as_float(price_asof, 0.0))

    return totals


# Projecting price and value for 2040
 """
    Total predicted value in 2040 using:
      ProjectedPrice(2040) = PurchasePrice * (1 + CAGR_asof)^(2040 - PurchaseYear)
      ProjectedValue(2040) = Shares * ProjectedPrice(2040)
    """

def compute_predicted_total_2040(entries: list[dict], as_of: datetime) -> float:
   
    total = 0.0
    for e in entries or []:
        t = str(e.get("Ticker", "")).strip().upper()
        if not t:
            continue

        shares = as_float(e.get("Shares"), 0.0)

        pds = e.get("PurchaseDate")
        try:
            purchase_dt = datetime.fromisoformat(pds) if pds else as_of
        except Exception:
            purchase_dt = as_of

        purchase_price = e.get("PurchasePrice")
        if purchase_price is None or (isinstance(purchase_price, float) and np.isnan(purchase_price)):
            purchase_price = price_on_date_cached(t, purchase_dt)

        purchase_price = as_float(purchase_price, 100.0)
        if purchase_price <= 0:
            purchase_price = 100.0

        cagr = get_cagr_since_2010_asof(t, as_of)
        years = 2040 - purchase_dt.year
        if years < 0:
            years = 0

        projected_price = purchase_price * ((1.0 + cagr) ** years)
        total += shares * projected_price

    return float(total)


# ===========================
# PRICES + SIMULATION (Planner/Summary)
# ===========================
# getting a stock price if missing or if weekends/holidays
def get_price_on_date(ticker: str, date_: datetime) -> float | None:
    t = str(ticker).strip().upper()
    try:
        df = yf.download(t, period="max", auto_adjust=True, progress=False, threads=False)
    except Exception:
        return None

    if df is None or df.empty or "Close" not in df.columns:
        return None

    prices = df["Close"].dropna().sort_index()
    if isinstance(prices, pd.DataFrame):
        prices = prices.iloc[:, 0]

    if prices.empty:
        return None

    idx = prices.index
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)

    target = pd.Timestamp(date_)
    pos = idx.get_indexer([target], method="nearest")[0]
    if pos == -1:
        return None

    return as_float(prices.iloc[pos], default=None)

# calculating CAGR since 2010
def get_cagr_since_2010(ticker: str) -> float:
    t = str(ticker).strip().upper()
    try:
        df = yf.download(t, period="max", auto_adjust=True, progress=False, threads=False)
    except Exception:
        return 0.05

    if df is None or df.empty or "Close" not in df.columns:
        return 0.05

    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df["Year"] = df.index.year
    df_2010 = df[df["Year"] >= 2010]
    if len(df_2010) < 2:
        return 0.05

    start_price = as_float(df_2010["Close"].iloc[0], 0.0)
    end_price = as_float(df_2010["Close"].iloc[-1], 0.0)

    years = (df_2010.index[-1] - df_2010.index[0]).days / 365.25
    if years <= 0 or start_price <= 0:
        return 0.05

    cagr = (end_price / start_price) ** (1 / years) - 1
    return max(min(cagr, 0.5), -0.9)


# -------------------------------------------------------------------
# Core model: simulate a single investment year-by-year to a horizon year
# Returns both: (1) time series for charts, (2) summary row for tables
# -------------------------------------------------------------------

 """
    Simulate the value of a single investment (fixed number of shares)
    from its purchase date to horizon_year using a constant CAGR
    estimated from 2010.

    Returns:
    - df_years: DataFrame with columns [Ticker, Investment, Year, Value]
    - summary: dict with aggregate metrics for the portfolio table
    """

def simulate_investment(entry: dict, horizon_year: int):
    ticker = str(entry["Ticker"]).strip().upper()
    shares = as_float(entry["Shares"], 0.0)
    inv_type = entry["InvType"]
    inv_name = entry["InvestmentName"]
    purchase_date_str = entry["PurchaseDate"]

    purchase_date = datetime.today() if not purchase_date_str else datetime.fromisoformat(purchase_date_str)
    base_year = purchase_date.year

    purchase_price = entry.get("PurchasePrice")
    if purchase_price is None:
        purchase_price = get_price_on_date(ticker, purchase_date)

    purchase_price = as_float(purchase_price, default=100.0)
    if purchase_price <= 0:
        purchase_price = 100.0

    g = get_cagr_since_2010(ticker)

    years = list(range(base_year, horizon_year + 1))
    values = []
    for y in years:
        t = y - base_year
        price_y = purchase_price * ((1 + g) ** t)
        values.append(shares * price_y)

    df_years = pd.DataFrame({"Ticker": ticker, "Investment": inv_name, "Year": years, "Value": values})

    # ------------------All Calculations ------------------------------
    
    initial_value = shares * purchase_price
    price_today = get_price_on_date(ticker, datetime.today())
    price_today = as_float(price_today, default=purchase_price)

    current_value_today = shares * price_today
    capital_gain_to_now = current_value_today - initial_value
    dividends_to_now = dividends_to_date_cash(ticker, purchase_date, shares)
    gain = capital_gain_to_now
    cgt_today = gain*0.33 - 1070.0

    tax_saved = 0.0
    espp_benefit = 0.0
    if inv_type == "Tax Reduction":
        tax_saved = initial_value * 0.40
    elif inv_type == "ESPP":
        espp_benefit = initial_value * 0.15

    #cgt_today = max(0.0, (current_value_today * 0.33) - 1070.0) - incorrect
    
    total_income_to_now = (capital_gain_to_now - cgt_today) + tax_saved + espp_benefit + dividends_to_now

    summary = {
        "Ticker": ticker,
        "Shares": shares,
        "Purchase Date": purchase_date.date().isoformat(),
        "Purchase Price": round(purchase_price, 2),
        "Price Today": round(as_float(price_today, purchase_price), 2),
        "Inv Type": inv_type,
        "Tax Saved": round(as_float(tax_saved, 0.0), 2),
        "ESPP 15% Benefit": round(as_float(espp_benefit, 0.0), 2),
        "Dividends": round(as_float(dividends_to_now, 0.0), 2),
        "CGT Today": round(as_float(cgt_today, 0.0), 2),
        "Total Income Net": round(as_float(total_income_to_now, 0.0), 2),
    }

    return df_years, summary


# ===========================
# TOP100 TABLE HELPERS
# ===========================

# pulling top1000 csv to get the top100 companies and calculating the 6M return
def safe_load_top100(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=["Ticker", "Price Now", "Price 6M Ago", "Return 6M %"])

    df.columns = [str(c).strip() for c in df.columns]
    tcol = "ticker" if "ticker" in df.columns else ("Ticker" if "Ticker" in df.columns else None)
    if not tcol:
        return pd.DataFrame(columns=["Ticker", "Price Now", "Price 6M Ago", "Return 6M %"])

    df[tcol] = df[tcol].astype(str).str.strip().str.upper()

    pcol = "price_now" if "price_now" in df.columns else None
    rcol = "return_6m_pct" if "return_6m_pct" in df.columns else None
    if pcol is None or rcol is None:
        return pd.DataFrame(columns=["Ticker", "Price Now", "Price 6M Ago", "Return 6M %"])

    df[pcol] = pd.to_numeric(df[pcol], errors="coerce")
    df[rcol] = pd.to_numeric(df[rcol], errors="coerce")

    denom = (1.0 + df[rcol] / 100.0)
    df["price_6m_ago"] = df[pcol] / denom.replace(0, np.nan)

    out = df[[tcol, pcol, "price_6m_ago", rcol]].copy()
    out = out.rename(columns={
        tcol: "Ticker",
        pcol: "Price Now",
        "price_6m_ago": "Price 6M Ago",
        rcol: "Return 6M %",
    })
    return out.round({"Price Now": 2, "Price 6M Ago": 2, "Return 6M %": 2}).reset_index(drop=True)


# ===========================
# TOP CHART: DAILY SINCE 2020 
# ===========================
_PRICE_DAILY_CACHE: dict[tuple[str, str, str], pd.DataFrame] = {}

# helparound in case yfinance returns weird close columns
def _close_series_from_hist(hist: pd.DataFrame) -> pd.Series:
    """
    yfinance sometimes returns:
    - normal columns: Close as Series
    - MultiIndex columns
    - Close as DataFrame
    Return a 1D Series safely.
    """
    if hist is None or hist.empty:
        return pd.Series(dtype=float)

    if isinstance(hist.columns, pd.MultiIndex):
        try:
            close_df = hist.xs("Close", axis=1, level=0, drop_level=True)
            if isinstance(close_df, pd.DataFrame):
                return close_df.iloc[:, 0].dropna()
            return close_df.dropna()
        except Exception:
            return pd.Series(dtype=float)

    if "Close" not in hist.columns:
        return pd.Series(dtype=float)

    close = hist["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.dropna()

# getting daily close prices for a ticker between start and end date
def get_daily_close_range(ticker: str, start_dt: date, end_dt: date) -> pd.DataFrame:
    t = str(ticker).strip().upper()
    start_dt = max(start_dt, CHART_START_DATE)
    end_dt = min(end_dt, date.today())

    key = (t, start_dt.isoformat(), end_dt.isoformat())
    if key in _PRICE_DAILY_CACHE:
        return _PRICE_DAILY_CACHE[key]

    end_plus = end_dt + timedelta(days=1)  # to include end date

    try:
        hist = yf.download(
            tickers=t,
            start=start_dt.isoformat(),
            end=end_plus.isoformat(),
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
    except Exception: 
        hist = pd.DataFrame()

    # extract close series safely
    close = _close_series_from_hist(hist)
    if close is None or len(close) == 0:
        out = pd.DataFrame({"ticker": [], "date": [], "close": []})
        _PRICE_DAILY_CACHE[key] = out
        return out

# assigning date and close columns
    df = close.rename("close").reset_index()
    date_col = "Date" if "Date" in df.columns else df.columns[0]
    df = df.rename(columns={date_col: "date"})

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["date", "close"]).sort_values("date")

    df["ticker"] = t
    df = df[["ticker", "date", "close"]]

    _PRICE_DAILY_CACHE[key] = df
    return df


# ===========================
# UI: PAGE 1 
# ===========================

# UI: PAGE 1 (PORTFOLIO MANAGEMENT)
page1_ui = ui.page_fluid(
    ui.h3("Add Investments and track the performance", style="margin:0 0 10px 0; font-weight:900;"),
    # LAYOUT WITH SIDEBAR default filters
    ui.layout_sidebar(
        ui.sidebar(
            ui.input_selectize( # dropdowqn select ticker
            "ticker",
            "Share ticker",
            choices=MASTER_CHOICES,
            selected="ABT" if "ABT" in MASTER_CHOICES else (MASTER_CHOICES[0] if MASTER_CHOICES else ""),
            multiple=False,
            options={"placeholder": "Type or pick ticker (e.g. LLY, ABT...)", "create": True, "persist": False},
            ),
            ui.input_numeric("shares", "Number of shares", value=163, min=1),
            ui.input_date("purchase_date", "Purchase date", value="2021-05-21"),
            ui.input_numeric("purchase_price", "Purchase price per share (optional)", value=None, min=0),
            ui.input_select(
                "inv_type",
                "Investment type",
                {
                    "Tax Reduction": "Tax Reduction (40% tax back)",
                    "ESPP": "ESPP (15% discount)",
                    "Bonus": "Bonus (no tax/discount)",
                    "Regular": "Regular investment",
                },
            ), # action buttons
            ui.input_action_button("add_inv", "Add investment"),
            ui.input_action_button("clear_all", "Clear portfolio"),
            ui.hr(),
            ui.input_action_button("refresh_now", "Refresh prices/calcs"),
            ui.output_text("refresh_status"),
            width=300,
        ),

        # MAIN AREA size of cards and plots
        ui.div(
            ui.card(
                ui.h4("Investment Preview"),
                ui.output_ui("kpi_row"),
                output_widget("timeline_plot"),
                height="900px",
            ),

            ui.card(
                ui.h4("Portfolio details (income calculated up to current year)"),
                ui.output_data_frame("portfolio_table"),
                height="420px",
            ),
        ),
    ),
)


# ===========================
# UI: PAGE 2 
# ===========================

# same as page 1 but different filters and plots
page2_ui = ui.page_fluid(
    ui.h3("Total Investment per Company", style="margin:0 0 10px 0; font-weight:900;"),

    ui.layout_sidebar(
        ui.sidebar(
            ui.input_date_range("sum_date", "Purchase date range", start=None, end=None),
            ui.input_select(
                "sum_year",
                "Year from",
                choices=["All"] + [str(y) for y in range(CHART_START_DATE.year, 2041)],
                selected="All",
            ),
            ui.input_select(
                "sum_type",
                "Investment type",
                choices=["All", "Tax Reduction", "ESPP", "Bonus", "Regular"],
                selected="All",
            ),
            ui.input_select("sum_focus_ticker", "Ticker", choices=["All"], selected="All"),
            ui.hr(),
            ui.input_action_button("refresh_now_2", "Refresh prices/calcs"),
            ui.output_text("refresh_status_2"),
            width=300,
        ),

        # MAIN AREA (everything that should be to the right of the sidebar)
        ui.div(
            # CSS 
            ui.tags.style("""
/* KPI ticker buttons (radio as pills) */
#sum_ticker_btns .form-check { display:inline-block; margin-right: 8px; }
#sum_ticker_btns input[type="radio"] { display:none; }
#sum_ticker_btns label {
  display:inline-block;
  padding: 6px 10px;
  border: 1px solid #ddd;
  border-radius: 999px;
  cursor: pointer;
  user-select: none;
  font-size: 0.9rem;
}
#sum_ticker_btns input[type="radio"]:checked + label {
  border-color: #999;
  font-weight: 600;
}

/* hide leftover radio circle/legend spacing */
#sum_ticker_btns input[type="radio"],
#sum_ticker_btns .form-check-input { display: none !important; }

#sum_ticker_btns fieldset,
#sum_ticker_btns legend {
  display: none !important;
  padding: 0 !important;
  margin: 0 !important;
}

#sum_ticker_btns .shiny-input-radiogroup,
#sum_ticker_btns .form-group {
  margin: 0 !important;
  padding: 0 !important;
}
"""),

            ui.card(
                ui.h4("Investment Value by Ticker over time"),
                ui.output_ui("sum_ticker_cards"),
                output_widget("summary_timeline_plot"),
                height="1000px",
            ),

            ui.card(
                ui.h4("Totals per Ticker - Today"),
                ui.output_data_frame("summary_table"),
                height="260px",
            ),
        ),
    ),
)



# ===========================
# ===========================
# UI: PAGE 3 (TOP100 & CHART)
# ===========================

#similar to page 1 but different filters and plots
page3_ui = ui.page_fluid(
    ui.h3("New Investment opportunities and Top 100 Best Performing Companies in the past 5 years", 
          style="margin:0 0 10px 0; font-weight:900;"),
    ui.layout_sidebar(
        ui.sidebar(
            ui.input_date_range("chart_date", "Chart date range", start=CHART_START_DATE, end=date.today()),
            ui.input_selectize(
                "tickers_selected",
                "Company chart selection (multi)",
                choices=MASTER_CHOICES,
                selected=["LLY", "ABT", "ABBV"],
                multiple=True,
                options={"placeholder": "Select tickers...", "create": True, "persist": False},
            ),
            ui.input_selectize( # adding option to filter top100 table
                "company_filter",
                "Filter Top100 companies table",
                choices=[],          # filled from CSV 
                selected="",
                multiple=False,
                options={"placeholder": "All (type ticker e.g. LLY, AMD...)", "create": False},
            ),

            ui.hr(),
            ui.input_action_button("export_chart_csv", "Export chart data to CSV"),
            ui.output_text("export_status"),
            ui.hr(),
            ui.input_action_button("refresh_now_3", "Refresh prices/calcs"),
            ui.output_text("refresh_status_3"),
            width=320,
        ),
        ui.card(
            ui.output_ui("top100_price_cards"),
            height="180px",
        ),
        ui.card(
            output_widget("price_5y_plot"),
            height="900px",
        ),
        ui.card(
            ui.h4("Top 100 Companies list"),
            ui.output_data_frame("top100_table"),
            height="520px",
        ),
 
    ),
)

# theme design CSS

MAIN_THEME_CSS = ui.tags.style("""
:root{
  --sidebar-bg:#f1e7d2;      /* beige */
  --topbar-bg:#e3d4bd;       /* slightly darker beige (greige) */
  --text:#2f2a24;
  --border:#d7c9b1;
  --wine:#722F37;
}

/* ---- NAVBAR / TOP MENU ---- */
.navbar{
  background: var(--topbar-bg) !important;
  border-bottom: 1px solid var(--border) !important;
}
.navbar .navbar-brand{
  color: var(--wine) !important;
  font-weight: 900 !important;
  font-size: 1.92rem !important; /* +20% */
  letter-spacing: 0.2px;
}
.navbar .nav-link{
  color: var(--text) !important;
  font-size: 1.2rem !important;  /* +20% page titles */
}
.navbar .nav-link:hover{
  opacity: 0.85;
}
.navbar .nav-link.active{
  color: var(--wine) !important;
  font-weight: 700;
}

/* ---- LEFT SIDEBAR  ---- */
.sidebar{
  background: var(--sidebar-bg) !important;
  color: var(--text) !important;
  border-right: 1px solid var(--border) !important;
}
.sidebar label,
.sidebar .form-label,
.sidebar h3,
.sidebar h4,
.sidebar h5,
.sidebar h6{
  color: var(--text) !important;
}

/* Sidebar inputs */
.sidebar .form-control,
.sidebar .selectize-control .selectize-input,
.sidebar .selectize-dropdown{
  background: #ffffff !important;
  color: #222 !important;
  border-color: var(--border) !important;
}

/* General headings ( +20% ) */
h3 { font-size: 1.38rem; }
h4 { font-size: 1.20rem; }

/* Moving page tabs a bit to the right */
.navbar-nav{
  margin-left: 90px !important;
}

/* space for footer */
body{
  padding-bottom: 64px;
}

/* Footer */
.app-footer{
  position: fixed;
  left: 0;
  bottom: 0;
  width: 100%;
  background: var(--topbar-bg);
  border-top: 1px solid var(--border);
  padding: 8px 14px;
  font-size: 0.95rem;
  color: var(--text);
  display: flex;
  justify-content: center;
  gap: 18px;
  z-index: 9999;
}
.app-footer a{
  color: var(--wine);
  text-decoration: none;
  font-weight: 700;
}
.app-footer a:hover{
  text-decoration: underline;
}

""")


# ===========================
# MAIN UI STRUCTURE
# ===========================
app_ui = ui.page_fluid(
    MAIN_THEME_CSS,
    ui.page_navbar(
        ui.nav_panel("Portfolio Planner", page1_ui),
        ui.nav_panel("Portfolio Summary", page2_ui),
        ui.nav_panel("New Investment", page3_ui),
        title="My Finance App",
    ),
    ui.tags.div(
        ui.tags.span("Created by Kasia Mc Art"),
        ui.tags.a(
            "✉ kasia.79@hotmail.com",
            href="mailto:kasia.79@hotmail.com",
            title="Please share your feedback",
        ),
        class_="app-footer",
    ),
)


# ===========================
# SERVER LOGIC
# ===========================

# main logic for the app
def server(input, output, session):
    portfolio = reactive.Value(load_portfolio_local())

    refresh_token = reactive.Value(0)
    refresh_msg = reactive.Value("")
    export_msg = reactive.Value("")
# refresh button logic
    def do_refresh():
        _DIVIDENDS_CACHE.clear()
        _PRICE_DAILY_CACHE.clear()
        _PRICE_ON_DATE_CACHE.clear()
        _HIST_CLOSE_CACHE.clear()
        refresh_token.set(refresh_token.get() + 1)
        refresh_msg.set(f"Last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
     # filling company filter choices from top100 csv
    @reactive.effect
    def _fill_company_filter_choices():
        _ = refresh_token.get()  # refresh updates choices 
        try:
            df0 = pd.read_csv(TOP100_CSV)
            if df0.empty or "Ticker" not in df0.columns:
                return
            choices = sorted(
                df0["Ticker"].astype(str).str.upper().str.strip().dropna().unique().tolist()
            )
            ui.update_selectize("company_filter", choices=choices, selected="", server=True)
        except Exception:
            pass
 
    # updating summary ticker choices based on portfolio entries   
    @reactive.effect
    def _update_sum_ticker_choices():
        _ = refresh_token.get()
        data = portfolio.get() or []
        tickers = sorted({(e.get("Ticker") or "").upper().strip() for e in data if e.get("Ticker")})
        # check if Shiny python supports updating via ui.update_select in newer versions; if not, use ui.update_selectize
        ui.update_select("sum_focus_ticker", choices=["All"] + tickers, selected="All")

    # refresh buttons effects
    @reactive.effect
    @reactive.event(input.refresh_now)
    def _refresh_btn_1():
        do_refresh()

    @reactive.effect
    @reactive.event(input.refresh_now_2)
    def _refresh_btn_2():
        do_refresh()

    @reactive.effect
    @reactive.event(input.refresh_now_3)
    def _refresh_btn_3():
        do_refresh()

    @output
    @render.text
    def refresh_status():
        return refresh_msg.get()

    @output
    @render.text
    def refresh_status_2():
        return refresh_msg.get()

    @output
    @render.text
    def refresh_status_3():
        return refresh_msg.get()

    @output
    @render.text
    def export_status():
        return export_msg.get()

    # =======================
    # Page 1: add / clear
    # =======================
    # add investment effect
    @reactive.effect
    @reactive.event(input.add_inv)
    def add_investment():
        ticker = (input.ticker() or "").upper().strip()
        shares = input.shares()
        inv_type = input.inv_type()
        purchase_date = input.purchase_date()
        purchase_price = input.purchase_price()

        if not ticker or shares is None or shares <= 0 or purchase_date is None:
            return

        current = portfolio.get() or []
        inv_name = f"{ticker} ({purchase_date.isoformat()})"  # legend shows purchase date

        entry = {
            "Ticker": ticker,
            "Shares": as_float(shares, 0.0),
            "InvType": inv_type,
            "PurchaseDate": purchase_date.isoformat(),
            "PurchasePrice": as_float(purchase_price, default=np.nan) if purchase_price is not None else None,
            "InvestmentName": inv_name,
        }

        portfolio.set(current + [entry])
        save_portfolio_local(portfolio.get())
        refresh_token.set(refresh_token.get() + 1)

    # clear portfolio effect
    @reactive.effect
    @reactive.event(input.clear_all)
    def clear_portfolio():
        portfolio.set([])
        save_portfolio_local([])
        refresh_token.set(refresh_token.get() + 1)

    # =======================
    # Page 1 outputs
    # =======================
    
    @output
    @render.data_frame
    def portfolio_table():
        _ = refresh_token.get()
        data = portfolio.get()
        if not data:
            return render.DataGrid(pd.DataFrame({"Info": ["No investments yet. Use the form in the sidebar."]}), filters=False)

        current_year = datetime.today().year
        summaries = [simulate_investment(entry, horizon_year=current_year)[1] for entry in data]
        return render.DataGrid(pd.DataFrame(summaries), filters=False)

    # =======================
    # Page 1 timeline plot
    # =======================
    
    @output
    @render_plotly
    def timeline_plot():
        _ = refresh_token.get()
        data = portfolio.get()

        if not data:
            fig = go.Figure()
            fig.update_layout(title="No data to display", height=400)
            return fig

        horizon_year = 2040

        # build df_all for all investments
        all_rows = [simulate_investment(entry, horizon_year=horizon_year)[0] for entry in data]
        df_all = pd.concat(all_rows, ignore_index=True)

        # Map purchase price per investment for hover
        inv_to_pp = {}
        for e in data:
            inv_name = str(e.get("InvestmentName", "")).strip()
            tkr = str(e.get("Ticker", "")).strip().upper()
            pds = e.get("PurchaseDate")
            try:
                purchase_dt = datetime.fromisoformat(pds) if pds else datetime.today()
            except Exception:
                purchase_dt = datetime.today()

            pp = e.get("PurchasePrice")
            if pp is None or (isinstance(pp, float) and np.isnan(pp)):
                pp = price_on_date_cached(tkr, purchase_dt)
            inv_to_pp[inv_name] = as_float(pp, 0.0)

        df_all["PurchasePrice"] = df_all["Investment"].map(inv_to_pp).fillna(0.0)

        # plot 
        colors = palette_for_n(int(df_all["Investment"].nunique()))
        fig = px.bar(
            df_all,
            x="Year",
            y="Value",
            color="Investment",
            color_discrete_sequence=colors,
            title="",
            labels={"Year": "Year", "Value": "Value (€)", "Investment": "Investment"},
            custom_data=["PurchasePrice"],
        )

        fig.update_layout(
            barmode="stack",
            height=720,
            margin=dict(l=60, r=20, t=80, b=110),  # extra top space for labels
            legend_title_text="Investment",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        # hoover template
        fig.update_traces(
            hovertemplate="%{fullData.name}<br>Purchase price=€%{customdata[0]:,.0f}<br>Value=€%{y:,.0f}<extra></extra>"
        )

        fig.update_xaxes(
            title_text="Year",
            showticklabels=True,
            tickmode="linear",
            dtick=1,
            tickangle=0,
            automargin=True,
        )
        fig.update_yaxes(automargin=True)

        # ===== totals per year labels above bars =====
        totals = (
            df_all.groupby("Year", as_index=False)["Value"]
            .sum()
            .sort_values("Year")
        )

        fig.add_trace(
            go.Scatter(
                x=totals["Year"],
                y=totals["Value"],
                mode="text",
                text=[f"€{v/1000:.1f}k" for v in totals["Value"]],
                textposition="top center",
                showlegend=False,
                hoverinfo="skip",
            )
        )

        fig.update_yaxes(range=[0, float(totals["Value"].max()) * 1.15])

        return fig

    # =======================
    # Page 1 KPI row
    # =======================
    
    @output
    @render.ui
    def kpi_row():
        _ = refresh_token.get()
        entries = portfolio.get() or []
        if not entries:
            return ui.div()
        # assigning dates
        now = datetime.today()
        days_180_ago = now - timedelta(days=180)
        days_30_ago = now - timedelta(days=30)
        # comparing KPIs at different dates
        now_kpi = compute_portfolio_kpis(entries, now)
        kpi_180 = compute_portfolio_kpis(entries, days_180_ago)
        kpi_30 = compute_portfolio_kpis(entries, days_30_ago)
        # predicted 2040 values
        now_pred_2040 = compute_predicted_total_2040(entries, now)
        old_pred_2040 = now_pred_2040
        pred_180 = compute_predicted_total_2040(entries, days_180_ago)
        def fmt_money(x):
            return f"€{float(x):,.2f}"

        # assigning icons and colors for deltas
        def delta_span(now_val, old_val):
            d = float(now_val) - float(old_val)
            if abs(d) < 0.005:
                return ui.tags.span("→  €0.00", style="color:#666; font-weight:600; margin-left:8px;")
            if d > 0:
                return ui.tags.span(f"▲  {fmt_money(d)}", style="color:#1b7f3a; font-weight:700; margin-left:8px;")
            return ui.tags.span(f"▼  {fmt_money(abs(d))}", style="color:#b00020; font-weight:700; margin-left:8px;")

        style = ui.tags.style("""
        .kpi-wrap { display:flex; gap:12px; margin: 8px 0 12px 0; flex-wrap:wrap; }
        .kpi-card { flex:1; min-width:220px; border:1px solid #e6e6e6; border-radius:12px; padding:10px 12px; background:#fafafa; }
        .kpi-title { font-size:0.85rem; color:#555; margin:0; }
        .kpi-value { font-size:1.25rem; font-weight:800; margin:2px 0 0 0; }
        .kpi-sub { font-size:0.8rem; color:#666; margin-top:2px; }
        """)

        # KPI cards layout
        return ui.TagList(
            style,
            ui.tags.div(
                {"class": "kpi-wrap"},
                ui.tags.div(
                    {"class": "kpi-card"},
                    ui.tags.p("Total Investment Value", class_="kpi-title"),
                    ui.tags.p(fmt_money(now_kpi["total_investment"]), class_="kpi-value"),
                    ui.tags.p(ui.tags.span("vs 180 days ago"), delta_span(now_kpi["total_investment"], kpi_180["total_investment"]), class_="kpi-sub"),
                ),
                ui.tags.div(
                    {"class": "kpi-card"},
                    ui.tags.p("Total Dividends", class_="kpi-title"),
                    ui.tags.p(fmt_money(now_kpi["total_dividends"]), class_="kpi-value"),
                    ui.tags.p(ui.tags.span("vs 180 days ago"), delta_span(now_kpi["total_dividends"], kpi_180["total_dividends"]), class_="kpi-sub"),
                ),
                ui.tags.div(
                    {"class": "kpi-card"},
                    ui.tags.p("Total Profit if sold today", class_="kpi-title"),
                    ui.tags.p(fmt_money(now_kpi["total_profit"]), class_="kpi-value"),
                    ui.tags.p(ui.tags.span("vs 30 days ago"), delta_span(now_kpi["total_profit"], kpi_30["total_profit"]), class_="kpi-sub"),
                ),
                ui.tags.div(
                    {"class": "kpi-card"},
                    ui.tags.p("Predicted Portfolio Value (2040)", class_="kpi-title"),
                    ui.tags.p(fmt_money(now_pred_2040), class_="kpi-value"),
                    ui.tags.p(ui.tags.span("vs 180 days ago"), delta_span(now_pred_2040, old_pred_2040), class_="kpi-sub"),
                ),
            ),
        )



    # =======================
    # Page 2: Summary filters
    # =======================
    
    # filtered portfolio entries based on summary filters
    @reactive.calc
    def filtered_portfolio_entries() -> list[dict]:
        _ = refresh_token.get()
        data = portfolio.get() or []
        if not data:
            return []

        dr = input.sum_date()
        start = dr[0] if dr and dr[0] else None
        end = dr[1] if dr and dr[1] else None

        # Investment type dropdown
        # "All" - no filtering
        try:
            type_sel = input.sum_type()
        except Exception:
            type_sel = "All"
        type_sel = (str(type_sel).strip() if type_sel else "All")

        # Year filter (purchase year)
        year_sel = (input.sum_year() if hasattr(input, "sum_year") else "All") or "All"
        year_sel = str(year_sel)

        # Focus ticker (from ticker buttons)
        try:
            focus_ticker = input.sum_focus_ticker()
        except Exception:
            focus_ticker = "All"
        focus_ticker = (str(focus_ticker).upper().strip() if focus_ticker else "All")

        out = []
        for e in data:
            tkr = (e.get("Ticker") or "").upper().strip()
            typ = str(e.get("InvType") or "").strip()
            pds = e.get("PurchaseDate")

            try:
                pd_dt = datetime.fromisoformat(pds) if pds else None
            except Exception:
                pd_dt = None

            # ticker buttons filter
            if focus_ticker != "ALL" and tkr != focus_ticker:
                continue

            # dropdown type filter
            if type_sel != "All" and typ != type_sel:
                continue

            # year filter
            if year_sel != "All":
                if pd_dt is None or pd_dt.year < int(year_sel):
                    continue

            # date range filter
            if start and pd_dt and pd_dt.date() < start:
                continue
            if end and pd_dt and pd_dt.date() > end:
                continue

            out.append(e)

        return out
    
# =======================
#total_income_to_now = (capital_gain_to_now - cgt_today) + tax_saved + espp_benefit + dividends_to_now
# =======================

    # summary table output    
    @output
    @render.data_frame
    def summary_table():
        _ = refresh_token.get()
        data = filtered_portfolio_entries()
        if not data:
            return render.DataGrid(pd.DataFrame({"Info": ["No matching investments. Adjust filters."]}), filters=False) #error info

        current_year = datetime.today().year
        summaries = [simulate_investment(entry, horizon_year=current_year)[1] for entry in data]
        df = pd.DataFrame(summaries)

        # price today per ticker
        df["Price Today"] = df["Ticker"].apply(lambda t: as_float(price_on_date_cached(t, datetime.today()), np.nan))

        # ensure numeric
        num_cols = ["Shares", "Tax Saved", "ESPP 15% Benefit", "Dividends", "CGT Today", "Total Income Net", "Purchase Price", "Price Today"]
        for c in num_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

        # weighted purchase price helper
        df["_cost"] = df["Shares"] * df["Purchase Price"]

        # ### NEW: current value per row (shares * price today)
        df["_current_value"] = df["Shares"] * df["Price Today"]

        def inv_type_rollup(s: pd.Series) -> str:
            uniq = sorted({str(x) for x in s.dropna().tolist()})
            return uniq[0] if len(uniq) == 1 else "Mixed"

        agg = (
            df.groupby("Ticker", as_index=False)
            .agg(
                Shares=("Shares", "sum"),
                _cost=("_cost", "sum"),
                Inv_Type=("Inv Type", inv_type_rollup),
                Tax_Saved=("Tax Saved", "sum"),
                ESPP_15_Benefit=("ESPP 15% Benefit", "sum"),
                Dividends=("Dividends", "sum"),
                #CGT_Today=("CGT Today", "sum"),
                Total_Income=("Total Income Net", "sum"),
                # total investment value today per ticker
                Total_Inv=("_current_value", "sum"),
            )
        )

        # CGT per ticker
        gain = (agg["Total_Inv"] - agg["_cost"]).clip(lower=0.0)
        agg["CGT Today"] = (gain * 0.33 - 1070.0).clip(lower=0.0)

        # weighted avg purchase price
        agg["Purchase Price"] = np.where(agg["Shares"] > 0, agg["_cost"] / agg["Shares"], 0.0)

        # droping columns
        agg = agg.drop(columns=["_cost"]).rename(
            columns={
            "Inv_Type": "Inv Type",
            "Tax_Saved": "Tax Saved",
            "ESPP_15_Benefit": "ESPP 15% Benefit",
            "CGT_Today": "CGT Today",
            "Total_Income": "Total Income Net",
            "Total_Inv": "Total Inv",
        })

        # rounding to no decimals
        for c in ["Purchase Price", "Shares", "Tax Saved", "ESPP 15% Benefit", "Dividends", "CGT Today", "Total Income Net", "Total Inv"]:
            if c in agg.columns:
                agg[c] = pd.to_numeric(agg[c], errors="coerce").fillna(0.0).round(0)

        # adding Total Inv in the table
        return render.DataGrid(
            agg[["Ticker","Shares","Total Inv","Inv Type","Tax Saved","ESPP 15% Benefit","Dividends","CGT Today","Total Income Net"]],
            filters=False
        )

    # =======================
    # Page 2: Ticker buttons & KPI cards
    # =======================
    
    @output
    @render.ui
    def sum_ticker_buttons():
        _ = refresh_token.get()
        data = portfolio.get() or []
        if not data:
            return ui.tags.div(id="sum_ticker_btns")

        dr = input.sum_date()
        start = dr[0] if dr and dr[0] else None
        end = dr[1] if dr and dr[1] else None
        types_sel = list(input.sum_types() or [])
        year_sel = (input.sum_year() if hasattr(input, "sum_year") else "All") or "All"
        year_sel = str(year_sel)

        # gathering tickers based on filters
        tickers = []
        for e in data:
            tkr = (e.get("Ticker") or "").upper().strip()
            typ = e.get("InvType")
            pds = e.get("PurchaseDate")
            try:
                pd_dt = datetime.fromisoformat(pds) if pds else None
            except Exception:
                pd_dt = None

            if types_sel and typ not in types_sel:
                continue
            if year_sel != "All":
                if pd_dt is None or pd_dt.year < int(year_sel):
                    continue
            if start and pd_dt and pd_dt.date() < start:
                continue
            if end and pd_dt and pd_dt.date() > end:
                continue
            if tkr:
                tickers.append(tkr)

        choices = ["All"] + sorted(set(tickers))
        return ui.tags.div(
            ui.input_radio_buttons(
                "sum_focus_ticker",
                None,
                choices=choices,
                selected="All",
                inline=True,
            ),
            id="sum_ticker_btns",
        )
    # KPI cards for selected tickers
    @output
    @render.ui
    def sum_price_kpis():
        data = filtered_portfolio_entries()
        tickers = sorted({(e.get("Ticker") or "").upper().strip() for e in data if e.get("Ticker")})

        focus = (input.sum_focus_ticker() or "All").upper().strip()
        if focus != "ALL":
            tickers = [t for t in tickers if t == focus]

        if not tickers:
            return ui.div()

        cards = []
        for t in tickers:
            dfp = get_daily_close_range(t, CHART_START_DATE, date.today())
            if dfp is None or dfp.empty:
                price = ma30 = ma100 = None
            else:
                s = pd.to_numeric(dfp["close"], errors="coerce").dropna()
                price = float(s.iloc[-1]) if len(s) else None
                ma30  = float(s.rolling(30).mean().iloc[-1]) if len(s) >= 30 else None
                ma100 = float(s.rolling(100).mean().iloc[-1]) if len(s) >= 100 else None

            def fmt(x):
                return "—" if x is None else f"${x:,.2f}"

            cards.append(
                ui.div(
                    ui.h6(t, style="margin:0 0 6px 0; font-weight:700;"),
                    ui.div(f"Price today: {fmt(price)}"),
                    ui.div(f"MA 30D: {fmt(ma30)}"),
                    ui.div(f"MA 100D: {fmt(ma100)}"),
                    style="min-width:240px; padding:12px 14px; border:1px solid #e6e6e6; border-radius:14px;"
                )
            )

        return ui.div(*cards, style="display:flex; gap:12px; flex-wrap:wrap; margin:10px 0 14px 0;")

    # =======================
    # Page 2: Ticker cards with investment values
    # ======================
    
    # KPI cards for total investment by ticker over time
    @output
    @render.ui
    def sum_ticker_cards():
        _ = refresh_token.get()

        entries = filtered_portfolio_entries() or []
        if not entries:
            return ui.div()

        # assigning dates
        now = datetime.today()
        days_30_ago = now - timedelta(days=30)
        days_100_ago = now - timedelta(days=100)
        days_180_ago = now - timedelta(days=180)

        # calculations for all tickers to set price diff
        now_by_ticker = compute_ticker_investment_values(entries, now)
        old_by_ticker = compute_ticker_investment_values(entries, days_30_ago)
        old100_by_ticker = compute_ticker_investment_values(entries, days_100_ago)
        old180_by_ticker = compute_ticker_investment_values(entries, days_180_ago)

        def fmt_money(x):
            return f"€{float(x):,.2f}"

        def arrow_for_delta(d):
            if abs(d) < 0.005:
                return ui.tags.span("→", style="color:#666; font-weight:900; margin-right:6px;")
            if d > 0:
                return ui.tags.span("▲", style="color:#1b7f3a; font-weight:900; margin-right:6px;")
            return ui.tags.span("▼", style="color:#b00020; font-weight:900; margin-right:6px;")

        # KPI CSS  
        style = ui.tags.style("""
        .kpi-wrap { display:flex; gap:12px; margin: 8px 0 12px 0; flex-wrap:wrap; }
        .kpi-card { flex:1; min-width:240px; border:1px solid #e6e6e6; border-radius:12px; padding:10px 12px; background:#fafafa; }
        .kpi-title { font-size:0.85rem; color:#555; margin:0; }
        .kpi-value { font-size:1.25rem; font-weight:800; margin:2px 0 0 0; }
        .kpi-sub { font-size:0.8rem; color:#666; margin-top:2px; }
        """)

    # building cards per ticker
        cards = []
        for t, v_now in sorted(now_by_ticker.items(), key=lambda kv: kv[1], reverse=True):
            v_old = float(old_by_ticker.get(t, 0.0))
            v_now = float(v_now)
            d = v_now - v_old

            v_old30  = float(old_by_ticker.get(t, 0.0))
            v_old100 = float(old100_by_ticker.get(t, 0.0))
            v_old180 = float(old180_by_ticker.get(t, 0.0))

            d30  = v_now - v_old30
            d100 = v_now - v_old100
            d180 = v_now - v_old180

            # value row with arrow + ticker - same as page1 list
            title_line = ui.tags.div(
                {"style": "display:flex; align-items:center; gap:6px;"},
                arrow_for_delta(d),
                ui.tags.span(t, style="font-weight:800;"),
            )

            # logic for comparisement of value between days and the icons
            def delta_text(d):
                if abs(d) < 0.005:
                    return ui.tags.span("→  €0.00", style="color:#666; font-weight:700; margin-left:8px;")
                if d > 0:
                    return ui.tags.span(f"▲  {fmt_money(d)}", style="color:#1b7f3a; font-weight:800; margin-left:8px;")
                return ui.tags.span(f"▼  {fmt_money(abs(d))}", style="color:#b00020; font-weight:800; margin-left:8px;")

            delta30_txt  = delta_text(d30)
            delta100_txt = delta_text(d100)
            delta180_txt = delta_text(d180)

            # arrow on the title line – choose which delta drives the arrow 
            title_line = ui.tags.div(
                {"style": "display:flex; align-items:center; gap:6px;"},
                arrow_for_delta(d30),
                ui.tags.span(t, style="font-weight:800;"),
            )

            cards.append(
                ui.tags.div(
                    {"class": "kpi-card"},
                    ui.tags.p("Total Investment by Ticker (today)", class_="kpi-title"),
                    title_line,
                    ui.tags.p(fmt_money(v_now), class_="kpi-value"),
                    ui.tags.p(ui.tags.span("vs 30 days ago"),  delta30_txt,  class_="kpi-sub"),
                    ui.tags.p(ui.tags.span("vs 100 days ago"), delta100_txt, class_="kpi-sub"),
                    ui.tags.p(ui.tags.span("vs 180 days ago"), delta180_txt, class_="kpi-sub"),
                )
            )

        return ui.TagList(
            style,
            ui.tags.div({"class": "kpi-wrap"}, *cards)
        )

    # total invested + total income net cards for selected tickers
    @output
    @render.ui
    def sum_ticker_kpis():
        data = filtered_portfolio_entries()
        if not data:
            return ui.tags.div()
        # calculating totals for selected tickers
        current_year = datetime.today().year
        summaries = [simulate_investment(entry, horizon_year=current_year)[1] for entry in data]
        df = pd.DataFrame(summaries)
        df["Price Today"] = df["Ticker"].apply(lambda t: round(as_float(price_on_date_cached(t, datetime.today()), np.nan), 2))

        for c in ["Shares", "Purchase Price", "Total Income Net"]:
            df[c] = pd.to_numeric(df.get(c, 0), errors="coerce").fillna(0.0)

        df["_cost"] = df["Shares"] * df["Purchase Price"]
        total_cost = float(df["_cost"].sum())
        total_income = float(pd.to_numeric(df.get("Total Income Net", 0), errors="coerce").fillna(0.0).sum())

        try:
            focus = input.sum_focus_ticker()
        except Exception:
            focus = "All"

        return ui.tags.div(
            ui.tags.div(ui.tags.strong(f"Selected: {focus}"), style="margin-bottom:6px;"),
            ui.tags.div(
                ui.tags.div(
                    [ui.tags.div("Total Invested"), ui.tags.div(f"€{total_cost:,.2f}")],
                    style="min-width:160px; padding:8px 10px; border:1px solid #eee; border-radius:12px;",
                ),
                ui.tags.div(
                    [ui.tags.div("Total Income Net"), ui.tags.div(f"€{total_income:,.2f}")],
                    style="min-width:160px; padding:8px 10px; border:1px solid #eee; border-radius:12px;",
                ),
                style="display:flex; gap:12px; flex-wrap:wrap;",
            ),
        )

    # =======================
    # Page 2: Summary timeline plot
    # =======================
    
    @output
    @render_plotly
    def summary_timeline_plot():
        _ = refresh_token.get()
        data = filtered_portfolio_entries()

        if not data:
            fig = go.Figure()
            fig.update_layout(title="No data to display", height=400)
            return fig

        horizon_year = 2040

        # build detailed rows
        all_rows = [simulate_investment(entry, horizon_year=horizon_year)[0] for entry in data]
        df_all = pd.concat(all_rows, ignore_index=True)

        if df_all.empty:
            fig = go.Figure()
            fig.update_layout(title="No data to plot", height=400)
            return fig

        # map Investment -> Ticker
        inv_to_ticker = {}
        for e in data:
            inv_name = str(e.get("InvestmentName", "")).strip()
            tkr = str(e.get("Ticker", "")).strip().upper()
            if inv_name:
                inv_to_ticker[inv_name] = tkr

        df_all["Ticker"] = df_all["Investment"].map(inv_to_ticker)

        # group by Year + Ticker
        df_tick = (
            df_all.dropna(subset=["Ticker"])
            .groupby(["Year", "Ticker"], as_index=False)["Value"]
            .sum()
        )

        if df_tick.empty:
            fig = go.Figure()
            fig.update_layout(title="No data to plot", height=400)
            return fig

        # plot
        colors = palette_for_n(int(df_tick["Ticker"].nunique()))

        fig = px.bar(
            df_tick,
            x="Year",
            y="Value",
            color="Ticker",
            color_discrete_sequence=colors,
            title="",
            labels={"Year": "Year", "Value": "Value (€)", "Ticker": "Ticker"},
        )

        fig.update_layout(
            barmode="stack",
            height=720,
            margin=dict(l=60, r=20, t=80, b=110),
            legend_title_text="Ticker",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )

        fig.update_xaxes(
            tickmode="linear",
            dtick=1,
            automargin=True,
        )

        fig.update_yaxes(
            tickformat=",~s",
            automargin=True,
        )

        # adding totals above bars
        totals = (
            df_tick.groupby("Year", as_index=False)["Value"]
            .sum()
            .sort_values("Year")
        )

        fig.add_trace(
            go.Scatter(
                x=totals["Year"],
                y=totals["Value"],
                mode="text",
                text=[f"€{v/1000:.1f}k" for v in totals["Value"]],
                textposition="top center",
                showlegend=False,
                hoverinfo="skip",
            )
        )

        fig.update_yaxes(range=[0, totals["Value"].max() * 1.15])

        return fig



    # =======================
    # Page 3: Top100 table + DAILY chart
    # =======================
    @reactive.calc
    def chart_prices_df() -> pd.DataFrame:
        _ = refresh_token.get()

        # getting selected tickers and date range
        selected = normalize_ticker_list(input.tickers_selected() or [])
        dr = input.chart_date()
        start = dr[0] if dr and dr[0] else CHART_START_DATE
        end = dr[1] if dr and dr[1] else date.today()

        start = max(start, CHART_START_DATE)
        end = min(end, date.today())
        # getting data for all selected tickers
        frames = [get_daily_close_range(t, start, end) for t in selected]
        df_all = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

        if not df_all.empty:
            df_all["date"] = pd.to_datetime(df_all["date"], errors="coerce")
            df_all["close"] = pd.to_numeric(df_all["close"], errors="coerce")
            df_all = df_all.dropna(subset=["date", "close"]).sort_values(["ticker", "date"])

        return df_all

    # export chart data to CSV
    @reactive.effect
    @reactive.event(input.export_chart_csv)
    def _export_chart():
        df = chart_prices_df()
        if df.empty:
            export_msg.set("No data to export.")
            return
        out_path = portfolio_path().parent / "chart_prices_debug.csv"
        df.to_csv(out_path, index=False)
        export_msg.set(f"Saved: {out_path}")

    # price cards for selected tickers
    @output
    @render.ui
    def top100_price_cards():
        _ = refresh_token.get()
        df_all = chart_prices_df()
        selected = normalize_ticker_list(input.tickers_selected() or [])
        if not selected:
            return ui.div()

        def fmt_price(x):
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return "—"
            return f"${float(x):,.2f}"

        # delta badge helper to show price change if
        def delta_badge(last_val, prev_val):
            if last_val is None or prev_val is None:
                return ui.tags.span("", style="margin-left:8px;")
            d = float(last_val) - float(prev_val)
            if abs(d) < 1e-9:
                return ui.tags.span("→ 0.00", style="color:#666; font-weight:700; margin-left:8px;")
            if d > 0:
                return ui.tags.span(f"▲ {abs(d):.2f}", style="color:#1b7f3a; font-weight:800; margin-left:8px;")
            return ui.tags.span(f"▼ {abs(d):.2f}", style="color:#b00020; font-weight:800; margin-left:8px;")

        style = ui.tags.style("""
        .price-wrap { display:flex; gap:12px; margin: 6px 0 6px 0; flex-wrap:wrap; }
        .price-card { flex:1; min-width:170px; border:1px solid #e6e6e6; border-radius:12px; padding:10px 12px; background:#fafafa; }
        .price-title { font-size:0.95rem; color:#555; margin:0; font-weight:800; }
        .price-value { font-size:1.25rem; font-weight:900; margin:2px 0 0 0; color: var(--wine); }
        .price-sub { font-size:0.80rem; color:#666; margin-top:2px; }
        """)

        cards = []
        for t in selected:
            df_t = df_all[df_all["ticker"] == t].sort_values("date")
            last_val = as_float(df_t["close"].iloc[-1], None) if not df_t.empty else None
            prev_val = as_float(df_t["close"].iloc[-2], None) if len(df_t) >= 2 else None
            last_date = df_t["date"].iloc[-1].date().isoformat() if not df_t.empty else "—"

            cards.append(
                ui.tags.div(
                    {"class": "price-card"},
                    ui.tags.p(t, class_="price-title"),
                    ui.tags.p(fmt_price(last_val), class_="price-value"),
                    ui.tags.p(
                        ui.tags.span(f"Latest: {last_date}"),
                        delta_badge(last_val, prev_val),
                        class_="price-sub",
                    ),
                )
            )

        return ui.TagList(style, ui.tags.div({"class": "price-wrap"}, *cards))

    # Top100 table output
    @output
    @render.data_frame
    def top100_table():
        _ = refresh_token.get()

        try:
            df = pd.read_csv(TOP100_CSV)
        except Exception:
            return render.DataGrid(
                pd.DataFrame({"Info": ["Top CSV not found (or empty). Check file path."]}),
                filters=False
            )

        if df is None or df.empty:
            return render.DataGrid(
                pd.DataFrame({"Info": ["Top CSV not found (or empty). Check file path."]}),
                filters=False
            )

        # Ensure ticker col
        if "Ticker" not in df.columns:
            maybe = [c for c in df.columns if c.lower() in ("ticker", "symbol")]
            if maybe:
                df = df.rename(columns={maybe[0]: "Ticker"})
            else:
                return render.DataGrid(pd.DataFrame({"Info": ["Ticker column not found in CSV."]}), filters=False)

        df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()

        # Filter box
        q = (input.company_filter() or "").strip().upper()
        if q:
            df = df[df["Ticker"].astype(str).str.upper() == q].copy()


        if df.empty:
            return render.DataGrid(pd.DataFrame({"Info": ["No matching tickers for this filter."]}), filters=False)

        df = df.drop(columns=["Region"], errors="ignore")
        
        # Keep key columns first
        screener_cols = [
            "AsOf", "Ticker", "Name", "Country", "Price", "Change", "Change %",
            "Volume", "Market cap", "52 week price % Change"
        ]
        base_cols = [c for c in screener_cols if c in df.columns]
        remaining = [c for c in df.columns if c not in base_cols]
        df = df[base_cols + remaining]

        #formating numeric columns
        change_num = pd.to_numeric(df["Change"], errors="coerce") if "Change" in df.columns else pd.Series([np.nan] * len(df), index=df.index)
        change_pct_num = pd.to_numeric(df["Change %"], errors="coerce") if "Change %" in df.columns else pd.Series([np.nan] * len(df), index=df.index)
        wk52_pct_num = (
            pd.to_numeric(
                df["52 week price % Change"].astype(str).str.replace("%", "", regex=False),
                errors="coerce"
            )
            if "52 week price % Change" in df.columns
            else pd.Series([np.nan] * len(df), index=df.index)
)

        vol_num = pd.to_numeric(df["Volume"], errors="coerce") if "Volume" in df.columns else pd.Series([np.nan] * len(df), index=df.index)
        mcap_num = pd.to_numeric(df["Market cap"], errors="coerce") if "Market cap" in df.columns else pd.Series([np.nan] * len(df), index=df.index)

        # formatting volume to M, Market cap to B ----------
        if "Volume" in df.columns:
            df["Volume"] = (vol_num / 1_000_000.0).map(lambda x: f"{x:,.2f}M" if pd.notna(x) else "")

        if "Market cap" in df.columns:
            df["Market cap"] = (mcap_num / 1_000_000_000.0).map(lambda x: f"{x:,.2f}B" if pd.notna(x) else "")

        # Conditional font colors for Change, Change %, 52 week price % Change
        styles = []

        if "Change" in df.columns:
            change_col_idx = list(df.columns).index("Change")
            pos_rows = change_num[change_num > 0].index.to_list()
            neg_rows = change_num[change_num < 0].index.to_list()

            if pos_rows:
                styles.append({"rows": pos_rows, "cols": [change_col_idx], "style": {"color": "#1b7f3a", "fontWeight": "700"}})
            if neg_rows:
                styles.append({"rows": neg_rows, "cols": [change_col_idx], "style": {"color": "#b00020", "fontWeight": "700"}})

        if "Change %" in df.columns:
            pct_col_idx = list(df.columns).index("Change %")
            pos_rows = change_pct_num[change_pct_num > 0].index.to_list()
            neg_rows = change_pct_num[change_pct_num < 0].index.to_list()

            if pos_rows:
                styles.append({"rows": pos_rows, "cols": [pct_col_idx], "style": {"color": "#1b7f3a", "fontWeight": "700"}})
            if neg_rows:
                styles.append({"rows": neg_rows, "cols": [pct_col_idx], "style": {"color": "#b00020", "fontWeight": "700"}})

        if "52 week price % Change" in df.columns:
            wk52_col_idx = list(df.columns).index("52 week price % Change")

            # red if dropped (< 0)
            neg_rows = wk52_pct_num[wk52_pct_num < 0].index.to_list()

            # green if > 100%
            huge_pos_rows = wk52_pct_num[wk52_pct_num > 100].index.to_list()

            if huge_pos_rows:
                styles.append({
                    "rows": huge_pos_rows,
                    "cols": [wk52_col_idx],
                    "style": {"color": "#1b7f3a", "fontWeight": "800"}
                })

            if neg_rows:
                styles.append({
                    "rows": neg_rows,
                    "cols": [wk52_col_idx],
                    "style": {"color": "#b00020", "fontWeight": "800"}
                })

        return render.DataGrid(df, filters=False, styles=styles)



     # Price 5Y plot output for selected tickers      
    @output
    @render_plotly
    def price_5y_plot():
        df_all = chart_prices_df()
        if df_all.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No data returned for the selected tickers/date range",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
            fig.update_layout(title="", height=860)
            return fig

        df_all = df_all.copy()
        df_all["date"] = pd.to_datetime(df_all["date"], errors="coerce")
        df_all = df_all.dropna(subset=["date", "close"])
        df_all["date_str"] = df_all["date"].dt.strftime("%Y-%m-%d")

        selected = normalize_ticker_list(input.tickers_selected() or [])
        colors = palette_for_n(len(selected))
        # plotting 
        fig = go.Figure()
        for i, t in enumerate(selected):
            df_t = df_all[df_all["ticker"] == t].sort_values("date")
            if df_t.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=df_t["date_str"],
                    y=df_t["close"],
                    mode="lines",
                    name=t,
                    line=dict(color=colors[i]),
                    hovertemplate=f"{t}<br>Date=%{{x}}<br>Close=%{{y:.2f}}<extra></extra>",
                )
            )

        if len(fig.data) == 0:
            fig.add_annotation(
                text="No traces added (check tickers/date range).",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )

        fig.update_layout(
            title="",
            height=860,
            xaxis_title="Date",
            yaxis_title="Share price ($)",
            legend_title_text="Ticker",
            hovermode="x unified",

            #removing background
            plot_bgcolor="rgba(0,0,0,0)",   # inside plot area
            paper_bgcolor="rgba(0,0,0,0)",  # outside plot area
        )

        fig.update_xaxes(type="date", rangeslider=dict(visible=True))
        return fig
    
    # Company detail table output
    @output
    @render.data_frame
    def company_detail_table():
        _ = refresh_token.get()

        # same base table data + same filter
        df = safe_load_top100(TOP100_CSV)
        if df.empty:
            return pd.DataFrame({"Info": ["Top100 CSV not found (or empty). Check file path."]})

        q = (input.company_filter() or "").strip().lower()
        if q:
            df = df[df["Ticker"].astype(str).str.lower().str.contains(q, na=False)].copy()

        tickers = df["Ticker"].astype(str).str.upper().tolist()

        N = min(20, len(tickers))

        rows = []
        for t in tickers[:N]:
            try:
                info = yf.Ticker(t).info or {}
            except Exception:
                info = {}

            rows.append({
                "Ticker": t,
                "Name": info.get("shortName") or info.get("longName") or "",
                "Country": info.get("country") or "",
                "Sector": info.get("sector") or "",
                "Industry": info.get("industry") or "",
                "Market Cap": as_float(info.get("marketCap"), 0.0),
                "P/E (TTM)": as_float(info.get("trailingPE"), np.nan),
                "Forward P/E": as_float(info.get("forwardPE"), np.nan),
                "Dividend Yield %": (as_float(info.get("dividendYield"), np.nan) * 100.0),
                "52W High": as_float(info.get("fiftyTwoWeekHigh"), np.nan),
                "52W Low": as_float(info.get("fiftyTwoWeekLow"), np.nan),
                "Beta": as_float(info.get("beta"), np.nan),
            })

        out = pd.DataFrame(rows)

        # tidy formatting
        if not out.empty:
            out["Market Cap"] = pd.to_numeric(out["Market Cap"], errors="coerce").fillna(0.0).round(0)
            for c in ["P/E (TTM)", "Forward P/E", "Dividend Yield %", "52W High", "52W Low", "Beta"]:
                out[c] = pd.to_numeric(out[c], errors="coerce")
                out[c] = out[c].round(2)

        return out

    
# ===========================
# App initialization
# ===========================

app = App(app_ui, server)




