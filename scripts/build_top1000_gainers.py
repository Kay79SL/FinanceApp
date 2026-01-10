# scripts/build_top1000_gainers.py
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import pandas as pd
import json
from yahooquery import Screener, Ticker


# ---------- Paths repo ----------
BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
DATA.mkdir(exist_ok=True)

MASTER_TICKERS_CSV = DATA / "master_tickers.csv"
TOP_1000_CSV = DATA / "top_1000CSV.csv"


# Helping to get data
def read_master_tickers() -> set[str]:
    if not MASTER_TICKERS_CSV.exists():
        return set()
    df = pd.read_csv(MASTER_TICKERS_CSV)
    col = "Ticker" if "Ticker" in df.columns else df.columns[0]
    return set(df[col].astype(str).str.upper().str.strip().tolist())


def pick_screener_id() -> str:
    """Try to use 52-week gainers first; fallback to close equivalents."""
    s = Screener()
    avail = set(s.available_screeners)
    for cand in ["52_week_gainers", "recent_52_week_highs", "day_gainers", "gainers"]:
        if cand in avail:
            return cand
    # last resort: just use one known to exist from yahooquery docs
    return "recent_52_week_highs"


def to_pct(x):
    """Yahoo sometimes returns percent as decimal (0.12) or percent (12). Normalize to percent."""
    if pd.isna(x):
        return None
    try:
        x = float(x)
    except Exception:
        return None
    # heuristics: if within [-1, 1], assume decimal
    if -1.0 <= x <= 1.0:
        return x * 100.0
    return x


def safe_num(x):
    try:
        return float(x)
    except Exception:
        return None


def enrich_country_region(tickers: list[str]) -> pd.DataFrame:
    """
    Batch fetch asset_profile for country/region.
    Returns df: Ticker, Country, Region
    """
    if not tickers:
        return pd.DataFrame(columns=["Ticker", "Country", "Region"])

    t = Ticker(tickers)
    ap = t.asset_profile  # dict keyed by symbol

    rows = []
    if isinstance(ap, dict):
        for sym, info in ap.items():
            sym_u = str(sym).upper().strip()
            if isinstance(info, dict):
                rows.append({
                    "Ticker": sym_u,
                    "Country": info.get("country"),
                    "Region": info.get("region"),
                })
    return pd.DataFrame(rows)


def main():
    master = read_master_tickers()  # both markets universe 
    sc = Screener()
    sid = pick_screener_id()
    
    print("Using sid:", sid)
    print("sid available:", sid in set(Screener().available_screeners))


    # Pull up to 1000 rows from the screener
    payload = sc.get_screeners([sid], count=50)

    print("payload type:", type(payload))
    print("payload keys (if dict):", list(payload.keys())[:10] if isinstance(payload, dict) else "n/a")
    
    # Debug + fix when payload comes back as a string
    if isinstance(payload, str):
        print("Screener payload is STRING. First 500 chars:")
        print(payload[:500])
        try:
            payload = json.loads(payload)  # if it's JSON text
        except Exception:
            raise RuntimeError(f"Screener returned a string (not JSON). Message:\n{payload}")

    block = payload.get(sid, {})
    if isinstance(block, str):
        raise RuntimeError(f"Screener '{sid}' returned a string:\n{block[:500]}")
    quotes = (block or {}).get("quotes", [])

    df = pd.DataFrame(quotes)

    print("Rows returned:", len(df))
    print("Columns:", list(df.columns)[:30])

    if df.empty:
        raise RuntimeError(f"No data returned from screener '{sid}'")

    # Normalising core columns
    df["Ticker"] = df.get("symbol", "").astype(str).str.upper().str.strip()
    df["Name"] = df.get("shortName", df.get("longName", "")).astype(str)

    df["Price"] = df.get("regularMarketPrice").apply(safe_num)
    df["Change"] = df.get("regularMarketChange").apply(safe_num)
    df["Change %"] = df.get("regularMarketChangePercent").apply(to_pct)

    df["Volume"] = df.get("regularMarketVolume").apply(safe_num)
    df["Market cap"] = df.get("marketCap").apply(safe_num)

    # 52-week change can appear under different keys; try a few
    if "fiftyTwoWeekChangePercent" in df.columns:
        df["52 week price % Change"] = df["fiftyTwoWeekChangePercent"].apply(to_pct)
    elif "fiftyTwoWeekChange" in df.columns:
        df["52 week price % Change"] = df["fiftyTwoWeekChange"].apply(to_pct)
    else:
        df["52 week price % Change"] = None

    # Optional: keep only tickers from your master universe (FAST filtering)
    if master:
        df = df[df["Ticker"].isin(master)].copy()

    # Enrich Country/Region (batched)
    meta = enrich_country_region(df["Ticker"].tolist())
    df = df.merge(meta, on="Ticker", how="left")

    # Order + add AsOf
    df.insert(0, "AsOf", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    out = df[[
        "AsOf",
        "Ticker",
        "Name",
        "Country",
        "Price",
        "Change",
        "Change %",
        "Volume",
        "Market cap",
        "52 week price % Change",
        "Region",
    ]].copy()

    # Clean formatting
    for c in ["Price", "Change", "Change %", "52 week price % Change"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").round(2)
    out["Market cap"] = pd.to_numeric(out["Market cap"], errors="coerce")
    out["Volume"] = pd.to_numeric(out["Volume"], errors="coerce")

    out.to_csv(TOP_1000_CSV, index=False, encoding="utf-8")
    print(f"Saved {len(out)} rows to: {TOP_1000_CSV}")
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
