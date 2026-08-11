# scripts/build_forbes_top100.py
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import pandas as pd
from yahooquery import Ticker


# ---------- Paths repo ----------
BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
DATA.mkdir(exist_ok=True)

TICKERS_CSV = DATA / "forbes_top100_tickers.csv"
OUT_CSV = DATA / "forbes_top100.csv"

BAND_LOW, BAND_HIGH = 20.0, 30.0   # 52-week change band of interest
RANGE_WEEK_MIN = 4.0               # a "swing week" = weekly high-low range >= 4%


def safe_num(x):
    try:
        return float(x)
    except Exception:
        return None


def main() -> None:
    base = pd.read_csv(TICKERS_CSV)
    tickers = base["Ticker"].astype(str).str.upper().str.strip().tolist()

    t = Ticker(tickers, asynchronous=False)

    price_mod = t.price
    detail_mod = t.summary_detail
    stats_mod = t.key_stats

    # Weekly history for range metrics (1 year, weekly bars)
    hist = t.history(period="1y", interval="1wk")

    rows = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for _, r in base.iterrows():
        tk = str(r["Ticker"]).upper().strip()
        p = price_mod.get(tk, {}) if isinstance(price_mod, dict) else {}
        d = detail_mod.get(tk, {}) if isinstance(detail_mod, dict) else {}
        s = stats_mod.get(tk, {}) if isinstance(stats_mod, dict) else {}
        if not isinstance(p, dict):
            p = {}
        if not isinstance(d, dict):
            d = {}
        if not isinstance(s, dict):
            s = {}

        price = safe_num(p.get("regularMarketPrice"))
        hi52 = safe_num(d.get("fiftyTwoWeekHigh"))
        lo52 = safe_num(d.get("fiftyTwoWeekLow"))
        pe_t = safe_num(d.get("trailingPE"))
        pe_f = safe_num(s.get("forwardPE"))

        # --- weekly range metrics + 12m change from history ---
        avg_wk_range = None
        swing_weeks = None
        chg_12m = None
        try:
            h = hist.loc[tk]
            if len(h) >= 10:
                rng = (h["high"] - h["low"]) / h["open"] * 100.0
                avg_wk_range = round(float(rng.mean()), 2)
                swing_weeks = int((rng >= RANGE_WEEK_MIN).sum())
                first_close = float(h["close"].iloc[0])
                last_close = float(h["close"].iloc[-1])
                if price is None:
                    price = round(last_close, 2)
                if first_close > 0:
                    chg_12m = round((last_close / first_close - 1) * 100.0, 1)
        except Exception:
            pass

        off_high = None
        if price is not None and hi52:
            off_high = round((price / hi52 - 1) * 100.0, 1)

        in_band = None
        if chg_12m is not None:
            in_band = BAND_LOW <= chg_12m <= BAND_HIGH

        rows.append({
            "AsOf": now,
            "Rank": r["Rank"],
            "Ticker": tk,
            "Name": r["Name"],
            "Sector": r["Sector"],
            "Price": price,
            "Above200": (price is not None and price > 200.0),
            "Change 12m %": chg_12m,
            "Band 20-30%": in_band,
            "52w High": hi52,
            "52w Low": lo52,
            "% off 52w High": off_high,
            "Trailing PE": pe_t,
            "Forward PE": pe_f,
            "Avg Weekly Range %": avg_wk_range,
            f"Weeks >= {RANGE_WEEK_MIN:.0f}%": swing_weeks,
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    got = out["Price"].notna().sum()
    print(f"Wrote {OUT_CSV} — {len(out)} tickers, {got} with live prices.")


if __name__ == "__main__":
    main()
