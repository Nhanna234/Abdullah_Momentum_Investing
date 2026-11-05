"""
Relative Momentum Lab — Streamlit app
=====================================
Relative momentum strategy (from "Unveiling Alpha") with configurable ranking,
entry filters and exit filters, on any of seven NSE indices.

  * Present holdings : runs the strategy from a start month up to the last completed
                       month and shows what it holds now + this month's orders.
  * Backtest         : month-by-month simulation between two months with charts
                       and analytics vs Nifty 50 / Nifty 500 / equal-weight universe.

Rules (each switchable in the sidebar)
  Score      : Close(t-skip) / Close(t-skip-lookback) - 1  on month-end closes
  Buy zone   : top N   |  Hold zone: top threshold  |  max N holdings
  Entry      : F1 close > EMA · F2 close/EMA-1 <= cap · F3 closing range (C-L)/(H-L) >= min
               (CR below min -> buy-stop at the month high during next month, or skip)
  Exit       : rank > threshold (always) · close < EMA · stop-loss from entry ·
               trailing stop from highest month-end close since entry
  Allocation : available cash / open slots; kept holdings are not rebalanced

Data: constituents from niftyindices.com / nsearchives.nseindia.com (official files);
stock OHLC from Yahoo Finance (yfinance); Nifty 50 / Nifty 500 index closes from the
niftyindices.com historical API with Yahoo as fallback.

Run locally:  pip install -r requirements.txt  &&  streamlit run app.py
"""

from __future__ import annotations

import io
import json
import time
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

# ════════════════════════════════════════════════════════════════════════════
# 1. CONSTANTS
# ════════════════════════════════════════════════════════════════════════════
INDEX_FILES = {
    "Nifty 50": "ind_nifty50list.csv",
    "Nifty 100": "ind_nifty100list.csv",
    "Nifty LargeMidcap 250": "ind_niftylargemidcap250list.csv",
    "Nifty Smallcap 250": "ind_niftysmallcap250list.csv",
    "Nifty 500": "ind_nifty500list.csv",
    "Nifty Microcap 250": "ind_niftymicrocap250_list.csv",
    "Nifty Total Market": "ind_niftytotalmarket_list.csv",
}
CONSTITUENT_MIRRORS = ["https://niftyindices.com/IndexConstituent/{f}",
                       "https://www.niftyindices.com/IndexConstituent/{f}",
                       "https://nsearchives.nseindia.com/content/indices/{f}"]
BENCH = {"Nifty 50": ("NIFTY 50", "^NSEI"), "Nifty 500": ("NIFTY 500", "^CRSLDX")}
NI_BASE = "https://www.niftyindices.com"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
IST = timezone(timedelta(hours=5, minutes=30))
FIELDS = ("Open", "High", "Low", "Close")

# validated categorical slots (strategy, Nifty 50, Nifty 500) + neutral for EW universe
COL = {"Strategy": "#2a78d6", "Nifty 50": "#eb6834", "Nifty 500": "#1baf7a",
       "Equal-weight universe": "#8a8986"}
GOOD, BAD = "#0ca30c", "#d03b3b"
DIVERGING = [[0.0, "#d03b3b"], [0.5, "#f0efec"], [1.0, "#2a78d6"]]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def last_completed_month() -> pd.Period:
    return pd.Period(datetime.now(IST).date(), "M") - 1


# ════════════════════════════════════════════════════════════════════════════
# 2. DATA
# ════════════════════════════════════════════════════════════════════════════
def _clean(s) -> str:
    return str(s).strip().upper().replace(".NS", "")


def _parse_constituents(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if "Symbol" not in df.columns:
        raise ValueError(f"No 'Symbol' column (columns: {list(df.columns)})")
    if "Series" in df.columns:
        df = df[df["Series"].astype(str).str.strip() == "EQ"]
    df["Symbol"] = df["Symbol"].map(_clean)
    for c in ("Company Name", "Industry"):
        if c not in df.columns:
            df[c] = ""
    return df.drop_duplicates("Symbol")[["Symbol", "Company Name", "Industry"]].reset_index(drop=True)


@st.cache_data(ttl=24 * 3600, show_spinner="Fetching index constituents…")
def load_constituents(index_name: str) -> tuple[pd.DataFrame, str]:
    fname, errors = INDEX_FILES[index_name], []
    for m in CONSTITUENT_MIRRORS:
        url = m.format(f=fname)
        try:
            r = requests.get(url, headers={"User-Agent": UA, "Accept": "text/csv,*/*"}, timeout=25)
            r.raise_for_status()
            return _parse_constituents(pd.read_csv(io.StringIO(r.text))), url
        except Exception as e:  # noqa: BLE001
            errors.append(f"{url}: {e}")
    raise RuntimeError("Could not download the constituent list. Upload it from the sidebar.\n"
                       + "\n".join(errors))


def parse_history(file) -> tuple[dict, list[str]]:
    """Report format: each column header is an effective date, values are symbols."""
    h = pd.read_csv(file)
    h.columns = pd.to_datetime(h.columns)
    lists = {d: [_clean(x) for x in h[d].dropna()] for d in sorted(h.columns)}
    return lists, sorted(set().union(*lists.values()))


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def load_ohlc(symbols: tuple[str, ...], start: str, end: str, adjusted: bool) -> dict:
    tickers = [f"{s}.NS" for s in symbols]
    parts = {f: [] for f in FIELDS}
    for i in range(0, len(tickers), 100):          # batches keep Yahoo from throttling
        chunk, df = tickers[i:i + 100], None
        for _ in range(2):
            try:
                df = yf.download(chunk, start=start, end=end, auto_adjust=adjusted,
                                 progress=False, threads=True, group_by="column")
                break
            except Exception:  # noqa: BLE001
                time.sleep(2)
        if df is None or df.empty:
            continue
        for f in FIELDS:
            x = df[f]
            if isinstance(x, pd.Series):
                x = x.to_frame(chunk[0])
            parts[f].append(x)
        time.sleep(0.5)
    if not parts["Close"]:
        raise RuntimeError("Yahoo Finance returned no price data.")
    out = {}
    for f in FIELDS:
        x = pd.concat(parts[f], axis=1)
        x.columns = [_clean(c) for c in x.columns]
        x = x.loc[:, ~x.columns.duplicated()]
        x.index = pd.to_datetime(x.index).tz_localize(None).normalize()
        out[f] = x.sort_index()
    return out


def _ni_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"})
    try:
        s.get(f"{NI_BASE}/reports/historical-data", timeout=15)
    except requests.RequestException:
        pass
    return s


def _ni_index(api_name: str, start: date, end: date) -> pd.Series:
    s, rows, cur = _ni_session(), [], start
    while cur <= end:
        nxt = min(cur + timedelta(days=180), end)
        cinfo = (f"{{'name':'{api_name}','startDate':'{cur:%d-%b-%Y}',"
                 f"'endDate':'{nxt:%d-%b-%Y}','indexName':'{api_name}'}}")
        r = s.post(f"{NI_BASE}/BackPage/getHistoricaldatatabletoString",
                   data=json.dumps({"cinfo": cinfo}), timeout=30,
                   headers={"Content-Type": "application/json; charset=utf-8",
                            "X-Requested-With": "XMLHttpRequest", "Origin": NI_BASE,
                            "Referer": f"{NI_BASE}/reports/historical-data"})
        r.raise_for_status()
        data = json.loads(r.text)
        if isinstance(data, dict):
            data = data.get("d", [])
            data = json.loads(data) if isinstance(data, str) else data
        rows += data
        cur = nxt + timedelta(days=1)
        time.sleep(0.3)
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["HistoricalDate"], format="%d %b %Y")
    ser = pd.to_numeric(df["CLOSE"], errors="coerce").set_axis(df["date"]).sort_index()
    return ser[~ser.index.duplicated(keep="last")].dropna()


def _resample(df, how: str):
    try:
        r = df.resample("ME")
    except ValueError:                             # pandas < 2.2
        r = df.resample("M")
    m = getattr(r, how)()
    m.index = m.index.to_period("M")
    return m


@st.cache_data(ttl=6 * 3600, show_spinner="Fetching Nifty 50 / Nifty 500 index closes…")
def load_benchmarks(start: str, end: str) -> tuple[pd.DataFrame, dict]:
    out, src = {}, {}
    s, e = pd.Timestamp(start).date(), pd.Timestamp(end).date()
    for name, (api, yh) in BENCH.items():
        try:
            ser = _ni_index(api, s, e)
            src[name] = "niftyindices.com"
        except Exception:  # noqa: BLE001
            try:
                df = yf.download(yh, start=start, end=end, auto_adjust=False, progress=False)
                ser = df["Close"]
                ser = ser.iloc[:, 0] if isinstance(ser, pd.DataFrame) else ser
                ser.index = pd.to_datetime(ser.index).tz_localize(None)
                src[name] = f"Yahoo ({yh})"
            except Exception as ex:  # noqa: BLE001
                src[name] = f"unavailable ({ex})"
                continue
        out[name] = _resample(ser.to_frame(name), "last")[name]
    return (pd.concat(out, axis=1) if out else pd.DataFrame()), src


# ════════════════════════════════════════════════════════════════════════════
# 3. STRATEGY ENGINE
# ════════════════════════════════════════════════════════════════════════════
def build_monthly(D: dict, P: dict) -> dict:
    C, H, L = _resample(D["Close"], "last"), _resample(D["High"], "max"), _resample(D["Low"], "min")
    Cf = C.ffill()
    ema = Cf.ewm(span=P["ema_span"], adjust=False).mean()
    score = C.shift(P["skip"]) / C.shift(P["skip"] + P["lookback"]) - 1
    rng = H - L
    cr = ((C - L) / rng).where(rng > 0, 1.0)
    return {"C": C, "Cf": Cf, "H": H, "L": L, "ema": ema, "score": score, "cr": cr,
            "ext": Cf / ema - 1}


def rank_month(M: dict, t, universe: list[str]):
    uni = [s for s in universe if s in M["C"].columns]
    sc = M["score"].loc[t, uni].dropna().sort_values(ascending=False)
    return sc, pd.Series(np.arange(1, len(sc) + 1), index=sc.index)


def evaluate_candidates(M, t, sc, held, open_slots, P) -> list[dict]:
    decisions, filled = [], 0
    for r, s in enumerate(sc.index[:P["threshold"]], start=1):
        if s in held:
            continue
        c, e, em = M["Cf"].loc[t, s], M["ext"].loc[t, s], M["ema"].loc[t, s]
        crv, hi = M["cr"].loc[t, s], M["H"].loc[t, s]
        row = {"Symbol": s, "Rank": r, "Score": sc[s], "Close": c, "EMA": em,
               "Extension": e, "CR": crv, "Month high": hi}
        if filled >= open_slots:
            row["Decision"] = "not needed (slots full)"
        elif P["f1"] and not c > em:
            row["Decision"] = "SKIP F1: close <= EMA"
        elif P["f2"] and e > P["max_ext"]:
            row["Decision"] = f"SKIP F2: extended {e:.0%} > {P['max_ext']:.0%}"
        elif not P["f3"] or pd.isna(crv) or crv >= P["min_cr"]:
            row["Decision"] = "BUY at close"
            filled += 1
        elif P["low_cr_action"] == "pending":
            row["Decision"] = f"BUY-STOP above {hi:,.2f} (CR {crv:.2f})"
            filled += 1
        else:
            row["Decision"] = f"SKIP F3: CR {crv:.2f} < {P['min_cr']}"
        decisions.append(row)
    return decisions


def exit_reason(M, t, s, rank, hold_zone, P, h: dict | None = None) -> str:
    px, reasons = M["Cf"].loc[t, s], []
    if s not in hold_zone:
        reasons.append(f"rank {rank.get(s)} > {P['threshold']}" if s in rank.index
                       else "not ranked / not in universe")
    if P["exit_ema"] and px < M["ema"].loc[t, s]:
        reasons.append("close < EMA")
    if h is not None and h.get("entry"):
        if P["exit_sl"] and px <= h["entry"] * (1 - P["sl_pct"]):
            reasons.append(f"stop-loss {P['sl_pct']:.0%}")
        if P["exit_ts"] and px <= h["peak"] * (1 - P["ts_pct"]):
            reasons.append(f"trailing stop {P['ts_pct']:.0%}")
    return "; ".join(reasons)


def run_strategy(D, M, universe_at, months, P, cost, trade_last=False):
    """months[0] = first rebalance (portfolio formed at its close)."""
    Cf = M["Cf"]
    cash_r = (1 + P["cash_ret"]) ** (1 / 12) - 1
    cash, reserved = float(P["capital"]), 0.0
    holdings, pending = {}, []
    positions, decisions_log, values, buys_value = [], [], [], []
    day_p = D["Close"].index.to_period("M")

    for i, t in enumerate(months):
        bought = 0.0
        if i > 0:
            cash *= 1 + cash_r
            days = D["Close"].index[day_p == t]
            for p in pending:                         # buy-stops live during month t
                s = p["Symbol"]
                hi = D["High"].loc[days, s]
                hit = hi[hi >= p["Trigger"]]
                if len(hit):
                    day = hit.index[0]
                    op = D["Open"].loc[day, s]
                    gap = pd.notna(op) and op > p["Trigger"]
                    fill = op if gap else p["Trigger"]
                    holdings[s] = {"qty": p["Alloc"] * (1 - cost) / fill, "pos": len(positions),
                                   "entry": fill, "peak": fill}
                    positions.append({**p["info"], "Entry date": day, "Entry price": fill,
                                      "Entry type": "buy-stop (gap open)" if gap else "buy-stop",
                                      "Invested": p["Alloc"]})
                    bought += p["Alloc"]
                else:
                    cash += p["Alloc"]
                    decisions_log.append({"Month": str(t), "Symbol": s, "Rank": p["info"]["Signal rank"],
                                          "Decision": f"EXPIRED: buy-stop {p['Trigger']:,.2f} not hit"})
                reserved -= p["Alloc"]
            pending = []

        px = Cf.loc[t]
        for s, h in holdings.items():
            h["peak"] = max(h["peak"], px[s])

        if i == len(months) - 1 and not trade_last:
            values.append((t, cash + reserved + sum(h["qty"] * px[s] for s, h in holdings.items()),
                           cash + reserved, len(holdings)))
            buys_value.append(bought)
            break

        sc, rank = rank_month(M, t, universe_at(t))
        hold_zone = set(sc.index[:P["threshold"]])
        for s in list(holdings):                      # exits
            why = exit_reason(M, t, s, rank, hold_zone, P, holdings[s])
            if why:
                h = holdings.pop(s)
                gross = h["qty"] * px[s]
                cash += gross * (1 - cost)
                positions[h["pos"]].update({"Exit month": str(t), "Exit date": t.to_timestamp(how="end").normalize(),
                                            "Exit price": px[s], "Exit reason": why})
        open_slots = P["top_n"] - len(holdings)       # entries
        if open_slots > 0:
            per_slot = cash / open_slots
            for d in evaluate_candidates(M, t, sc, set(holdings), open_slots, P):
                decisions_log.append({"Month": str(t), **d})
                s = d["Symbol"]
                info = {"Signal month": str(t), "Symbol": s, "Signal rank": d["Rank"],
                        "Signal ext": d["Extension"], "Signal CR": d["CR"]}
                if d["Decision"] == "BUY at close":
                    cash -= per_slot
                    holdings[s] = {"qty": per_slot * (1 - cost) / px[s], "pos": len(positions),
                                   "entry": px[s], "peak": px[s]}
                    positions.append({**info, "Entry date": t.to_timestamp(how="end").normalize(),
                                      "Entry price": px[s], "Entry type": "close", "Invested": per_slot})
                    bought += per_slot
                elif d["Decision"].startswith("BUY-STOP"):
                    cash -= per_slot
                    reserved += per_slot
                    pending.append({"Symbol": s, "Trigger": d["Month high"], "Alloc": per_slot,
                                    "info": info})
        values.append((t, cash + reserved + sum(h["qty"] * px[s] for s, h in holdings.items()),
                       cash + reserved, len(holdings)))
        buys_value.append(bought)

    last = months[-1]
    for s, h in holdings.items():
        positions[h["pos"]].update({"Exit month": "open", "Exit date": last.to_timestamp(how="end").normalize(),
                                    "Exit price": Cf.loc[last, s], "Exit reason": "still held"})
    pos = pd.DataFrame(positions)
    if len(pos):
        pos["Return"] = pos["Exit price"] / pos["Entry price"] - 1
        pos["P&L"] = pos["Invested"] * pos["Return"]
        pos["Months held"] = ((pd.to_datetime(pos["Exit date"]) - pd.to_datetime(pos["Entry date"])).dt.days / 30.44).round(1)
    vals = pd.DataFrame(values, columns=["Month", "Value", "Cash", "Holdings"]).set_index("Month")
    vals["Bought"] = buys_value
    state = {"holdings": holdings, "pending": pending}
    return vals, pos, pd.DataFrame(decisions_log), state


def ew_universe_returns(M, universe_at, months) -> pd.Series:
    """Monthly-rebalanced equal-weight return of the universe (same constituents as the strategy)."""
    out = {}
    for a, b in zip(months[:-1], months[1:]):
        uni = [s for s in universe_at(a) if s in M["Cf"].columns]
        r = (M["Cf"].loc[b, uni] / M["Cf"].loc[a, uni] - 1).dropna()
        out[b] = r.mean() if len(r) else np.nan
    return pd.Series(out)


# ════════════════════════════════════════════════════════════════════════════
# 4. ANALYTICS
# ════════════════════════════════════════════════════════════════════════════
def perf_stats(v: pd.Series, rf: float) -> dict:
    v = v.dropna()
    r = v.pct_change().dropna()
    if len(r) < 1:
        return {}
    yrs = len(r) / 12
    dd = v / v.cummax() - 1
    under = (dd < 0).astype(int)
    longest = under.groupby((under == 0).cumsum()).sum().max()
    vol = r.std() * np.sqrt(12) if len(r) > 1 else np.nan
    downside = r[r < 0].std() * np.sqrt(12) if (r < 0).sum() > 1 else np.nan
    cagr = (v.iloc[-1] / v.iloc[0]) ** (1 / yrs) - 1 if yrs > 0 else np.nan
    ex = r.mean() * 12 - rf
    return {"Total return": v.iloc[-1] / v.iloc[0] - 1,
            "CAGR": cagr if yrs >= 1 else np.nan,
            "Volatility (ann.)": vol,
            "Sharpe": ex / vol if vol else np.nan,
            "Sortino": ex / downside if downside else np.nan,
            "Max drawdown": dd.min(),
            "Longest DD (months)": int(longest),
            "Calmar": cagr / abs(dd.min()) if dd.min() < 0 and yrs >= 1 else np.nan,
            "Best month": r.max(), "Worst month": r.min(),
            "Positive months": (r > 0).mean()}


def relative_stats(strat: pd.Series, bench: pd.Series) -> dict:
    a = pd.concat([strat.pct_change(), bench.pct_change()], axis=1).dropna()
    if len(a) < 6:
        return {}
    s, b = a.iloc[:, 0], a.iloc[:, 1]
    beta = np.cov(s, b)[0, 1] / b.var()
    up, dn = b > 0, b < 0
    return {"Beta": beta,
            "Alpha (ann.)": (s.mean() - beta * b.mean()) * 12,
            "Correlation": s.corr(b),
            "Up capture": s[up].mean() / b[up].mean() if up.any() else np.nan,
            "Down capture": s[dn].mean() / b[dn].mean() if dn.any() else np.nan,
            "Months beating": (s > b).mean()}


def trade_stats(pos: pd.DataFrame, vals: pd.DataFrame) -> dict:
    if pos is None or not len(pos):
        return {}
    closed = pos[pos["Exit month"] != "open"]
    wins, losses = closed[closed["Return"] > 0], closed[closed["Return"] <= 0]
    gross_loss = -losses["P&L"].sum()
    yrs = max((len(vals) - 1) / 12, 1 / 12)
    return {"Entries": len(pos), "Closed trades": len(closed), "Open positions": len(pos) - len(closed),
            "Win rate": len(wins) / len(closed) if len(closed) else np.nan,
            "Avg win": wins["Return"].mean(), "Avg loss": losses["Return"].mean(),
            "Profit factor": wins["P&L"].sum() / gross_loss if gross_loss > 0 else np.nan,
            "Expectancy / trade": closed["Return"].mean(),
            "Avg months held": closed["Months held"].mean(),
            "Turnover (x / yr)": vals["Bought"].sum() / vals["Value"].mean() / yrs,
            "Avg cash %": (vals["Cash"] / vals["Value"]).iloc[:-1].mean() if len(vals) > 1 else np.nan}


# ════════════════════════════════════════════════════════════════════════════
# 5. CHARTS
# ════════════════════════════════════════════════════════════════════════════
def _layout(fig, h=380, yfmt=None, ytitle=None):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                      legend=dict(orientation="h", y=1.08, x=0), plot_bgcolor="rgba(0,0,0,0)")
    fig.update_yaxes(gridcolor="rgba(128,128,128,.18)", zeroline=False, tickformat=yfmt, title=ytitle)
    fig.update_xaxes(showgrid=False)
    return fig


def chart_growth(curves: pd.DataFrame, log: bool):
    fig = go.Figure()
    for c in curves.columns:
        fig.add_trace(go.Scatter(x=curves.index, y=curves[c], name=c, mode="lines",
                                 line=dict(width=2.5 if c == "Strategy" else 2, color=COL[c],
                                           dash="dot" if c == "Equal-weight universe" else None),
                                 hovertemplate="%{y:,.1f}"))
    _layout(fig, 420, ",.0f", "Growth of ₹100")
    if log:
        fig.update_yaxes(type="log")
    return fig


def chart_drawdown(curves: pd.DataFrame):
    dd = curves / curves.cummax() - 1
    fig = go.Figure()
    for c in dd.columns:
        fig.add_trace(go.Scatter(x=dd.index, y=dd[c], name=c, mode="lines",
                                 line=dict(width=2.5 if c == "Strategy" else 2, color=COL[c],
                                           dash="dot" if c == "Equal-weight universe" else None),
                                 fill="tozeroy" if c == "Strategy" else None,
                                 fillcolor="rgba(42,120,214,.12)", hovertemplate="%{y:.1%}"))
    return _layout(fig, 340, ".0%", "Drawdown from peak")


def chart_monthly_heatmap(ret: pd.Series):
    df = pd.DataFrame({"y": ret.index.year, "m": ret.index.month, "r": ret.values})
    piv = df.pivot(index="y", columns="m", values="r").reindex(columns=range(1, 13))
    yearly = (1 + piv.fillna(0)).prod(axis=1) - 1
    z = piv.values
    lim = np.nanmax(np.abs(z)) if np.isfinite(z).any() else 0.1
    text = [[("" if np.isnan(v) else f"{v:+.1%}") for v in row] for row in z]
    fig = go.Figure(go.Heatmap(z=z, x=MONTHS, y=[str(y) for y in piv.index], text=text,
                               texttemplate="%{text}", colorscale=DIVERGING, zmin=-lim, zmax=lim,
                               xgap=2, ygap=2, showscale=False,
                               hovertemplate="%{y} %{x}: %{z:+.2%}<extra></extra>"))
    fig.update_layout(height=max(220, 34 * len(piv) + 60), margin=dict(l=10, r=10, t=10, b=10),
                      yaxis=dict(autorange="reversed", type="category"))
    return fig, yearly


def chart_yearly(yearly: pd.DataFrame):
    fig = go.Figure()
    for c in yearly.columns:
        fig.add_trace(go.Bar(x=yearly.index.astype(str), y=yearly[c], name=c, marker_color=COL[c],
                             marker_line_width=0, hovertemplate="%{y:+.1%}"))
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08)
    return _layout(fig, 340, ".0%", "Calendar-year return")


def chart_rolling(curves: pd.DataFrame):
    roll = curves.pct_change(12).dropna(how="all")
    fig = go.Figure()
    for c in roll.columns:
        fig.add_trace(go.Scatter(x=roll.index, y=roll[c], name=c, mode="lines",
                                 line=dict(width=2.5 if c == "Strategy" else 2, color=COL[c],
                                           dash="dot" if c == "Equal-weight universe" else None),
                                 hovertemplate="%{y:+.1%}"))
    fig.add_hline(y=0, line_color="rgba(128,128,128,.5)", line_width=1)
    return _layout(fig, 320, ".0%", "Rolling 12-month return")


def chart_bar(x, y, h=280, fmt=None, color=COL["Strategy"], title=None):
    fig = go.Figure(go.Bar(x=list(x), y=list(y), marker_color=color, marker_line_width=0,
                           hovertemplate="%{x}: %{y}<extra></extra>"))
    return _layout(fig, h, fmt, title)


def chart_hist(returns: pd.Series):
    fig = go.Figure(go.Histogram(x=returns, nbinsx=30, marker_color=COL["Strategy"],
                                 marker_line_width=1, marker_line_color="rgba(255,255,255,.6)",
                                 hovertemplate="%{x}: %{y} trades<extra></extra>"))
    fig.add_vline(x=0, line_color="rgba(128,128,128,.7)", line_width=1)
    _layout(fig, 300, None, "Trades")
    fig.update_xaxes(tickformat=".0%", title="Trade return")
    return fig


def chart_scatter(pos: pd.DataFrame, xcol: str, xtitle: str, xfmt: str):
    d = pos.dropna(subset=[xcol, "Return"])
    fig = go.Figure(go.Scatter(x=d[xcol], y=d["Return"], mode="markers", text=d["Symbol"],
                               marker=dict(size=9, color=np.where(d["Return"] > 0, GOOD, BAD),
                                           line=dict(width=1.5, color="rgba(255,255,255,.8)"), opacity=.85),
                               hovertemplate="%{text}<br>" + xtitle + ": %{x:" + xfmt + "}<br>Return: %{y:+.1%}<extra></extra>"))
    fig.add_hline(y=0, line_color="rgba(128,128,128,.5)", line_width=1)
    _layout(fig, 320, ".0%", "Trade return")
    fig.update_xaxes(tickformat=xfmt, title=xtitle)
    fig.update_layout(hovermode="closest")
    return fig


def bucket_table(pos: pd.DataFrame, col: str, bins, labels) -> pd.DataFrame:
    d = pos.dropna(subset=[col, "Return"]).copy()
    if not len(d):
        return pd.DataFrame()
    d["Bucket"] = pd.cut(d[col], bins=bins, labels=labels, include_lowest=True)
    g = d.groupby("Bucket", observed=False)["Return"]
    return pd.DataFrame({"Trades": g.size(), "Win rate": g.apply(lambda x: (x > 0).mean() if len(x) else np.nan),
                         "Avg return": g.mean(), "Median return": g.median()})


# ════════════════════════════════════════════════════════════════════════════
# 6. PAGE — sidebar
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Relative Momentum Lab", layout="wide")
st.title("Relative momentum lab")
st.caption("Rank by past return, hold the leaders, rotate monthly. Configure the rules in the sidebar, "
           "then check present holdings or run a backtest.")

with st.sidebar:
    st.header("Universe")
    index_name = st.selectbox("Index", list(INDEX_FILES), index=2)
    up_const = st.file_uploader("Override constituent list (NSE CSV with a 'Symbol' column)", type="csv")
    up_hist = st.file_uploader("Constituent history (report format: columns = effective dates) — "
                               "removes look-ahead bias in backtests", type="csv")

    st.header("Ranking")
    c1, c2 = st.columns(2)
    lookback = c1.number_input("Look-back (months)", 1, 24, 10)
    skip = c2.number_input("Skip (months)", 0, 6, 2)
    c1, c2 = st.columns(2)
    top_n = c1.number_input("Stocks held (buy zone)", 1, 50, 10)
    threshold = c2.number_input("Hold-zone rank", int(top_n), 200, max(20, int(top_n)))

    st.header("Entry filters")
    ema_span = st.number_input("EMA span (months)", 2, 36, 12)
    f1 = st.toggle(f"F1 · Close above {ema_span}-month EMA", True)
    f2 = st.toggle("F2 · Extension cap", True)
    max_ext = st.slider("Max close / EMA − 1", 0.05, 1.50, 0.25, 0.05, format="%.2f", disabled=not f2)
    f3 = st.toggle("F3 · Closing range of signal month", True)
    min_cr = st.slider("Min closing range (C−L)/(H−L)", 0.0, 1.0, 0.5, 0.05, disabled=not f3)
    low_cr_action = st.radio("If CR below min", ["pending", "skip"], disabled=not f3, horizontal=True,
                             format_func=lambda x: "Buy-stop at month high (next month)" if x == "pending"
                             else "Skip → next candidate")

    st.header("Exit filters")
    st.caption(f"Always: exit when rank > {threshold}.")
    exit_ema = st.toggle(f"Close below {ema_span}-month EMA", True)
    exit_sl = st.toggle("Stop-loss from entry (month-end close)", False)
    sl_pct = st.slider("Stop-loss %", 0.05, 0.50, 0.20, 0.05, disabled=not exit_sl)
    exit_ts = st.toggle("Trailing stop from highest month-end close", False)
    ts_pct = st.slider("Trailing stop %", 0.05, 0.60, 0.25, 0.05, disabled=not exit_ts)

    st.header("Portfolio & costs")
    capital = st.number_input("Starting capital (₹)", 10_000, 1_000_000_000, 1_000_000, 100_000)
    cost = st.number_input("Cost per side (%)", 0.0, 2.0, 0.40, 0.05) / 100
    cash_ret = st.number_input("Return on idle cash (% p.a.)", 0.0, 12.0, 0.0, 0.5) / 100
    rf = st.number_input("Risk-free rate for Sharpe (% p.a.)", 0.0, 12.0, 6.5, 0.25) / 100
    adjusted = st.toggle("Dividend-adjusted prices", False,
                         help="Off = split/bonus-adjusted closes (as in the report). On = Yahoo adjusted "
                              "prices incl. dividends; benchmarks stay price indices, so this flatters the strategy.")
    if st.button("Clear cached data", width="stretch"):
        st.cache_data.clear()

P = dict(lookback=int(lookback), skip=int(skip), top_n=int(top_n), threshold=int(threshold),
         ema_span=int(ema_span), f1=f1, f2=f2, max_ext=max_ext, f3=f3, min_cr=min_cr,
         low_cr_action=low_cr_action, exit_ema=exit_ema, exit_sl=exit_sl, sl_pct=sl_pct,
         exit_ts=exit_ts, ts_pct=ts_pct, cash_ret=cash_ret, capital=float(capital))

# ---- universe ---------------------------------------------------------------
try:
    if up_const is not None:
        const, const_src = _parse_constituents(pd.read_csv(up_const)), "uploaded file"
    else:
        const, const_src = load_constituents(index_name)
except Exception as e:  # noqa: BLE001
    st.error(str(e))
    st.stop()
names = const.set_index("Symbol")["Company Name"].to_dict()
industry = const.set_index("Symbol")["Industry"].to_dict()
hist_lists = None
if up_hist is not None:
    try:
        hist_lists, hist_syms = parse_history(up_hist)
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not read the history file: {e}")
        st.stop()


def universe_at(t):
    if hist_lists is None:
        return const["Symbol"].tolist()
    end = t.to_timestamp(how="end")
    ok = [d for d in hist_lists if d <= end]
    return hist_lists[max(ok)] if ok else hist_lists[min(hist_lists)]


all_syms = sorted(set(const["Symbol"]) | (set(hist_syms) if hist_lists else set()))
lcm = last_completed_month()
month_opts = [str(p) for p in pd.period_range("2010-01", lcm, freq="M")]


def get_data(first_rebalance: pd.Period, last: pd.Period):
    warm = max(36, P["lookback"] + P["skip"] + 2)
    start = (first_rebalance - warm).to_timestamp()
    start = f"{start.year}-01-01"                    # coarse key -> cache reuse across runs
    end = str((last + 1).to_timestamp().date() + timedelta(days=1)) if last < lcm else \
        str(datetime.now(IST).date() + timedelta(days=1))
    with st.spinner(f"Downloading {len(all_syms)} stock histories from {start[:4]} "
                    "(first load can take 1–3 minutes; cached for 6 h)…"):
        D = load_ohlc(tuple(all_syms), start, end, adjusted)
    return D, start, end


# ════════════════════════════════════════════════════════════════════════════
# 7. PAGE — controls
# ════════════════════════════════════════════════════════════════════════════
st.markdown(f"**Universe:** {index_name} · {len(const)} stocks ({const_src})"
            + (" · constituent history uploaded" if hist_lists else
               " · current list used for all past months (look-ahead bias in backtests)"))
cA, cB = st.columns(2)
with cA, st.container(border=True):
    st.subheader("Present holdings")
    p_start = st.selectbox("Portfolio started in", month_opts,
                           index=month_opts.index(str(lcm - 11)) if str(lcm - 11) in month_opts else 0,
                           help="Holdings depend on when the portfolio began; first buys at the close "
                                "of the month before this one.")
    run_present = st.button("Show present holdings", type="primary", width="stretch")
with cB, st.container(border=True):
    st.subheader("Backtest")
    b1, b2 = st.columns(2)
    default_start = "2026-01" if "2026-01" in month_opts else month_opts[-12]
    bt_start = b1.selectbox("First month held", month_opts, index=month_opts.index(default_start))
    bt_end = b2.selectbox("Last month", month_opts, index=len(month_opts) - 1)
    run_bt = st.button("Run backtest", type="primary", width="stretch")


def _months(first_held: str, last: str):
    first_reb = pd.Period(first_held, "M") - 1
    return list(pd.period_range(first_reb, pd.Period(last, "M"), freq="M"))


if run_present:
    months = _months(p_start, str(lcm))
    try:
        D, _, _ = get_data(months[0], lcm)
        M = build_monthly(D, P)
        vals, pos, dec, state = run_strategy(D, M, universe_at, months, P, cost, trade_last=True)
        st.session_state["present"] = dict(P=P.copy(), months=months, M=M, vals=vals, pos=pos,
                                           dec=dec, state=state, index=index_name)
    except Exception as e:  # noqa: BLE001
        st.error(f"Present-holdings run failed: {e}")
if run_bt:
    months = _months(bt_start, bt_end)
    if len(months) < 3:
        st.error("Pick at least two months between start and end.")
    else:
        try:
            D, start, end = get_data(months[0], months[-1])
            M = build_monthly(D, P)
            g_vals, pos, dec, _ = run_strategy(D, M, universe_at, months, P, 0.0)
            n_vals, n_pos, _, _ = run_strategy(D, M, universe_at, months, P, cost)
            bench, bsrc = load_benchmarks(start, end)
            ew = ew_universe_returns(M, universe_at, months)
            st.session_state["bt"] = dict(P=P.copy(), months=months, M=M, g=g_vals, n=n_vals, pos=pos,
                                          n_pos=n_pos, dec=dec, bench=bench, bsrc=bsrc, ew=ew,
                                          index=index_name, cost=cost, hist=hist_lists is not None)
        except Exception as e:  # noqa: BLE001
            st.error(f"Backtest failed: {e}")

tab_p, tab_b = st.tabs(["Present holdings", "Backtest"])


def _pct(df, cols):
    return {c: "{:+.1%}" for c in cols if c in df.columns}


# ════════════════════════════════════════════════════════════════════════════
# 8. PRESENT HOLDINGS VIEW
# ════════════════════════════════════════════════════════════════════════════
with tab_p:
    R = st.session_state.get("present")
    if R is None:
        st.info("Choose when the portfolio started and press **Show present holdings**.")
    else:
        if R["P"] != P or R["index"] != index_name:
            st.warning("Settings changed since this run — press the button again to refresh.")
        t, M, vals, pos, state = R["months"][-1], R["M"], R["vals"], R["pos"], R["state"]
        v = vals["Value"]
        k = st.columns(5)
        k[0].metric("Signal month", t.strftime("%b %Y"))
        k[1].metric("Portfolio value", f"₹{v.iloc[-1]:,.0f}", f"{v.iloc[-1] / R['P']['capital'] - 1:+.1%} since start")
        k[2].metric("Stocks held", f"{len(state['holdings'])} / {R['P']['top_n']}")
        k[3].metric("Buy-stops pending", len(state["pending"]))
        k[4].metric("Cash", f"{vals['Cash'].iloc[-1] / v.iloc[-1]:.0%}")
        st.caption(f"Positions reflect trades at the {t.strftime('%b %Y')} close. In live trading the signal "
                   "is known only after that close: execute in NSE's post-close session or at the next open.")

        sc, rank = rank_month(M, t, universe_at(t))
        cur = pos[pos["Exit month"] == "open"].copy() if len(pos) else pd.DataFrame()
        st.subheader("Current holdings")
        if len(cur):
            cur["Company"] = cur["Symbol"].map(names)
            cur["Industry"] = cur["Symbol"].map(industry)
            cur["Rank now"] = cur["Symbol"].map(rank)
            cur["Extension now"] = cur["Symbol"].map(M["ext"].loc[t])
            cur["Weight"] = [state["holdings"][s]["qty"] * M["Cf"].loc[t, s] / v.iloc[-1] for s in cur["Symbol"]]
            cur["Entry date"] = pd.to_datetime(cur["Entry date"]).dt.date
            show = cur[["Symbol", "Company", "Industry", "Entry date", "Entry type", "Entry price",
                        "Exit price", "Return", "Weight", "Rank now", "Extension now"]].rename(
                columns={"Exit price": "Last close"})
            st.dataframe(show.style.format({"Entry price": "{:,.2f}", "Last close": "{:,.2f}", "Return": "{:+.1%}",
                                            "Weight": "{:.1%}", "Extension now": "{:+.1%}", "Rank now": "{:.0f}"}),
                         hide_index=True, width="stretch")
        else:
            st.write("No open positions.")

        dec = R["dec"]
        this = dec[dec["Month"] == str(t)] if len(dec) else pd.DataFrame()
        exits = pos[pos.get("Exit month", pd.Series(dtype=str)) == str(t)] if len(pos) else pd.DataFrame()
        c1, c2 = st.columns(2)
        with c1:
            st.subheader(f"Exits at {t.strftime('%b %Y')} close")
            if len(exits):
                st.dataframe(exits[["Symbol", "Exit reason", "Entry price", "Exit price", "Return"]].style.format(
                    {"Entry price": "{:,.2f}", "Exit price": "{:,.2f}", "Return": "{:+.1%}"}),
                    hide_index=True, width="stretch")
            else:
                st.write("None.")
        with c2:
            st.subheader(f"Buy-stops for {(t + 1).strftime('%b %Y')}")
            if state["pending"]:
                pdf = pd.DataFrame([{"Symbol": p["Symbol"], "Company": names.get(p["Symbol"], ""),
                                     "Trigger": p["Trigger"],
                                     "Close": M["Cf"].loc[t, p["Symbol"]],
                                     "vs close": p["Trigger"] / M["Cf"].loc[t, p["Symbol"]] - 1,
                                     "₹ alloc": p["Alloc"]} for p in state["pending"]])
                st.dataframe(pdf.style.format({"Trigger": "{:,.2f}", "Close": "{:,.2f}",
                                               "vs close": "{:+.1%}", "₹ alloc": "{:,.0f}"}),
                             hide_index=True, width="stretch")
                st.caption(f"Trigger = high of the signal month. Place as GTT / stop orders; cancel any "
                           f"not hit by the end of {(t + 1).strftime('%b %Y')}.")
            else:
                st.write("None.")

        st.subheader(f"Top {R['P']['threshold']} ranks — {t.strftime('%b %Y')}")
        top = pd.DataFrame({"Rank": rank, "Score": sc}).head(R["P"]["threshold"])
        top.insert(0, "Company", top.index.map(names))
        top["Industry"] = top.index.map(industry)
        top["Extension"] = M["ext"].loc[t, top.index]
        top["CR"] = M["cr"].loc[t, top.index]
        top["F1"] = M["Cf"].loc[t, top.index] > M["ema"].loc[t, top.index]
        top["F2"] = top["Extension"] <= R["P"]["max_ext"]
        top["F3"] = top["CR"] >= R["P"]["min_cr"]
        top["Held"] = top.index.isin(list(state["holdings"]))
        dmap = this.set_index("Symbol")["Decision"].to_dict() if len(this) else {}
        top["Decision this month"] = top.index.map(lambda s: dmap.get(s, "held" if s in state["holdings"] else ""))
        st.dataframe(top.style.format({"Score": "{:+.1%}", "Extension": "{:+.1%}", "CR": "{:.2f}"}),
                     width="stretch")

        with st.expander("Check my own holdings against the exit rules"):
            txt = st.text_area("One per line: SYMBOL or SYMBOL,entry_price", "", height=120)
            if txt.strip():
                hz = set(sc.index[:R["P"]["threshold"]])
                rows = []
                for line in txt.strip().splitlines():
                    parts = [x.strip() for x in line.split(",")]
                    s = _clean(parts[0])
                    if s not in M["Cf"].columns:
                        rows.append({"Symbol": s, "Action": "?", "Reason": "no price data (not downloaded)"})
                        continue
                    ep = float(parts[1]) if len(parts) > 1 and parts[1] else None
                    h = {"entry": ep, "peak": max(ep or 0, M["Cf"].loc[t, s])} if ep else None
                    why = exit_reason(M, t, s, rank, hz, R["P"], h)
                    rows.append({"Symbol": s, "Rank": rank.get(s), "Close": M["Cf"].loc[t, s],
                                 "Action": "SELL" if why else "KEEP", "Reason": why or "passes all exit rules"})
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
                st.caption("Trailing stop uses only the entry price and this month's close here.")


# ════════════════════════════════════════════════════════════════════════════
# 9. BACKTEST VIEW
# ════════════════════════════════════════════════════════════════════════════
with tab_b:
    B = st.session_state.get("bt")
    if B is None:
        st.info("Pick the first and last month and press **Run backtest**.")
    else:
        if B["P"] != P or B["index"] != index_name:
            st.warning("Settings changed since this run — press **Run backtest** again to refresh.")
        if not B["hist"]:
            st.warning("Using today's constituent list for every past month. Stocks that joined the index "
                       "because they rallied are available from day one, which flatters any momentum backtest. "
                       "Compare with the equal-weight universe line, which shares the same bias.")
        months, g, n = B["months"], B["g"], B["n"]
        view = st.radio("Returns shown", ["Net of costs", "Gross"], horizontal=True)
        V = n if view.startswith("Net") else g
        posv = B["n_pos"] if view.startswith("Net") else B["pos"]

        idx = [m.to_timestamp(how="end").normalize() for m in V.index]
        curves = pd.DataFrame({"Strategy": V["Value"].values}, index=idx)
        bench = B["bench"]
        for c in ("Nifty 50", "Nifty 500"):
            if c in bench.columns:
                s = bench[c].reindex(V.index)
                curves[c] = s.values
        ew = B["ew"].reindex(V.index).fillna(0)
        curves["Equal-weight universe"] = (1 + ew).cumprod().values
        curves = curves.dropna(axis=1, how="any")
        curves = curves / curves.iloc[0] * 100

        stats = {c: perf_stats(curves[c], rf) for c in curves.columns}
        S = stats["Strategy"]
        k = st.columns(6)
        k[0].metric("Total return", f"{S['Total return']:+.1%}",
                    f"{S['Total return'] - stats.get('Nifty 50', {}).get('Total return', np.nan):+.1%} vs Nifty 50"
                    if "Nifty 50" in stats else None)
        k[1].metric("CAGR", "—" if pd.isna(S["CAGR"]) else f"{S['CAGR']:.1%}")
        k[2].metric("Max drawdown", f"{S['Max drawdown']:.1%}")
        k[3].metric("Sharpe", "—" if pd.isna(S["Sharpe"]) else f"{S['Sharpe']:.2f}")
        k[4].metric("Avg cash", f"{(V['Cash'] / V['Value']).iloc[:-1].mean():.0%}")
        k[5].metric("Entries", len(posv))
        st.caption(f"Held {(months[0] + 1).strftime('%b %Y')} – {months[-1].strftime('%b %Y')} · "
                   f"portfolio formed at the {months[0].strftime('%b %Y')} close · costs {B['cost']:.2%} per side · "
                   "benchmarks: " + ", ".join(f"{k_}: {v_}" for k_, v_ in B["bsrc"].items()))

        st.subheader("Growth of ₹100")
        logscale = st.toggle("Log scale", value=len(months) > 36)
        st.plotly_chart(chart_growth(curves, logscale), width="stretch")
        st.subheader("Drawdown")
        st.plotly_chart(chart_drawdown(curves), width="stretch")

        st.subheader("Performance vs benchmarks")
        ROWFMT = {"Total return": "{:+.1%}", "CAGR": "{:+.1%}", "Volatility (ann.)": "{:.1%}",
                  "Sharpe": "{:.2f}", "Sortino": "{:.2f}", "Max drawdown": "{:.1%}",
                  "Longest DD (months)": "{:.0f}", "Calmar": "{:.2f}", "Best month": "{:+.1%}",
                  "Worst month": "{:+.1%}", "Positive months": "{:.0%}", "Beta": "{:.2f}",
                  "Alpha (ann.)": "{:+.1%}", "Correlation": "{:.2f}", "Up capture": "{:.2f}",
                  "Down capture": "{:.2f}", "Months beating": "{:.0%}"}

        def _fmt_table(df):
            out = df.copy().astype(object)
            for r in df.index:
                out.loc[r] = [("—" if pd.isna(x) else ROWFMT.get(r, "{}").format(x)) for x in df.loc[r]]
            return out

        tbl = pd.DataFrame(stats)
        rel = pd.DataFrame({c: relative_stats(curves["Strategy"], curves[c]) for c in curves.columns[1:]})
        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown("**Absolute**")
            st.dataframe(_fmt_table(tbl), width="stretch", height=36 * (len(tbl) + 1) + 4)
        with c2:
            st.markdown("**Strategy vs each benchmark**")
            if len(rel):
                st.dataframe(_fmt_table(rel), width="stretch", height=36 * (len(rel) + 1) + 4)
                st.caption("Up/down capture: strategy's average return in benchmark up/down months as a "
                           "fraction of the benchmark's. Down capture below 1 = loses less in bad months.")
            else:
                st.write("Needs at least 6 months.")
        tbl = pd.concat([tbl, rel.add_prefix("vs ")], axis=1)

        mret = curves["Strategy"].pct_change().dropna()
        st.subheader("Monthly returns (strategy)")
        hm, yearly_s = chart_monthly_heatmap(mret)
        st.plotly_chart(hm, width="stretch")

        st.subheader("Calendar-year returns")
        yr = pd.DataFrame({c: (1 + curves[c].pct_change().dropna()).groupby(lambda d: d.year).prod() - 1
                           for c in curves.columns if c != "Equal-weight universe"})
        st.plotly_chart(chart_yearly(yr), width="stretch")
        if len(curves) > 24:
            st.subheader("Rolling 12-month return")
            st.plotly_chart(chart_rolling(curves), width="stretch")

        st.subheader("Exposure")
        c1, c2 = st.columns(2)
        ex_idx = [m.strftime("%b %y") for m in V.index]
        c1.plotly_chart(chart_bar(ex_idx, V["Holdings"], 260, None, title="Stocks held"), width="stretch")
        c2.plotly_chart(chart_bar(ex_idx, V["Cash"] / V["Value"], 260, ".0%", COL["Equal-weight universe"],
                                  "Cash share of portfolio"), width="stretch")

        st.subheader("Trade analytics")
        ts = trade_stats(posv, V)
        if ts:
            k = st.columns(6)
            k[0].metric("Closed trades", ts["Closed trades"])
            k[1].metric("Win rate", "—" if pd.isna(ts["Win rate"]) else f"{ts['Win rate']:.0%}")
            k[2].metric("Avg win / loss", f"{ts['Avg win']:+.0%} / {ts['Avg loss']:+.0%}"
                        if pd.notna(ts["Avg win"]) and pd.notna(ts["Avg loss"]) else "—")
            k[3].metric("Profit factor", "—" if pd.isna(ts["Profit factor"]) else f"{ts['Profit factor']:.2f}")
            k[4].metric("Avg months held", "—" if pd.isna(ts["Avg months held"]) else f"{ts['Avg months held']:.1f}")
            k[5].metric("Turnover", f"{ts['Turnover (x / yr)']:.1f}× / yr")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Trade return distribution**")
                st.plotly_chart(chart_hist(posv["Return"]), width="stretch")
            with c2:
                st.markdown("**Exit reasons**")
                er = (posv["Exit reason"].fillna("").str.split("; ").explode()
                      .str.replace(r"^rank \d+ > (\d+)$", r"rank > \1", regex=True).value_counts())
                st.plotly_chart(chart_bar(er.index, er.values, 300, None, title="Count"), width="stretch")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Entry types**")
                et = posv["Entry type"].value_counts()
                st.plotly_chart(chart_bar(et.index, et.values, 260, None, title="Count"), width="stretch")
            with c2:
                st.markdown("**Industries entered**")
                ind = posv["Symbol"].map(industry).replace("", np.nan).fillna("Unknown").value_counts().head(12)
                st.plotly_chart(chart_bar(ind.index, ind.values, 260, None, title="Entries"), width="stretch")

            st.subheader("Signal diagnostics")
            st.caption("Does extension above the EMA or a weak closing range at the signal month predict worse trades? "
                       "Most informative with F2/F3 switched OFF, so the full range of entries is present.")
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(chart_scatter(posv, "Signal ext", "Extension above EMA at signal", ".0%"), width="stretch")
                bt_ = bucket_table(posv, "Signal ext", [-10, 0, 0.10, 0.25, 0.50, 1.0, 100],
                                   ["<0%", "0–10%", "10–25%", "25–50%", "50–100%", ">100%"])
                st.dataframe(bt_.style.format({"Win rate": "{:.0%}", "Avg return": "{:+.1%}", "Median return": "{:+.1%}"}, na_rep="—"),
                             width="stretch")
            with c2:
                st.plotly_chart(chart_scatter(posv, "Signal CR", "Closing range at signal", ".2f"), width="stretch")
                bt_ = bucket_table(posv, "Signal CR", [0, 0.3, 0.5, 0.7, 1.0001], ["0–0.3", "0.3–0.5", "0.5–0.7", "0.7–1.0"])
                st.dataframe(bt_.style.format({"Win rate": "{:.0%}", "Avg return": "{:+.1%}", "Median return": "{:+.1%}"}, na_rep="—"),
                             width="stretch")
            dec = B["dec"]
            if len(dec):
                kind = dec["Decision"].astype(str).str.extract(r"^(SKIP F\d|EXPIRED|BUY-STOP|BUY at close)")[0]
                st.markdown("**Candidate decisions** " + " · ".join(f"{a}: {b}" for a, b in kind.value_counts().items()))

            c1, c2 = st.columns(2)
            cols = ["Symbol", "Signal month", "Entry type", "Entry price", "Exit price", "Return", "Months held", "Exit reason"]
            fm = {"Entry price": "{:,.2f}", "Exit price": "{:,.2f}", "Return": "{:+.1%}", "Months held": "{:.1f}"}
            c1.markdown("**Best trades**")
            c1.dataframe(posv.nlargest(10, "Return")[cols].style.format(fm), hide_index=True, width="stretch")
            c2.markdown("**Worst trades**")
            c2.dataframe(posv.nsmallest(10, "Return")[cols].style.format(fm), hide_index=True, width="stretch")
            with st.expander("Full trade log"):
                log = posv.copy()
                log["Company"] = log["Symbol"].map(names)
                st.dataframe(log, hide_index=True, width="stretch")

        xbuf = io.BytesIO()
        with pd.ExcelWriter(xbuf, engine="openpyxl") as xw:
            tbl.to_excel(xw, sheet_name="Performance")
            (curves.rename_axis("Month end")).to_excel(xw, sheet_name="Growth of 100")
            V.rename(index=str).to_excel(xw, sheet_name="Monthly values")
            posv.to_excel(xw, sheet_name="Trades", index=False)
            B["dec"].to_excel(xw, sheet_name="Candidate decisions", index=False)
            pd.DataFrame(list(B["P"].items()), columns=["Setting", "Value"]).astype(str).to_excel(
                xw, sheet_name="Settings", index=False)
        st.download_button("Download backtest (Excel)", xbuf.getvalue(),
                           f"momentum_backtest_{months[0] + 1}_{months[-1]}.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.caption(f"Data: constituents {const_src}; stock prices Yahoo Finance "
           f"({'dividend-adjusted' if adjusted else 'split-adjusted'}). Rules output, not investment advice. "
           f"Rendered {datetime.now(IST):%d %b %Y %H:%M} IST.")
