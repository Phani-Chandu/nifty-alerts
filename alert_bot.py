#!/usr/bin/env python3
"""
NIFTY 5m alert bot - single file, runs on free GitHub Actions, alerts to Telegram.
​
This is the phone-only version of the strategy in STRATEGY.md. It is deliberately
ONE FILE with no local imports, because you will be creating it through the GitHub
web editor on a phone and uploading twelve files that way is miserable.
​
It does NOT place orders. It watches NIFTY, and when the setup fires it sends you a
card with the strike, premium, lot count, stop and targets. You place the trade
yourself in your broker app.
​
NO SECOND DEMAT ACCOUNT NEEDED
------------------------------
Earlier advice here said Kotak Neo's API needs a static IP and therefore can't run
on GitHub Actions. That was WRONG. Kotak Neo's own documentation is explicit:
​
    "IP validation is enforced only on order APIs."
    With IP validation:    Place Order, Modify Order, Cancel Order
    WITHOUT IP validation: Login APIs, Report APIs, Portfolio APIs,
                           Data APIs, Websocket streams
​
Since this bot only logs in and reads data - it never places an order - it runs
fine from GitHub's dynamic IPs on your existing Kotak Neo account. Same is true
of ICICI Direct Breeze.
​
CHOOSING A BROKER  (set BROKER=kotak or BROKER=breeze)
------------------------------------------------------
  KOTAK   Login is TOTP-based, so it can authenticate itself every morning with
          no human involved. That is what makes unattended running possible.
          Catch: Kotak Neo has no historical-candle API ("Historical data is
          unavailable at the moment" per their own support page), so the bot
          builds its own 5m bars from live quotes and keeps them in bars.csv in
          your repo. See the BAR HISTORY section below.
                -> Best choice for the live alert bot.
​
  BREEZE  ICICI Direct. Has proper historical data, so no warmup problem.
          Catch: the session token expires every midnight and can only be
          refreshed by logging in through a browser. Fine for a backtest you run
          by hand; painful for a bot that must start itself at 09:20 daily.
                -> Best choice for BACKTESTING (see download_breeze_data.py).
                   Usable for alerts only if you refresh the token each morning.
​
Environment (set as GitHub repository Secrets):
    BROKER                 "kotak" (default) or "breeze"
    -- if kotak --
    KOTAK_CONSUMER_KEY, KOTAK_CONSUMER_SECRET,
    KOTAK_MOBILE (with +91), KOTAK_UCC, KOTAK_MPIN, KOTAK_TOTP_SECRET
    -- if breeze --
    BREEZE_API_KEY, BREEZE_API_SECRET, BREEZE_SESSION_TOKEN
    -- notifications, at least one --
    TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID,  and/or  NTFY_TOPIC
    CAPITAL                optional, default 300000
"""
​
from __future__ import annotations
​
import math
import os
import sys
import time
import traceback
from datetime import datetime, time as dtime, timedelta
​
import numpy as np
import pandas as pd
import requests
​
# Broker SDKs are imported lazily so this file can be imported for testing
# (see mobile/verify_bot_matches_engine.py) on a machine with no broker libraries.
try:
    import pyotp
except ImportError:  # pragma: no cover
    pyotp = None
​
IST = "Asia/Kolkata"
BROKER = os.environ.get("BROKER", "kotak").strip().lower()
​
# ---------------------------------------------------------------------------
# CONFIG - keep these in sync with nifty5m/strategy.py
# ---------------------------------------------------------------------------
EMA_FAST, EMA_SLOW = 20, 50
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
RSI_LEN, ATR_LEN, SLOPE_LOOKBACK = 14, 14, 10
​
RSI_LONG_MIN, RSI_LONG_MAX = 55.0, 72.0
RSI_SHORT_MIN, RSI_SHORT_MAX = 28.0, 45.0
MIN_EMA_GAP_ATR = 0.30
MIN_ATR_RATIO = 0.95
REQUIRE_MACD_ZERO_SIDE = True
WATCH_HIST_BARS, WATCH_PROXIMITY_ATR = 2, 0.06
​
STOP_PCT, TARGET1_PCT = 0.30, 0.40
TARGET_PREMIUM, MIN_PREMIUM, MAX_PREMIUM = 75.0, 50.0, 100.0
MIN_DTE_DAYS = 1
​
SESSION_START = dtime(9, 30)
LAST_ENTRY = dtime(14, 45)
HARD_EXIT = dtime(15, 10)
MAX_TRADES_DAY, MAX_CONSEC_LOSSES = 3, 2
​
RISK_PER_TRADE = 0.01
CAPITAL = float(os.environ.get("CAPITAL", 300_000))
​
LOT_SIZE = 65  # NIFTY, effective from the 06-Jan-2026 weekly expiry
STRIKE_STEP = 50
​
# NSE trading holidays. UPDATE THIS EVERY YEAR - a wrong holiday list shifts the
# expiry date, which makes every premium the bot quotes you wrong for that week.
NSE_HOLIDAYS_2026 = {
    "2026-01-26", "2026-03-04", "2026-03-20", "2026-04-01", "2026-04-03",
    "2026-04-14", "2026-05-01", "2026-08-15", "2026-09-14", "2026-10-02",
    "2026-10-21", "2026-11-09", "2026-12-25",
}
HOLIDAYS = {pd.Timestamp(d) for d in NSE_HOLIDAYS_2026}
​
​
def log(msg: str):
    print(f"{datetime.now().strftime('%H:%M:%S')} {msg}", flush=True)
​
​
# ---------------------------------------------------------------------------
# INDICATORS - TradingView-compatible (SMA-seeded EMA, Wilder RSI/ATR)
# ---------------------------------------------------------------------------
def ema(s: pd.Series, length: int) -> pd.Series:
    v = s.astype(float).to_numpy()
    out = np.full(len(v), np.nan)
    if len(v) < length:
        return pd.Series(out, index=s.index)
    a = 2.0 / (length + 1.0)
    prev = np.nanmean(v[:length])
    out[length - 1] = prev
    for i in range(length, len(v)):
        prev = a * v[i] + (1 - a) * prev
        out[i] = prev
    return pd.Series(out, index=s.index)
​
​
def rma(s: pd.Series, length: int) -> pd.Series:
    v = s.astype(float).to_numpy()
    out = np.full(len(v), np.nan)
    if len(v) < length:
        return pd.Series(out, index=s.index)
    a = 1.0 / length
    prev = np.nanmean(v[:length])
    out[length - 1] = prev
    for i in range(length, len(v)):
        prev = a * v[i] + (1 - a) * prev
        out[i] = prev
    return pd.Series(out, index=s.index)
​
​
def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    # dropna() before seeding: close.diff() leaves a NaN at index 0, and averaging
    # it in as a zero drags the seed down for dozens of bars. Wilder smoothing has
    # a long memory, so this is not a rounding detail.
    d = close.astype(float).diff().dropna()
    ag, al = rma(d.clip(lower=0), length), rma((-d).clip(lower=0), length)
    rs = ag / al.replace(0.0, np.nan)
    vals = 100.0 - 100.0 / (1.0 + rs)
    vals = vals.where(al != 0.0, 100.0).where(ag != 0.0, 0.0)
    vals = vals.where(ag.notna() & al.notna(), np.nan)
    out = pd.Series(np.nan, index=close.index, dtype=float)
    out.loc[vals.index] = vals.to_numpy()
    return out
​
​
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["ema_fast"] = ema(d["close"], EMA_FAST)
    d["ema_slow"] = ema(d["close"], EMA_SLOW)
    macd_line = ema(d["close"], MACD_FAST) - ema(d["close"], MACD_SLOW)
    valid = macd_line.dropna()
    sig = ema(valid, MACD_SIGNAL)
    signal_line = pd.Series(np.nan, index=d.index)
    signal_line.loc[sig.index] = sig.to_numpy()
    d["macd"], d["macd_signal"] = macd_line, signal_line
    d["macd_hist"] = macd_line - signal_line
    d["rsi"] = rsi(d["close"], RSI_LEN)
    pc = d["close"].shift(1)
    tr = pd.concat(
        [d["high"] - d["low"], (d["high"] - pc).abs(), (d["low"] - pc).abs()], axis=1
    ).max(axis=1)
    d["atr"] = rma(tr, ATR_LEN)
    d["ema_slow_slope"] = (d["ema_slow"] - d["ema_slow"].shift(SLOPE_LOOKBACK)) / SLOPE_LOOKBACK
    d["ema_gap_atr"] = (d["ema_fast"] - d["ema_slow"]) / d["atr"].replace(0.0, np.nan)
    d["atr_ratio"] = d["atr"] / d["atr"].rolling(20).mean()
    return d
​
​
# ---------------------------------------------------------------------------
# SIGNAL - identical logic to nifty5m/strategy.evaluate_bar
# ---------------------------------------------------------------------------
def evaluate(d: pd.DataFrame, i: int):
    """Returns (state, detail). States: flat / watch_ce / watch_pe / enter_ce / enter_pe."""
    if i < max(1, WATCH_HIST_BARS):
        return "flat", {}
    r, p = d.iloc[i], d.iloc[i - 1]
    need = ["ema_fast", "ema_slow", "macd", "macd_signal", "macd_hist", "rsi", "atr",
            "ema_slow_slope", "ema_gap_atr", "atr_ratio"]
    if any(pd.isna(r.get(c)) for c in need):
        return "flat", {"reason": "warming up"}
​
    up = r["ema_fast"] > r["ema_slow"] and r["ema_slow_slope"] > 0 and r["close"] > r["ema_fast"]
    dn = r["ema_fast"] < r["ema_slow"] and r["ema_slow_slope"] < 0 and r["close"] < r["ema_fast"]
    regime = abs(r["ema_gap_atr"]) >= MIN_EMA_GAP_ATR and r["atr_ratio"] >= MIN_ATR_RATIO
​
    rsi_l = RSI_LONG_MIN <= r["rsi"] <= RSI_LONG_MAX
    rsi_s = RSI_SHORT_MIN <= r["rsi"] <= RSI_SHORT_MAX
    x_up = p["macd_hist"] <= 0 < r["macd_hist"]
    x_dn = p["macd_hist"] >= 0 > r["macd_hist"]
​
    detail = {
        "time": d.index[i], "close": float(r["close"]),
        "ema_fast": float(r["ema_fast"]), "ema_slow": float(r["ema_slow"]),
        "rsi": float(r["rsi"]), "macd": float(r["macd"]),
        "macd_signal": float(r["macd_signal"]), "macd_hist": float(r["macd_hist"]),
        "atr": float(r["atr"]), "ema_gap_atr": float(r["ema_gap_atr"]),
        "atr_ratio": float(r["atr_ratio"]),
    }
​
    if up and regime and rsi_l and x_up and (not REQUIRE_MACD_ZERO_SIDE or r["macd"] > 0):
        return "enter_ce", detail
    if dn and regime and rsi_s and x_dn and (not REQUIRE_MACD_ZERO_SIDE or r["macd"] < 0):
        return "enter_pe", detail
​
    h_now, h_then = r["macd_hist"], d.iloc[i - WATCH_HIST_BARS]["macd_hist"]
    near = abs(h_now) < WATCH_PROXIMITY_ATR * r["atr"]
    if up and regime and rsi_l and h_now <= 0 and h_now > h_then and near:
        return "watch_ce", detail
    if dn and regime and rsi_s and h_now >= 0 and h_now < h_then and near:
        return "watch_pe", detail
    return "flat", detail
​
​
# ---------------------------------------------------------------------------
# EXPIRY + OPTION PRICING (fallback only; live quotes preferred)
# ---------------------------------------------------------------------------
def next_weekly_expiry(now: pd.Timestamp, min_dte: int = MIN_DTE_DAYS) -> pd.Timestamp:
    """NIFTY weeklies expire TUESDAY (since 01-Sep-2025). min_dte=1 avoids 0-DTE."""
    day = pd.Timestamp(now).normalize()
    if day.tzinfo is not None:
        day = day.tz_localize(None)
    for ahead in range(0, 21):
        c = day + timedelta(days=ahead)
        if c.dayofweek != 1:  # Tuesday
            continue
        while c.normalize() in HOLIDAYS or c.dayofweek >= 5:
            c -= timedelta(days=1)
        if (c.normalize() - day).days >= min_dte:
            return c.normalize()
    raise RuntimeError("no expiry found")
​
​
def _ncdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
​
​
def bs_price(spot, strike, t, iv, typ, r=0.065):
    if t <= 0 or iv <= 0:
        return max(0.0, spot - strike) if typ == "CE" else max(0.0, strike - spot)
    st = math.sqrt(t)
    d1 = (math.log(spot / strike) + (r + 0.5 * iv * iv) * t) / (iv * st)
    d2 = d1 - iv * st
    disc = math.exp(-r * t)
    if typ == "CE":
        return spot * _ncdf(d1) - strike * disc * _ncdf(d2)
    return strike * disc * _ncdf(-d2) - spot * _ncdf(-d1)
​
​
def t_years(now: pd.Timestamp, expiry: pd.Timestamp) -> float:
    now = pd.Timestamp(now)
    e = pd.Timestamp(expiry).normalize()
    if e.tzinfo is not None:
        e = e.tz_localize(None)
    e = e + timedelta(hours=15, minutes=30)
    if now.tzinfo is not None:
        e = e.tz_localize(now.tz)
    return max((e - now).total_seconds(), 60.0) / (365.0 * 24 * 3600)
​
​
# ---------------------------------------------------------------------------
# NOTIFICATIONS - Telegram and/or ntfy.sh
# ---------------------------------------------------------------------------
# Configure either or both. If both are set, every alert goes to both, which is
# worth doing: Indian ISPs blocked Telegram at network level in June 2026 and the
# regulatory position is still unsettled. A second channel costs nothing.
#
# ntfy.sh needs no account, no token and no bot. You invent a topic name, install
# the ntfy app, and subscribe to it. The topic name IS the secret - anyone who
# guesses it can read your alerts - so make it long and random, e.g.
#   nifty-a7f3k92m-bgx
TG_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TG_CHAT = os.environ.get("TELEGRAM_CHAT_ID")
NTFY_TOPIC = os.environ.get("NTFY_TOPIC")
NTFY_SERVER = os.environ.get("NTFY_SERVER", "https://ntfy.sh").rstrip("/")
​
_TAG_RE = None
​
​
def strip_html(s: str) -> str:
    """Telegram takes HTML; ntfy takes plain text. Convert between them."""
    global _TAG_RE
    if _TAG_RE is None:
        import re
        _TAG_RE = re.compile(r"<[^>]+>")
    out = _TAG_RE.sub("", s)
    for a, b in [("&lt;", "<"), ("&gt;", ">"), ("&amp;", "&"), ("&quot;", '"')]:
        out = out.replace(a, b)
    return out
​
​
def _send_telegram(text: str, silent: bool) -> bool:
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            json={"chat_id": TG_CHAT, "text": text, "parse_mode": "HTML",
                  "disable_notification": silent},
            timeout=15,
        )
        if not r.ok:
            log(f"telegram error {r.status_code}: {r.text[:150]}")
        return r.ok
    except Exception as e:
        log(f"telegram exception: {e}")
        return False
​
​
def _send_ntfy(text: str, silent: bool) -> bool:
    plain = strip_html(text)
    lines = [ln for ln in plain.split("\n") if ln.strip()]
    title = lines[0][:100] if lines else "NIFTY alert"
    body = "\n".join(lines[1:]) if len(lines) > 1 else plain
    try:
        r = requests.post(
            f"{NTFY_SERVER}/{NTFY_TOPIC}",
            data=body.encode("utf-8"),
            headers={
                # ntfy headers must be latin-1 safe, hence "Rs" not the rupee sign
                # everywhere in this file.
                "Title": title.encode("ascii", "ignore").decode(),
                "Priority": "2" if silent else "4",
                "Tags": "chart_with_upwards_trend",
            },
            timeout=15,
        )
        if not r.ok:
            log(f"ntfy error {r.status_code}: {r.text[:150]}")
        return r.ok
    except Exception as e:
        log(f"ntfy exception: {e}")
        return False
​
​
def notify(text: str, silent: bool = False) -> bool:
    """Send to every configured channel. True if at least one succeeded."""
    sent = False
    if TG_TOKEN and TG_CHAT:
        sent |= _send_telegram(text, silent)
    if NTFY_TOPIC:
        sent |= _send_ntfy(text, silent)
    if not sent:
        log(f"[no channel delivered] {strip_html(text)[:140]}")
    return sent
​
​
# Kept so older call sites and any scripts of yours keep working.
tg = notify  # backwards-compatible alias
​
​
# ---------------------------------------------------------------------------
# BAR HISTORY
# ---------------------------------------------------------------------------
# Kotak Neo has no historical-candle API, but EMA50 on 5m needs ~50 closed bars
# before it means anything - about 4 hours of trading. So the bot keeps its own
# rolling history: it aggregates live quotes into 5m bars, appends them to
# bars.csv, and the workflow commits that file back to the repo at the end of
# each session. Tomorrow it starts warm.
#
# First run will be quiet until enough bars accumulate. To skip that, seed
# bars.csv once using download_breeze_data.py (ICICI) and commit it.
BARS_FILE = os.environ.get("BARS_FILE", "bars.csv")
MIN_BARS = 70  # EMA50 + MACD(26,9) + ATR ratio all need room
​
​
class BarStore:
    """Rolling 5m OHLC history, persisted as CSV in the repo."""
​
    def __init__(self, path: str = BARS_FILE, keep_days: int = 12):
        self.path = path
        self.keep_days = keep_days
        self.df = self._load()
        self._cur_key = None
        self._o = self._h = self._l = self._c = None
​
    def _load(self) -> pd.DataFrame:
        cols = ["open", "high", "low", "close"]
        if not os.path.exists(self.path):
            return pd.DataFrame(columns=cols, index=pd.DatetimeIndex([], tz=IST))
        try:
            d = pd.read_csv(self.path)
            ts = pd.to_datetime(d[d.columns[0]])
            ts = ts.dt.tz_convert(IST) if ts.dt.tz is not None else ts.dt.tz_localize(IST)
            d = d.set_index(ts)[cols].astype(float).sort_index()
            d = d[~d.index.duplicated(keep="last")]
            log(f"loaded {len(d)} bars from {self.path} "
                f"({d.index[0]:%d-%b %H:%M} to {d.index[-1]:%d-%b %H:%M})")
            return d
        except Exception as e:
            log(f"could not read {self.path} ({e}) - starting empty")
            return pd.DataFrame(columns=cols, index=pd.DatetimeIndex([], tz=IST))
​
    def add_tick(self, ts: pd.Timestamp, price: float) -> pd.Timestamp | None:
        """Fold a quote into the current 5m bucket. Returns a bar ts when one closes."""
        key = ts.floor("5min")
        closed = None
        if self._cur_key is None:
            self._cur_key, self._o = key, price
            self._h = self._l = self._c = price
        elif key != self._cur_key:
            self.df.loc[self._cur_key] = [self._o, self._h, self._l, self._c]
            self.df = self.df.sort_index()
            closed = self._cur_key
            self._cur_key, self._o = key, price
            self._h = self._l = self._c = price
        else:
            self._h, self._l, self._c = max(self._h, price), min(self._l, price), price
        return closed
​
    def save(self):
        try:
            cutoff = pd.Timestamp.now(tz=IST) - timedelta(days=self.keep_days)
            self.df = self.df[self.df.index >= cutoff]
            self.df.to_csv(self.path, index_label="timestamp")
            log(f"saved {len(self.df)} bars to {self.path}")
        except Exception as e:
            log(f"bar save failed: {e}")
​
​
# ---------------------------------------------------------------------------
# BROKER FEEDS
# ---------------------------------------------------------------------------
class KotakFeed:
    """Kotak Neo. TOTP login means it can authenticate itself, unattended.
​
    Data and login APIs are NOT subject to the static-IP rule - only order APIs
    are - so this works from a GitHub Actions runner.
    """
​
    name = "kotak"
    has_history = False
​
    def __init__(self):
        self.client = None
        self.at = 0.0
        self._scrips = None
​
    def connect(self):
        if self.client is not None and (time.time() - self.at) < 6 * 3600:
            return self.client
        try:
            from neo_api_client import NeoAPI
        except ImportError:
            try:
                from kotakneoapi import NeoAPI  # v3 package name
            except ImportError as e:
                raise ImportError("pip install kotakneoapi") from e
        if pyotp is None:
            raise ImportError("pip install pyotp")
​
        c = NeoAPI(
            consumer_key=os.environ["KOTAK_CONSUMER_KEY"],
            consumer_secret=os.environ["KOTAK_CONSUMER_SECRET"],
            environment="prod",
        )
        totp = pyotp.TOTP(os.environ["KOTAK_TOTP_SECRET"]).now()
        c.totp_login(
            mobile_number=os.environ["KOTAK_MOBILE"],
            ucc=os.environ["KOTAK_UCC"],
            totp=totp,
        )
        c.totp_validate(mpin=os.environ["KOTAK_MPIN"])
        self.client, self.at = c, time.time()
        log("Kotak Neo session established (TOTP)")
        return c
​
    def spot(self) -> float | None:
        c = self.connect()
        try:
            r = c.quotes(
                instrument_tokens=[{"instrument_token": "Nifty 50",
                                    "exchange_segment": "nse_cm"}],
                quote_type="ltp",
            )
            return _dig_ltp(r)
        except Exception as e:
            log(f"kotak spot failed: {e}")
            return None
​
    def _scrip_master(self) -> pd.DataFrame:
        if self._scrips is not None:
            return self._scrips
        c = self.connect()
        log("downloading Kotak scrip master (nse_fo)...")
        raw = c.scrip_master(exchange_segment="nse_fo")
        df = pd.DataFrame(raw if isinstance(raw, list) else raw.get("data", raw))
        df.columns = [str(x).strip().lower() for x in df.columns]
        self._scrips = df
        log(f"scrip master: {len(df):,} rows, columns {list(df.columns)[:8]}")
        return df
​
    def option_quotes(self, expiry, strikes, typ) -> dict:
        """strike -> LTP for the given expiry and CE/PE."""
        c = self.connect()
        out = {}
        for s in strikes:
            try:
                hit = c.search_scrip(
                    exchange_segment="nse_fo", symbol="NIFTY",
                    expiry=pd.Timestamp(expiry).strftime("%d%b%Y").upper(),
                    option_type=typ, strike_price=str(int(s)),
                )
                rows = hit if isinstance(hit, list) else hit.get("data", [])
                if not rows:
                    continue
                tok = (rows[0].get("pSymbol") or rows[0].get("instrument_token")
                       or rows[0].get("pTrdSymbol"))
                q = c.quotes(
                    instrument_tokens=[{"instrument_token": str(tok),
                                        "exchange_segment": "nse_fo"}],
                    quote_type="ltp",
                )
                ltp = _dig_ltp(q)
                if ltp:
                    out[int(s)] = ltp
            except Exception:
                continue
        return out
​
​
class BreezeFeed:
    """ICICI Direct Breeze. Has real historical data - no warmup problem.
​
    Catch: the session token dies at midnight and can only be refreshed through a
    browser login, so for unattended daily running you must update the
    BREEZE_SESSION_TOKEN secret each morning. Fine for backtests, tedious for a bot.
    """
​
    name = "breeze"
    has_history = True
​
    def __init__(self):
        self.client = None
​
    def connect(self):
        if self.client is not None:
            return self.client
        try:
            from breeze_connect import BreezeConnect
        except ImportError as e:
            raise ImportError("pip install breeze-connect") from e
        c = BreezeConnect(api_key=os.environ["BREEZE_API_KEY"])
        c.generate_session(
            api_secret=os.environ["BREEZE_API_SECRET"],
            session_token=os.environ["BREEZE_SESSION_TOKEN"],
        )
        self.client = c
        log("Breeze session established")
        return c
​
    def candles(self, days: int = 6) -> pd.DataFrame:
        c = self.connect()
        end = pd.Timestamp.now(tz=IST)
        start = end - timedelta(days=days)
        r = c.get_historical_data_v2(
            interval="5minute",
            from_date=start.strftime("%Y-%m-%dT09:15:00.000Z"),
            to_date=end.strftime("%Y-%m-%dT%H:%M:00.000Z"),
            stock_code="NIFTY", exchange_code="NSE", product_type="cash",
        )
        rows = (r or {}).get("Success") or []
        if not rows:
            raise RuntimeError(f"Breeze returned no candles: {str(r)[:200]}")
        df = pd.DataFrame(rows)
        ts = pd.to_datetime(df["datetime"])
        ts = ts.dt.tz_localize(IST) if ts.dt.tz is None else ts.dt.tz_convert(IST)
        df = df.set_index(ts)[["open", "high", "low", "close"]].astype(float)
        return df.sort_index().between_time("09:15", "15:30")
​
    def spot(self) -> float | None:
        try:
            return float(self.candles(days=3)["close"].iloc[-1])
        except Exception as e:
            log(f"breeze spot failed: {e}")
            return None
​
    def option_quotes(self, expiry, strikes, typ) -> dict:
        c = self.connect()
        right = "call" if typ == "CE" else "put"
        exp = pd.Timestamp(expiry).strftime("%Y-%m-%dT07:00:00.000Z")
        out = {}
        for s in strikes:
            try:
                r = c.get_quotes(stock_code="NIFTY", exchange_code="NFO",
                                 product_type="options", expiry_date=exp,
                                 right=right, strike_price=str(int(s)))
                rows = (r or {}).get("Success") or []
                if rows:
                    out[int(s)] = float(rows[0]["ltp"])
            except Exception:
                continue
        return out
​
​
def _dig_ltp(resp) -> float | None:
    """Pull an LTP out of a broker response without assuming its exact shape.
​
    Broker SDKs rename and reshape fields between versions far more often than
    their docs admit, so we walk the structure rather than trusting one path.
    """
    keys = {"ltp", "last_traded_price", "lasttradedprice", "lp", "last_price", "iLastTradedPrice"}
    stack = [resp]
    while stack:
        x = stack.pop()
        if isinstance(x, dict):
            for k, v in x.items():
                if str(k).strip().lower() in keys:
                    try:
                        f = float(v)
                        if f > 0:
                            return f
                    except (TypeError, ValueError):
                        pass
                if isinstance(v, (dict, list)):
                    stack.append(v)
        elif isinstance(x, list):
            stack.extend(x)
    return None
​
​
def make_feed():
    if BROKER == "breeze":
        return BreezeFeed()
    if BROKER == "kotak":
        return KotakFeed()
    raise SystemExit(f"BROKER must be 'kotak' or 'breeze', got {BROKER!r}")
​
# ---------------------------------------------------------------------------
# TRADE PLAN
# ---------------------------------------------------------------------------
def build_plan(feed, spot: float, typ: str, now: pd.Timestamp):
    expiry = next_weekly_expiry(now)
    atm = int(round(spot / STRIKE_STEP) * STRIKE_STEP)
    strikes = [atm + i * STRIKE_STEP if typ == "CE" else atm - i * STRIKE_STEP
               for i in range(0, 21)]
​
    quotes = {}
    try:
        quotes = feed.option_quotes(expiry, strikes, typ)
    except Exception as e:
        log(f"quote fetch failed: {e}")
​
    source = "live"
    if quotes:
        band = {k: v for k, v in quotes.items() if MIN_PREMIUM <= v <= MAX_PREMIUM}
        if not band:
            return None
        strike = min(band, key=lambda k: abs(band[k] - TARGET_PREMIUM))
        prem = band[strike]
    else:
        source = "model"
        best, err = None, 1e9
        for s in strikes:
            p = bs_price(spot, s, t_years(now, expiry), 0.13, typ)
            if abs(p - TARGET_PREMIUM) < err:
                best, err, prem = s, abs(p - TARGET_PREMIUM), p
        if best is None or not (MIN_PREMIUM <= prem <= MAX_PREMIUM):
            return None
        strike = best
​
    risk_per_lot = STOP_PCT * prem * LOT_SIZE
    lots = int(CAPITAL * RISK_PER_TRADE // risk_per_lot)
    if lots < 1:
        return {"error": (f"Need Rs {risk_per_lot / RISK_PER_TRADE:,.0f} capital for 1 lot "
                          f"at {RISK_PER_TRADE*100:.0f}% risk. You have Rs {CAPITAL:,.0f}."),
                "strike": strike, "premium": prem}
    if prem * lots * LOT_SIZE > CAPITAL * 0.35:
        return {"error": "Outlay would exceed 35% of capital.", "strike": strike, "premium": prem}
​
    today = now.normalize()
    if today.tzinfo is not None:
        today = today.tz_localize(None)
    return {
        "strike": strike, "side": typ, "expiry": expiry,
        "dte": (expiry - today).days, "premium": prem, "source": source,
        "lots": lots, "outlay": prem * lots * LOT_SIZE,
        "stop": prem * (1 - STOP_PCT), "target1": prem * (1 + TARGET1_PCT),
        "risk": risk_per_lot * lots,
    }
​
​
# ---------------------------------------------------------------------------
# MESSAGES
# ---------------------------------------------------------------------------
def msg_watch(side, d):
    return (f"<b>WATCH - {'CALL (CE)' if side == 'CE' else 'PUT (PE)'} forming</b>\n"
            f"<i>Not a trade. MACD has not crossed.</i>\n\n"
            f"NIFTY <b>{d['close']:,.1f}</b>\n"
            f"EMA20 {d['ema_fast']:,.1f} | EMA50 {d['ema_slow']:,.1f}\n"
            f"RSI {d['rsi']:.1f} | hist {d['macd_hist']:+.2f}\n"
            f"ATR {d['atr']:.1f} | gap {d['ema_gap_atr']:.2f} ATR\n"
            f"<code>{d['time']:%H:%M}</code>")
​
​
def msg_entry(side, d, p):
    return (f"<b>&gt;&gt;&gt; BUY {'CALL (CE)' if side == 'CE' else 'PUT (PE)'}</b>\n\n"
            f"<b>{p['strike']} {side}</b>  exp {p['expiry']:%d-%b} ({p['dte']}d)\n"
            f"Premium  <b>Rs {p['premium']:.2f}</b>"
            f"{' (model est.)' if p['source'] == 'model' else ''}\n"
            f"Lots     <b>{p['lots']}</b> ({p['lots'] * LOT_SIZE} qty)\n"
            f"Outlay   Rs {p['outlay']:,.0f}\n\n"
            f"Stop     Rs {p['stop']:.2f}  (-{STOP_PCT*100:.0f}%)\n"
            f"Target 1 Rs {p['target1']:.2f}  (+{TARGET1_PCT*100:.0f}%, book half)\n"
            f"Then     trail the 20 EMA\n"
            f"Risk     Rs {p['risk']:,.0f}\n\n"
            f"NIFTY {d['close']:,.1f} | RSI {d['rsi']:.1f} | "
            f"MACD {d['macd']:+.1f}/{d['macd_signal']:+.1f}\n"
            f"Hard exit 15:10. No overnight.\n"
            f"<code>{d['time']:%H:%M}</code>")
​
​
def msg_manage(kind, pos, prem):
    pct = 100 * (prem / pos["premium"] - 1)
    head = {"target1": "TARGET 1 HIT - book half", "stop": "STOP HIT - exit now",
            "trail": "TRAIL BROKEN - exit the rest",
            "structural": "TREND BROKEN (EMA50) - exit",
            "time": "15:10 - close everything"}[kind]
    return (f"<b>{head}</b>\n\n{pos['strike']} {pos['side']}\n"
            f"Entry Rs {pos['premium']:.2f} -> now Rs {prem:.2f} (<b>{pct:+.1f}%</b>)")
​
​
# ---------------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------------
def main():
    feed = make_feed()
    store = BarStore()
    last_bar = None
    pos = None
    trades_today = 0
    consec_losses = 0
    signals = 0
    day = None
    warned_warmup = False
​
    log(f"broker: {feed.name} (history API: {'yes' if feed.has_history else 'no'})")
​
    channels = []
    if TG_TOKEN and TG_CHAT:
        channels.append("Telegram")
    if NTFY_TOPIC:
        channels.append(f"ntfy ({NTFY_SERVER})")
    if not channels:
        log("FATAL: no notification channel configured.")
        log("  Set TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID, and/or NTFY_TOPIC.")
        log("  Running without one would just burn six hours of CPU silently.")
        sys.exit(1)
    log(f"channels: {', '.join(channels)}")
​
    now = pd.Timestamp.now(tz=IST)
    if now.dayofweek >= 5 or now.normalize().tz_localize(None) in HOLIDAYS:
        log("market closed today - exiting")
        return
​
    notify(f"<b>Bot online</b>\nNIFTY 5m EMA/MACD/RSI\nCapital Rs {CAPITAL:,.0f} | "
           f"risk {RISK_PER_TRADE*100:.0f}%/trade\nWatching until 15:10.", silent=True)
​
    # Poll cadence. With a history API we only need to wake on 5m boundaries.
    # Without one we must sample often enough to build honest OHLC bars.
    poll = 300 if feed.has_history else 20
​
    while True:
        now = pd.Timestamp.now(tz=IST)
        if now.time() >= dtime(15, 15):
            break
​
        if feed.has_history:
            # Sleep to just past the next 5m boundary so the bar has closed and the
            # broker has published it. 20s of slack is enough in practice.
            time.sleep(max((5 - now.minute % 5) * 60 - now.second + 20, 5))
        else:
            time.sleep(poll)
​
        try:
            if feed.has_history:
                df = feed.candles(days=6)
                now2 = pd.Timestamp.now(tz=IST)
                # Drop the bar still forming.
                if len(df) and (now2 - df.index[-1]).total_seconds() < 300:
                    df = df.iloc[:-1]
            else:
                # No history API: fold this quote into our own rolling bars, and
                # only act when a 5m bucket actually closes.
                px = feed.spot()
                if px is None:
                    continue
                closed = store.add_tick(pd.Timestamp.now(tz=IST), px)
                if closed is None:
                    continue
                df = store.df
​
            if len(df) < MIN_BARS:
                if not warned_warmup:
                    warned_warmup = True
                    msg = (f"<b>Warming up</b>\n{len(df)}/{MIN_BARS} bars. "
                           f"EMA50 and MACD need history before any signal is real.\n"
                           f"Alerts start once there is enough.")
                    log(msg.replace("\n", " | "))
                    notify(msg, silent=True)
                continue
​
            if df.index[-1] == last_bar:
                continue
            last_bar = df.index[-1]
​
            if last_bar.normalize() != day:
                day, trades_today, consec_losses, signals = last_bar.normalize(), 0, 0, 0
​
            d = add_indicators(df)
            i = len(d) - 1
            state, detail = evaluate(d, i)
            spot = float(d["close"].iloc[i])
            bar_t = d.index[i]
            log(f"bar {bar_t:%H:%M} spot {spot:,.1f} rsi {detail.get('rsi', 0):.1f} "
                f"hist {detail.get('macd_hist', 0):+.2f} -> {state}")
​
            # ---- manage an open position ----
            if pos is not None:
                prem = None
                try:
                    q = feed.option_quotes(pos["expiry"], [pos["strike"]], pos["side"])
                    prem = q.get(pos["strike"])
                except Exception:
                    pass
                if prem is None:
                    prem = bs_price(spot, pos["strike"], t_years(bar_t, pos["expiry"]),
                                    0.13, pos["side"])
​
                row = d.iloc[i]
                exit_kind = None
                if prem <= pos["stop"]:
                    exit_kind = "stop"
                elif not pos["t1"] and prem >= pos["target1"]:
                    pos["t1"] = True
                    notify(msg_manage("target1", pos, prem))
                elif pos["t1"] and ((pos["side"] == "CE" and spot < row["ema_fast"])
                                    or (pos["side"] == "PE" and spot > row["ema_fast"])):
                    exit_kind = "trail"
                elif ((pos["side"] == "CE" and spot < row["ema_slow"])
                      or (pos["side"] == "PE" and spot > row["ema_slow"])):
                    exit_kind = "structural"
                elif bar_t.time() >= HARD_EXIT:
                    exit_kind = "time"
​
                if exit_kind:
                    notify(msg_manage(exit_kind, pos, prem))
                    if prem < pos["premium"]:
                        consec_losses += 1
                    else:
                        consec_losses = 0
                    pos = None
                    if consec_losses >= MAX_CONSEC_LOSSES:
                        notify(f"<b>Circuit breaker</b>\n{consec_losses} losses in a row. "
                           f"No more entries today.")
                continue
​
            # ---- look for a new entry ----
            in_window = SESSION_START <= bar_t.time() <= LAST_ENTRY
            blocked = trades_today >= MAX_TRADES_DAY or consec_losses >= MAX_CONSEC_LOSSES
            if not in_window or blocked:
                continue
​
            if state in ("enter_ce", "enter_pe"):
                signals += 1
                typ = "CE" if state == "enter_ce" else "PE"
                plan = build_plan(feed, spot, typ, bar_t)
                if plan is None:
                    notify(f"<b>Signal {typ} at {spot:,.0f} - SKIPPED</b>\n"
                       f"No strike in the Rs {MIN_PREMIUM:.0f}-{MAX_PREMIUM:.0f} band.")
                elif "error" in plan:
                    notify(f"<b>Signal {typ} - SKIPPED</b>\n{plan['error']}")
                else:
                    notify(msg_entry(typ, detail, plan))
                    plan["t1"] = False
                    pos = plan
                    trades_today += 1
​
            elif state in ("watch_ce", "watch_pe"):
                notify(msg_watch("CE" if state == "watch_ce" else "PE", detail), silent=True)
​
        except Exception as e:
            log(f"ERROR: {e}")
            traceback.print_exc()
            time.sleep(30)
​
    # Persist the day's bars so tomorrow starts warm. The workflow commits this
    # file back to the repo - without that step the bot re-warms from scratch
    # every morning and cannot signal before roughly 13:30.
    if not feed.has_history:
        store.save()
​
    notify(f"<b>Session done</b>\nSignals {signals} | trades taken {trades_today}\n"
           f"Close anything still open.", silent=True)
    log("finished")
​
​
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        notify(f"<b>Bot crashed</b>\n<code>{str(e)[:300]}</code>")
        traceback.print_exc()
        sys.exit(1)
