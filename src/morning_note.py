"""
Morning Note: ETF Holdings Movement Tracker
============================================
Delivery:   ~8:00 AM EST on U.S. trading days via GitHub Actions (cron 08:00 UTC; ~5 hr GitHub delay)

Two signal layers
-----------------
1. EOD signal (Yahoo Finance)
   Source:    Yahoo Finance via yfinance
   Measure:   Prior close-to-close percent change on each ETF in the watchlist
   Threshold: >= +/- 2%

2. Pre-market constituent signal
   Source:    Yahoo Finance (yfinance Ticker.info["preMarketPrice"])
   Reference: Nasdaq pre-market page
              https://www.nasdaq.com/market-activity/stocks/<ticker>/pre-market
   Measure:   Constituent pre-market price vs. prior close
   Threshold: >= +/- 1%  (lower bar; pre-market sessions are noisier)
   Trigger:   Any constituent moving >= 1% pre-market causes ALL parent ETFs
              in the watchlist to appear in the report (Option A)

News sources (aggregated, deduplicated)
----------------------------------------
- Yahoo Finance news feed via yfinance
- NewsAPI.org free tier  (100 req/day)  -- set NEWS_API_KEY secret
- GNews free tier        (100 req/day)  -- set GNEWS_API_KEY secret

Email delivery: Gmail SMTP via App Password
"""
import os
from dotenv import load_dotenv

if os.path.exists(".env"):
    load_dotenv()

import time  

import smtplib
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import yfinance as yf
import pandas_market_calendars as mcal
import requests
from anthropic import Anthropic

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WATCHLIST = ["QQQ", "SMH", "DTCR", "SPY"]

EOD_THRESHOLD       = 0.02   # 2%  ETF prior close-to-close
PREMARKET_THRESHOLD = 0.01   # 1%  constituent pre-market vs. prior close
TOP_N_CONSTITUENTS  = 5      # Max constituent rows per ETF section

RECIPIENT_EMAIL    = os.environ["RECIPIENT_EMAIL"]
SENDER_EMAIL       = os.environ["SENDER_EMAIL"]
GMAIL_APP_PASSWORD = os.environ["GMAIL_APP_PASSWORD"]
ANTHROPIC_API_KEY  = os.environ["ANTHROPIC_API_KEY"]
NEWS_API_KEY       = os.environ.get("NEWS_API_KEY",  "")  # newsapi.org
GNEWS_API_KEY      = os.environ.get("GNEWS_API_KEY", "")  # gnews.io

anthropic = Anthropic(api_key=ANTHROPIC_API_KEY)

# ---------------------------------------------------------------------------
# Reference data — holdings and GICS
# ---------------------------------------------------------------------------

# Top constituents per ETF as of Q2 2026. Review quarterly.
#   QQQ  -> invesco.com/qqq-etf
#   SMH  -> vaneck.com/etf/SMH
#   SPY  -> ssga.com/us/en/institutional/etfs/funds/spdr-sp-500-etf-trust-spy
#   DTCR -> globalxetfs.com/funds/dtcr
FALLBACK_HOLDINGS: dict[str, list[str]] = {
    "QQQ":  ["NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "GOOG", "AVGO", "TSLA", "COST"],
    "SMH":  ["NVDA", "TSM", "MU", "AMD", "INTC", "AVGO", "QCOM", "TXN", "LRCX", "KLAC"],
    "SPY":  ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "BRK.B", "TSLA", "AVGO"],
    "DTCR": ["EQIX", "DLR", "AMT", "CCI", "MRVL", "SBAC", "IRM", "VRT", "GDS", "VNET"],
}

# GICS sub-industry map for sympathy flagging.
GICS_MAP: dict[str, str] = {
    "NVDA": "Semiconductors",         "AMD":  "Semiconductors",
    "INTC": "Semiconductors",         "AVGO": "Semiconductors",
    "QCOM": "Semiconductors",         "TXN":  "Semiconductors",
    "MU":   "Semiconductors",         "TSM":  "Semiconductors",
    "AMAT": "Semiconductor Equipment","KLAC": "Semiconductor Equipment",
    "LRCX": "Semiconductor Equipment","ASML": "Semiconductor Equipment",
    "MSFT": "Systems Software",       "CRWD": "Systems Software",
    "NET":  "Systems Software",       "ZS":   "Systems Software",
    "PANW": "Systems Software",       "FTNT": "Systems Software",
    "S":    "Systems Software",       "OKTA": "Systems Software",
    "CYBR": "Systems Software",       "TENB": "Systems Software",
    "GOOGL":"Interactive Media",      "GOOG": "Interactive Media",
    "META": "Interactive Media",
    "AMZN": "Internet Retail",
    "AAPL": "Technology Hardware",
    "TSLA": "Automobile Manufacturers",
    "COST": "Hypermarkets",
    "LLY":  "Pharmaceuticals",
    "RPM":  "Specialty Chemicals",
}

# ---------------------------------------------------------------------------
# Trading calendar
# ---------------------------------------------------------------------------

def is_trading_day(date: datetime) -> bool:
    nyse = mcal.get_calendar("NYSE")
    return not nyse.schedule(
        start_date=date.strftime("%Y-%m-%d"),
        end_date=date.strftime("%Y-%m-%d"),
    ).empty


def get_prior_trading_day(date: datetime) -> datetime:
    nyse = mcal.get_calendar("NYSE")
    candidate = date - timedelta(days=1)
    while True:
        if not nyse.schedule(
            start_date=candidate.strftime("%Y-%m-%d"),
            end_date=candidate.strftime("%Y-%m-%d"),
        ).empty:
            return candidate
        candidate -= timedelta(days=1)

SPLIT_THRESHOLD = 0.35  # abs(pct_change) above which we check for a split event

def detect_split(ticker: str, as_of: datetime) -> float | None:
    """Return the split ratio if yfinance records a split on or within 2 days of as_of, else None."""
    try:
        splits = yf.Ticker(ticker).splits
        if splits.empty:
            return None
        for days_back in range(3):
            check = (as_of - timedelta(days=days_back)).strftime("%Y-%m-%d")
            mask  = splits.index.strftime("%Y-%m-%d") == check
            if mask.any():
                return float(splits[mask].iloc[-1])
        return None
    except Exception as exc:
        log.warning(f"[Split] Check failed for {ticker}: {exc}")
        return None

# ---------------------------------------------------------------------------
# EOD price data — Yahoo Finance
# ---------------------------------------------------------------------------

def get_eod_change(ticker: str, prior_date: datetime, current_date: datetime) -> dict | None:
    """Prior close-to-close percent change. Source: Yahoo Finance via yfinance."""
    start = (prior_date - timedelta(days=7)).strftime("%Y-%m-%d")
    end   = (current_date + timedelta(days=1)).strftime("%Y-%m-%d")
    data  = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if data.empty or len(data) < 2:
        log.warning(f"[EOD] Insufficient data for {ticker}")
        return None
    closes        = data["Close"].dropna()
    prior_close   = float(closes.iloc[-2])
    current_close = float(closes.iloc[-1])
    pct           = (current_close - prior_close) / prior_close
    return {
        "ticker":        ticker,
        "prior_close":   round(prior_close,   2),
        "current_close": round(current_close, 2),
        "pct_change":    round(pct,            4),
        "direction":     "up" if pct > 0 else "down",
    }

# ---------------------------------------------------------------------------
# Pre-market price data — Yahoo Finance / Nasdaq reference
# ---------------------------------------------------------------------------

def get_premarket_change(ticker: str) -> dict | None:
    """
    Pre-market percent change vs. prior close.

    Data source : Yahoo Finance (yfinance Ticker.info preMarketPrice field)
    Reference   : Nasdaq pre-market page
                  https://www.nasdaq.com/market-activity/stocks/<ticker>/pre-market

    Returns None when outside the 4:00–9:30 AM ET pre-market window or when
    Yahoo Finance reports no pre-market activity for the ticker.
    """
    try:
        info            = yf.Ticker(ticker).info
        premarket_price = info.get("preMarketPrice")
        prior_close     = info.get("previousClose") or info.get("regularMarketPreviousClose")
        if not premarket_price or not prior_close:
            log.debug(f"[Pre-market] No pre-market data for {ticker}")
            return None
        pct = (premarket_price - prior_close) / prior_close
        return {
            "ticker":          ticker,
            "prior_close":     round(prior_close,     2),
            "premarket_price": round(premarket_price, 2),
            "pct_change":      round(pct,              4),
            "direction":       "up" if pct > 0 else "down",
            "nasdaq_ref_url":  f"https://www.nasdaq.com/market-activity/stocks/{ticker.lower()}/pre-market",
        }
    except Exception as exc:
        log.warning(f"[Pre-market] Fetch failed for {ticker}: {exc}")
        return None

# ---------------------------------------------------------------------------
# News aggregation — Yahoo Finance + NewsAPI + GNews
# ---------------------------------------------------------------------------

def _yahoo_news(ticker: str) -> list[str]:
    try:
        items = yf.Ticker(ticker).news or []
        out   = []
        for item in items[:6]:
            title = item.get("title") or (item.get("content") or {}).get("title", "")
            if title:
                out.append(title)
        return out
    except Exception as exc:
        log.warning(f"[Yahoo news] {ticker}: {exc}")
        return []


def _newsapi_headlines(ticker: str, company: str) -> list[str]:
    if not NEWS_API_KEY:
        return []
    try:
        r = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q":        f"{ticker} OR \"{company}\"",
                "sortBy":   "publishedAt",
                "pageSize": 5,
                "language": "en",
                "apiKey":   NEWS_API_KEY,
            },
            timeout=10,
        )
        r.raise_for_status()
        return [a["title"] for a in r.json().get("articles", []) if a.get("title")]
    except Exception as exc:
        log.warning(f"[NewsAPI] {ticker}: {exc}")
        return []


def _gnews_headlines(ticker: str, company: str) -> list[str]:
    if not GNEWS_API_KEY:
        return []
    try:
        r = requests.get(
            "https://gnews.io/api/v4/search",
            params={
                "q":      f"{ticker} {company}",
                "lang":   "en",
                "max":    5,
                "sortby": "publishedAt",
                "apikey": GNEWS_API_KEY,
            },
            timeout=10,
        )
        r.raise_for_status()
        return [a["title"] for a in r.json().get("articles", []) if a.get("title")]
    except Exception as exc:
        log.warning(f"[GNews] {ticker}: {exc}")
        return []


def fetch_all_news(ticker: str, company: str) -> list[str]:
    """Aggregate and deduplicate headlines from all three sources. Returns up to 8."""
    seen, merged = set(), []
    for headline in (
        _newsapi_headlines(ticker, company)
        + _gnews_headlines(ticker, company)
        + _yahoo_news(ticker)
    ):
        key = headline.lower().strip()
        if key not in seen:
            seen.add(key)
            merged.append(headline)
        if len(merged) >= 8:
            break
    return merged

# ---------------------------------------------------------------------------
# Catalyst classification via Claude
# ---------------------------------------------------------------------------

CATALYST_TYPES = (
    "[Earnings], [Guidance Update], [M&A], [New Partnership], [Product Launch], "
    "[Macro/Sector], [Analyst Upgrade], [Analyst Downgrade], [Regulatory Filing], "
    "[Stock Split], [Sympathy Move], or [Unknown]"
)


def classify_catalyst(
    ticker: str,
    pct_change: float,
    headlines: list[str],
    is_premarket: bool = False,
) -> str:
    direction = "up" if pct_change > 0 else "down"
    pct_str   = f"{abs(pct_change) * 100:.1f}%"
    session   = "pre-market" if is_premarket else "prior trading session"

    if headlines:
        prompt = (
            f"{ticker} is moving {direction} {pct_str} in the {session}.\n\n"
            f"Recent headlines:\n"
            + "\n".join(f"- {h}" for h in headlines)
            + f"\n\nIn one sentence (max 20 words), identify the primary catalyst. "
            f"Start with the type in brackets. Choose from: {CATALYST_TYPES}. "
            f"Example: [Earnings] Beat on EPS and raised full-year guidance."
        )
    else:
        prompt = (
            f"{ticker} is moving {direction} {pct_str} in the {session}. "
            f"No headlines available. "
            f"Respond with: [Unknown] Insufficient news data to classify catalyst."
        )

    try:
        msg = anthropic.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=80,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip()
    except Exception as exc:
        log.warning(f"[Claude] Classification failed for {ticker}: {exc}")
        return "[Unknown] Catalyst classification unavailable."

# ---------------------------------------------------------------------------
# Sympathy flagging
# ---------------------------------------------------------------------------

def flag_sympathy(movers: list[dict]) -> list[dict]:
    """
    Flags a mover as a sympathy trade when it shares the same GICS sub-industry
    as the top absolute mover and is not itself the primary driver.
    """
    if not movers:
        return movers
    primary_gics = movers[0].get("gics", "Unknown")
    for i, mover in enumerate(movers):
        mover["sympathy"] = (
            i > 0
            and mover.get("gics") == primary_gics
            and primary_gics != "Unknown"
        )
    return movers

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def get_company_name(ticker: str) -> str:
    try:
        return yf.Ticker(ticker).info.get("shortName", ticker)
    except Exception:
        return ticker

# ---------------------------------------------------------------------------
# Email formatting helpers
# ---------------------------------------------------------------------------

def _color(pct: float) -> str:
    return "#16a34a" if pct > 0 else "#dc2626"

def _arrow(pct: float) -> str:
    return "&#9650;" if pct > 0 else "&#9660;"

def _pct_str(pct: float, sign: bool = True) -> str:
    prefix = "+" if (sign and pct > 0) else ""
    return f"{prefix}{pct * 100:.2f}%"


def format_mover_row(mover: dict) -> str:
    pct      = mover["pct_change"]
    is_pm    = "premarket_price" in mover
    label    = "PRE-MKT" if is_pm else "EOD"
    price    = mover.get("premarket_price") or mover.get("current_close", "")
    ref_url  = mover.get("nasdaq_ref_url", "")

    source_html = (
        f'<a href="{ref_url}" style="color:#94a3b8;font-size:10px;text-decoration:none;"'
        f' target="_blank">Nasdaq ref &rarr;</a>'
        if ref_url else
        '<span style="color:#94a3b8;font-size:10px;">Yahoo Finance</span>'
    )
    sympathy_badge = (
        ' <span style="background:#fef9c3;color:#854d0e;font-size:10px;'
        'padding:1px 5px;border-radius:3px;font-weight:700;">SYMPATHY</span>'
        if mover.get("sympathy") else ""
    )
    signal_badge = (
        f'<span style="background:#eff6ff;color:#1d4ed8;font-size:10px;'
        f'padding:1px 5px;border-radius:3px;font-weight:700;margin-right:4px;">{label}</span>'
    )
    catalyst     = mover.get("catalyst", "[Unknown] No data available.")
    top_headline = mover.get("top_headline", "")
    is_split     = mover.get("split_ratio") is not None

    headline_html = (
        f'<br><span style="color:#94a3b8;font-size:11px;font-style:italic;">'
        f'&ldquo;{top_headline}&rdquo;</span>'
        if top_headline else ""
    )
    split_badge = (
        ' <span style="background:#f1f5f9;color:#475569;font-size:10px;'
        'padding:1px 5px;border-radius:3px;font-weight:700;">SPLIT</span>'
        if is_split else ""
    )
    move_color = "#64748b" if is_split else _color(pct)

    return f"""
    <tr>
      <td style="padding:10px 12px;border-bottom:1px solid #f1f5f9;vertical-align:top;">
        {signal_badge}
        <strong style="font-size:13px;">{mover['ticker']}</strong>
        <span style="color:#64748b;font-size:11px;margin-left:5px;">
          {mover.get('company','')}
        </span>{sympathy_badge}<br>
        <span style="font-size:11px;color:#94a3b8;">
          Prior close ${mover['prior_close']} &rarr; ${price} &nbsp; {source_html}
        </span>
      </td>
      <td style="padding:10px 12px;border-bottom:1px solid #f1f5f9;
                 text-align:right;vertical-align:top;white-space:nowrap;">
        <span style="color:{move_color};font-weight:700;font-size:14px;">
          {_arrow(pct)} {abs(pct)*100:.2f}%
        </span>{split_badge}
      </td>
      <td style="padding:10px 12px;border-bottom:1px solid #f1f5f9;
                 color:#374151;font-size:12px;vertical-align:top;">
        {catalyst}{headline_html}
      </td>
    </tr>"""


def format_etf_section(etf: dict) -> str:
    eod = etf.get("eod")
    pm  = etf.get("premarket")

    pm_html = (
        f'<span style="margin-left:14px;font-size:13px;color:{_color(pm["pct_change"])};font-weight:600;">'
        f'Pre-Mkt {_arrow(pm["pct_change"])} {abs(pm["pct_change"])*100:.2f}%</span>'
        if pm else ""
    )
    eod_html = (
        f'<span style="margin-left:10px;font-size:13px;color:{_color(eod["pct_change"])};font-weight:600;">'
        f'EOD {_arrow(eod["pct_change"])} {abs(eod["pct_change"])*100:.2f}%</span>'
        if eod else ""
    )
    trigger_badge = (
        f'<span style="background:#fef3c7;color:#92400e;font-size:10px;'
        f'padding:2px 8px;border-radius:10px;font-weight:700;margin-left:10px;">'
        f'TRIGGERED: {etf.get("triggered_by","")}</span>'
    )

    mover_rows = "".join(format_mover_row(m) for m in etf.get("movers", []))
    if not mover_rows:
        mover_rows = (
            '<tr><td colspan="3" style="padding:10px 12px;color:#64748b;font-style:italic;">'
            'No constituent moves exceeded the 1% pre-market threshold.</td></tr>'
        )

    return f"""
    <div style="margin-bottom:28px;">
      <div style="background:#1e293b;color:#f8fafc;padding:12px 16px;
                  border-radius:6px 6px 0 0;line-height:1.8;">
        <span style="font-size:19px;font-weight:700;">{etf['ticker']}</span>
        {pm_html}{eod_html}{trigger_badge}
      </div>
      <table style="width:100%;border-collapse:collapse;background:#fff;
                    border:1px solid #e2e8f0;border-top:none;
                    border-radius:0 0 6px 6px;">
        <thead>
          <tr style="background:#f8fafc;">
            <th style="padding:8px 12px;text-align:left;font-size:11px;
                       color:#64748b;font-weight:600;
                       border-bottom:1px solid #e2e8f0;">CONSTITUENT</th>
            <th style="padding:8px 12px;text-align:right;font-size:11px;
                       color:#64748b;font-weight:600;
                       border-bottom:1px solid #e2e8f0;">MOVE</th>
            <th style="padding:8px 12px;text-align:left;font-size:11px;
                       color:#64748b;font-weight:600;
                       border-bottom:1px solid #e2e8f0;">CATALYST</th>
          </tr>
        </thead>
        <tbody>{mover_rows}</tbody>
      </table>
    </div>"""


def format_quiet_pill(entry: dict) -> str:
    eod = entry.get("eod")
    pm  = entry.get("premarket")
    parts = []
    if eod:
        parts.append(
            f'EOD <span style="color:{_color(eod["pct_change"])};font-weight:600;">'
            f'{_pct_str(eod["pct_change"])}</span>'
        )
    if pm:
        parts.append(
            f'Pre-Mkt <span style="color:{_color(pm["pct_change"])};font-weight:600;">'
            f'{_pct_str(pm["pct_change"])}</span>'
        )
    detail = " &nbsp;|&nbsp; ".join(parts) if parts else "No data"
    return (
        f'<div style="display:inline-block;background:#f1f5f9;color:#475569;'
        f'padding:5px 12px;border-radius:20px;font-size:12px;margin:3px;">'
        f'<strong>{entry["ticker"]}</strong> &nbsp; {detail}</div>'
    )


def build_email(
    triggered: list[dict],
    quiet: list[dict],
    report_date: str,
    run_time_utc: str,
) -> str:
    triggered_html = "".join(format_etf_section(e) for e in triggered)
    quiet_html = (
        "".join(format_quiet_pill(e) for e in quiet)
        or '<span style="color:#64748b;font-size:13px;">All holdings triggered today.</span>'
    )
    n = len(triggered)
    badge = f"{n} holding{'s' if n != 1 else ''} triggered"
    no_trigger = (
        '<div style="color:#64748b;font-style:italic;margin-bottom:24px;'
        'padding:16px;background:#fff;border:1px solid #e2e8f0;border-radius:6px;">'
        'No pre-market constituent moves exceeded 1% and no ETF exceeded 2% EOD '
        'in the prior session.'
        '</div>'
    )

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
</head>
<body style="margin:0;padding:0;background:#f1f5f9;
             font-family:'Segoe UI',Arial,sans-serif;">
<div style="max-width:700px;margin:0 auto;padding:20px 12px;">

  <!-- Header -->
  <div style="background:#0f172a;color:#f8fafc;padding:20px 24px;
              border-radius:8px;margin-bottom:20px;">
    <div style="font-size:10px;color:#94a3b8;letter-spacing:1.5px;
                text-transform:uppercase;">Morning Note</div>
    <div style="font-size:22px;font-weight:700;margin-top:2px;">
      Portfolio Holdings Briefing
    </div>
    <div style="font-size:12px;color:#94a3b8;margin-top:4px;">
      {report_date} &nbsp;&middot;&nbsp; {run_time_utc} UTC
    </div>
    <div style="margin-top:12px;display:flex;gap:8px;flex-wrap:wrap;">
      <span style="background:#1e40af;color:#bfdbfe;padding:3px 10px;
                   border-radius:12px;font-size:12px;font-weight:600;">
        {badge}
      </span>
      <span style="background:#064e3b;color:#a7f3d0;padding:3px 10px;
                   border-radius:12px;font-size:12px;font-weight:600;">
        Pre-mkt threshold: 1% &nbsp;|&nbsp; EOD threshold: 2%
      </span>
    </div>
    <!-- Legend -->
    <div style="margin-top:10px;font-size:11px;color:#94a3b8;line-height:1.8;">
      <span style="background:#eff6ff;color:#1d4ed8;padding:1px 5px;
                   border-radius:3px;font-weight:700;">PRE-MKT</span>
      &nbsp;Constituent pre-market move &mdash; Yahoo Finance pricing,
      Nasdaq.com reference
      &nbsp;&nbsp;
      <span style="background:#eff6ff;color:#1d4ed8;padding:1px 5px;
                   border-radius:3px;font-weight:700;">EOD</span>
      &nbsp;Prior close-to-close &mdash; Yahoo Finance
    </div>
  </div>

  <!-- Triggered ETF sections -->
  {triggered_html if triggered_html else no_trigger}

  <!-- Quiet holdings -->
  <div style="background:#fff;border:1px solid #e2e8f0;border-radius:6px;
              padding:14px 16px;margin-bottom:20px;">
    <div style="font-size:11px;font-weight:600;color:#64748b;
                text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">
      Below Threshold
    </div>
    {quiet_html}
  </div>

  <!-- Footer -->
  <div style="font-size:10px;color:#94a3b8;text-align:center;
              padding-top:4px;line-height:1.8;">
    Personal use only &nbsp;|&nbsp;
    EOD prices: Yahoo Finance &nbsp;|&nbsp;
    Pre-market prices: Yahoo Finance &nbsp;|&nbsp;
    Pre-market reference: Nasdaq.com &nbsp;|&nbsp;
    News: Yahoo Finance, NewsAPI, GNews &nbsp;|&nbsp;
    Holdings approximate, updated manually &nbsp;|&nbsp;
    Not investment advice
  </div>

</div>
</body>
</html>"""

# ---------------------------------------------------------------------------
# Email delivery
# ---------------------------------------------------------------------------

def send_email(subject: str, html_body: str) -> None:
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = SENDER_EMAIL
    msg["To"]      = RECIPIENT_EMAIL
    msg.attach(MIMEText(html_body, "html"))
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(SENDER_EMAIL, GMAIL_APP_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, msg.as_string())
    log.info(f"Email sent to {RECIPIENT_EMAIL}")

# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main() -> None:
    # GitHub Actions runs in UTC.
    # Cron: 08:00 UTC = 03:00 EST (UTC-5); GitHub ~5 hr delay targets ~08:00 EST actual delivery
    # See workflow YAML for seasonal cron adjustment.
    now_utc = datetime.utcnow()

    if not is_trading_day(now_utc):
        log.info(f"{now_utc.strftime('%Y-%m-%d')} is not a U.S. trading day. Exiting.")
        return

    prior_day    = get_prior_trading_day(now_utc)
    report_date  = now_utc.strftime("%A, %B %d, %Y")
    run_time_utc = now_utc.strftime("%H:%M")
    log.info(f"Run date: {report_date} | Prior trading day: {prior_day.strftime('%Y-%m-%d')}")

    # ------------------------------------------------------------------
    # Step 1: EOD data for all ETFs
    # ------------------------------------------------------------------
    eod_map: dict[str, dict | None] = {}
    for ticker in WATCHLIST:
        eod_map[ticker] = get_eod_change(ticker, prior_day, now_utc)
        log.info(
            f"[EOD] {ticker}: "
            + (f"{eod_map[ticker]['pct_change']*100:.2f}%" if eod_map[ticker] else "no data")
        )

    # ------------------------------------------------------------------
    # Step 2: Pre-market scan — fetch once per unique constituent
    # ------------------------------------------------------------------
    constituent_to_etfs: dict[str, list[str]] = defaultdict(list)
    for etf in WATCHLIST:
        for c in FALLBACK_HOLDINGS.get(etf, []):
            constituent_to_etfs[c].append(etf)

    constituent_pm: dict[str, dict | None] = {}
    for constituent in constituent_to_etfs:
        constituent_pm[constituent] = get_premarket_change(constituent)
        time.sleep(0.5)
        if constituent_pm[constituent]:
            log.info(
                f"[Pre-mkt] {constituent}: "
                f"{constituent_pm[constituent]['pct_change']*100:.2f}%"
            )

    # ------------------------------------------------------------------
    # Step 3: Determine triggered vs. quiet ETFs
    #
    #   Triggered when:
    #     (a) ETF EOD move >= 2%, OR
    #     (b) >= 1 constituent has a pre-market move >= 1%
    # ------------------------------------------------------------------
    triggered_etfs: list[dict] = []
    quiet_etfs:     list[dict] = []

    for ticker in WATCHLIST:
        eod          = eod_map.get(ticker)
        eod_hit      = eod is not None and abs(eod["pct_change"]) >= EOD_THRESHOLD

        # Collect pre-market movers for this ETF
        pm_movers: list[dict] = []
        for c in FALLBACK_HOLDINGS.get(ticker, []):
            pm = constituent_pm.get(c)
            if pm and abs(pm["pct_change"]) >= PREMARKET_THRESHOLD:
                pm["gics"] = GICS_MAP.get(c, "Unknown")
                pm_movers.append(pm)

        pm_movers.sort(key=lambda x: abs(x["pct_change"]), reverse=True)
        pm_movers  = pm_movers[:TOP_N_CONSTITUENTS]
        pm_hit     = len(pm_movers) > 0

        etf_pm = get_premarket_change(ticker)   # ETF-level pre-market for header display

        if not eod_hit and not pm_hit:
            quiet_etfs.append({"ticker": ticker, "eod": eod, "premarket": etf_pm})
            continue

        # Build triggered_by label
        reasons = []
        if pm_hit:
            reasons.append("Pre-market constituent")
        if eod_hit:
            reasons.append("EOD >=2%")
        triggered_by = " + ".join(reasons)

        # Enrich each pre-market mover with company name, news, and catalyst
        for mover in pm_movers:
            mover["company"]      = get_company_name(mover["ticker"])
            headlines             = fetch_all_news(mover["ticker"], mover["company"])
            mover["top_headline"] = headlines[0] if headlines else ""

            # Detect stock splits before calling Claude — avoids misclassifying
            # a routine split as a large loss.
            split_ratio = None
            if abs(mover["pct_change"]) >= SPLIT_THRESHOLD:
                split_ratio = detect_split(mover["ticker"], now_utc)

            if split_ratio:
                ratio_str          = (
                    f"{int(split_ratio)}:1"
                    if split_ratio == int(split_ratio) else f"{split_ratio}:1"
                )
                mover["split_ratio"] = split_ratio
                mover["catalyst"]    = (
                    f"[Stock Split {ratio_str}] Price reflects {ratio_str} share split; "
                    f"not a fundamental move."
                )
            else:
                mover["catalyst"] = classify_catalyst(
                    mover["ticker"],
                    mover["pct_change"],
                    headlines,
                    is_premarket=True,
                )

        pm_movers = flag_sympathy(pm_movers)

        log.info(
            f"[TRIGGERED] {ticker} | {triggered_by} "
            f"| {len(pm_movers)} constituent mover(s)"
        )
        triggered_etfs.append({
            "ticker":       ticker,
            "eod":          eod,
            "premarket":    etf_pm,
            "movers":       pm_movers,
            "triggered_by": triggered_by,
        })

    # ------------------------------------------------------------------
    # Step 4: Build and deliver email
    # ------------------------------------------------------------------
    n       = len(triggered_etfs)
    subject = (
        f"Morning Note: {n} Holding{'s' if n != 1 else ''} Triggered "
        f"| {now_utc.strftime('%b %d, %Y')}"
    )
    html = build_email(triggered_etfs, quiet_etfs, report_date, run_time_utc)
    send_email(subject, html)
    log.info("Morning note complete.")


if __name__ == "__main__":
    main()
