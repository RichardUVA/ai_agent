from __future__ import annotations

import html
import io
import json
import os
import smtplib
import ssl
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from zoneinfo import ZoneInfo


WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
GOOGLE_NEWS_RSS = (
    "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
)
USER_AGENT = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
    )
}
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DIGEST_NAMES = {
    "VOO": "Vanguard S&P 500 ETF",
    "QQQM": "Invesco NASDAQ 100 ETF",
    "VGT": "Vanguard Information Technology ETF",
    "KLAC": "KLA Corporation",
    "NVDA": "NVIDIA Corporation",
    "JPM": "JPMorgan Chase & Co.",
}
OLLAMA_API_URL = "http://127.0.0.1:11434/api/generate"
GITHUB_MODELS_API_URL = "https://models.github.ai/inference/chat/completions"


@dataclass
class Config:
    smtp_host: str
    smtp_port: int
    smtp_username: str
    smtp_password: str
    email_from: str
    email_to: list[str]
    lookback_days: int
    timezone: str
    market_news_count: int
    digest_total_news_count: int
    digest_tickers: list[str]
    llm_provider: str
    ollama_url: str
    ollama_model: str
    github_models_url: str
    github_models_model: str
    github_models_token: str | None
    buy_candidate_count: int


def load_config() -> Config:
    load_dotenv(BASE_DIR / ".env")
    return Config(
        smtp_host=require_env("SMTP_HOST"),
        smtp_port=int(os.getenv("SMTP_PORT", "465")),
        smtp_username=require_env("SMTP_USERNAME"),
        smtp_password=require_env("SMTP_PASSWORD"),
        email_from=require_env("EMAIL_FROM"),
        email_to=parse_email_list_env("EMAIL_TO") or [require_env("EMAIL_TO")],
        lookback_days=int(os.getenv("LOOKBACK_DAYS", "90")),
        timezone=os.getenv("TIMEZONE", "America/Detroit"),
        market_news_count=int(os.getenv("MARKET_NEWS_COUNT", "6")),
        digest_total_news_count=int(os.getenv("DIGEST_TOTAL_NEWS_COUNT", "3")),
        digest_tickers=parse_csv_env(
            "DIGEST_TICKERS", default=["VOO", "QQQM", "VGT", "KLAC", "NVDA", "JPM"]
        ),
        llm_provider=os.getenv("LLM_PROVIDER", "ollama").strip().lower(),
        ollama_url=os.getenv("OLLAMA_URL", OLLAMA_API_URL),
        ollama_model=os.getenv("OLLAMA_MODEL", "llama3:latest"),
        github_models_url=os.getenv("GITHUB_MODELS_URL", GITHUB_MODELS_API_URL),
        github_models_model=os.getenv("GITHUB_MODELS_MODEL", "openai/gpt-4.1"),
        github_models_token=os.getenv("GITHUB_MODELS_TOKEN") or os.getenv("GITHUB_TOKEN"),
        buy_candidate_count=int(os.getenv("BUY_CANDIDATE_COUNT", "5")),
    )


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def parse_csv_env(name: str, default: list[str] | None = None) -> list[str]:
    raw_value = os.getenv(name, "")
    if not raw_value.strip():
        return default or []
    return [item.strip().upper() for item in raw_value.split(",") if item.strip()]


def parse_email_list_env(name: str) -> list[str]:
    raw_value = os.getenv(name, "")
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def get_sp500_table() -> pd.DataFrame:
    response = requests.get(WIKI_URL, headers=USER_AGENT, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table", {"id": "constituents"})
    if table is None:
        raise RuntimeError("Could not find the S&P 500 constituents table on Wikipedia.")

    rows: list[dict[str, str]] = []
    body = table.find("tbody")
    if body is None:
        raise RuntimeError("The S&P 500 constituents table is missing a table body.")

    for tr in body.find_all("tr"):
        cells = tr.find_all("td")
        if len(cells) < 4:
            continue
        symbol = clean_text(cells[0].get_text(" ", strip=True)).replace(".", "-")
        security = clean_text(cells[1].get_text(" ", strip=True))
        sector = clean_text(cells[3].get_text(" ", strip=True))
        if symbol and security and sector:
            rows.append(
                {
                    "Symbol": symbol,
                    "Security": security,
                    "GICS Sector": sector,
                }
            )

    if not rows:
        raise RuntimeError("No S&P 500 rows were parsed from Wikipedia.")

    return pd.DataFrame(rows)


def download_price_history(
    tickers: list[str], start: datetime, end: datetime
) -> pd.DataFrame:
    data = yf.download(
        tickers=tickers,
        start=start.date(),
        end=end.date(),
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    if data.empty:
        raise RuntimeError("No price history returned from Yahoo Finance.")
    return data


def extract_close_frame(raw_data: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if isinstance(raw_data.columns, pd.MultiIndex):
        close_data = raw_data.xs("Close", axis=1, level=1)
    else:
        close_data = raw_data[["Close"]].rename(columns={"Close": tickers[0]})

    if isinstance(close_data, pd.Series):
        close_data = close_data.to_frame(name=tickers[0])

    return close_data.dropna(axis=1, how="all").sort_index()


def build_performance_table(
    symbol_table: pd.DataFrame, close_frame: pd.DataFrame
) -> pd.DataFrame:
    filled = close_frame.ffill().bfill()
    start_prices = filled.iloc[0]
    end_prices = filled.iloc[-1]
    daily_returns = filled.pct_change().iloc[-1]
    perf_3m = (end_prices / start_prices) - 1

    performance = pd.DataFrame(
        {
            "Symbol": perf_3m.index,
            "ThreeMonthReturn": perf_3m.values,
            "DailyReturn": daily_returns.values,
            "LatestClose": end_prices.values,
        }
    )

    performance = performance.merge(symbol_table, on="Symbol", how="left")
    performance = performance.dropna(
        subset=["ThreeMonthReturn", "DailyReturn", "LatestClose"]
    )
    return performance.reset_index(drop=True)


def fetch_news(query: str, limit: int) -> list[dict[str, str]]:
    try:
        rss_url = GOOGLE_NEWS_RSS.format(query=quote_plus(query))
        response = requests.get(rss_url, headers=USER_AGENT, timeout=30)
        response.raise_for_status()

        root = ET.fromstring(response.content)
        items: list[dict[str, str]] = []
        for item in root.findall("./channel/item"):
            title = clean_text(item.findtext("title", default=""))
            link = item.findtext("link", default="")
            pub_date = item.findtext("pubDate", default="")
            if not title or not link:
                continue
            items.append({"title": title, "link": link, "pub_date": pub_date})
            if len(items) >= limit:
                break
        return items
    except (requests.RequestException, ET.ParseError):
        return []


def clean_text(value: str) -> str:
    return " ".join(value.split())


def build_general_news(limit: int) -> list[dict[str, str]]:
    query = 'stock market OR "S&P 500" OR "Nasdaq" OR "Federal Reserve" when:1d'
    return fetch_news(query=query, limit=limit)


def build_ticker_news(
    tickers_with_names: dict[str, str], per_ticker_limit: int
) -> dict[str, list[dict[str, str]]]:
    results: dict[str, list[dict[str, str]]] = {}
    for ticker, name in tickers_with_names.items():
        query = f'"{name}" OR {ticker} stock when:1d'
        results[ticker] = fetch_news(query=query, limit=per_ticker_limit)
    return results


def build_focus_news(
    tickers_with_names: dict[str, str], total_limit: int
) -> dict[str, list[dict[str, str]]]:
    results = {ticker: [] for ticker in tickers_with_names}
    if total_limit <= 0:
        return results

    for ticker, name in tickers_with_names.items():
        if total_limit <= 0:
            break
        query = f'"{name}" OR {ticker} stock when:1d'
        items = fetch_news(query=query, limit=1)
        if items:
            results[ticker] = items
            total_limit -= 1
    return results


def format_pct(value: float) -> str:
    return f"{value:.2%}"


def format_price(value: float) -> str:
    return f"${value:,.2f}"


def render_return_badge(value: float) -> str:
    bg_color = "#ecfdf3" if value >= 0 else "#fff1f2"
    text_color = "#166534" if value >= 0 else "#b42318"
    return (
        f"<span style='display:inline-block; padding:4px 8px; border-radius:999px; "
        f"background:{bg_color}; color:{text_color}; font-weight:600;'>"
        f"{html.escape(format_pct(value))}</span>"
    )


def build_focus_asset_table(
    digest_tickers: list[str], start: datetime, end: datetime
) -> pd.DataFrame:
    raw_data = download_price_history(digest_tickers, start=start, end=end)
    close_frame = extract_close_frame(raw_data, tickers=digest_tickers)
    filled = close_frame.ffill().bfill()

    rows = []
    for ticker in digest_tickers:
        if ticker not in filled.columns:
            continue
        series = filled[ticker].dropna()
        if series.empty:
            continue
        rows.append(
            {
                "Symbol": ticker,
                "Name": DEFAULT_DIGEST_NAMES.get(ticker, ticker),
                "LatestClose": float(series.iloc[-1]),
                "DailyReturn": float(series.pct_change().iloc[-1]),
                "ThreeMonthReturn": float((series.iloc[-1] / series.iloc[0]) - 1),
            }
        )

    return pd.DataFrame(rows)


def get_daily_movers(performance_table: pd.DataFrame) -> dict[str, dict[str, Any]]:
    top_row = performance_table.sort_values("DailyReturn", ascending=False).iloc[0]
    bottom_row = performance_table.sort_values("DailyReturn", ascending=True).iloc[0]
    return {
        "top": top_row.to_dict(),
        "bottom": bottom_row.to_dict(),
    }


def select_buy_candidates(
    performance_table: pd.DataFrame, candidate_count: int
) -> pd.DataFrame:
    negative_threshold = performance_table["ThreeMonthReturn"].quantile(0.25)
    candidates = performance_table[
        (performance_table["ThreeMonthReturn"] <= negative_threshold)
        & (performance_table["DailyReturn"] > 0)
    ].copy()

    if candidates.empty:
        candidates = performance_table.nsmallest(candidate_count, "ThreeMonthReturn").copy()
        candidates["Reason"] = "Deep 3-month underperformance candidate"
    else:
        candidates = candidates.sort_values(
            ["DailyReturn", "ThreeMonthReturn"],
            ascending=[False, True],
        ).head(candidate_count)
        candidates["Reason"] = "Oversold with positive 1-day reversal"

    return candidates.reset_index(drop=True)


def build_llm_input(
    generated_at: datetime,
    general_news: list[dict[str, str]],
    focus_assets: pd.DataFrame,
    focus_news: dict[str, list[dict[str, str]]],
    movers: dict[str, dict[str, Any]],
    mover_news: dict[str, list[dict[str, str]]],
    buy_candidates: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "generated_at": generated_at.isoformat(),
        "general_news": general_news,
        "focus_assets": focus_assets.to_dict(orient="records"),
        "focus_news": focus_news,
        "top_mover": movers["top"],
        "bottom_mover": movers["bottom"],
        "mover_news": mover_news,
        "buy_candidates": buy_candidates.to_dict(orient="records"),
    }


def strip_code_fences(value: str) -> str:
    cleaned = value.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


def build_brief_schema(digest_tickers: list[str]) -> dict[str, Any]:
    digest_properties = {
        ticker: {"type": "string"} for ticker in digest_tickers
    }
    return {
        "type": "object",
        "properties": {
            "market_news_briefs": {
                "type": "array",
                "minItems": 3,
                "maxItems": 3,
                "items": {
                    "type": "object",
                    "properties": {
                        "headline": {"type": "string"},
                        "takeaway": {"type": "string"},
                    },
                    "required": ["headline", "takeaway"],
                },
            },
            "digests": {
                "type": "object",
                "properties": digest_properties,
                "required": list(digest_properties.keys()),
            },
            "movers": {
                "type": "object",
                "properties": {
                    "top": {"type": "string"},
                    "bottom": {"type": "string"},
                },
                "required": ["top", "bottom"],
            },
            "buy_candidates_summary": {"type": "string"},
        },
        "required": ["market_news_briefs", "digests", "movers", "buy_candidates_summary"],
    }


def normalize_brief_output(brief: dict[str, Any], digest_tickers: list[str]) -> dict[str, Any]:
    normalized_news = []
    for item in brief.get("market_news_briefs", [])[:3]:
        if not isinstance(item, dict):
            item = {"headline": str(item), "takeaway": ""}
        headline = clean_text(str(item.get("headline", "") or "Market update"))
        takeaway = clean_text(
            str(
                item.get("takeaway")
                or item.get("summary")
                or item.get("detail")
                or "No additional takeaway provided."
            )
        )
        normalized_news.append({"headline": headline, "takeaway": takeaway})

    while len(normalized_news) < 3:
        normalized_news.append(
            {
                "headline": "No additional market headline available",
                "takeaway": "No additional takeaway provided.",
            }
        )

    digests = brief.get("digests", {})
    if not isinstance(digests, dict):
        digests = {}
    normalized_digests = {
        ticker: clean_text(str(digests.get(ticker, "No digest available.")))
        for ticker in digest_tickers
    }

    movers = brief.get("movers", {})
    if not isinstance(movers, dict):
        movers = {}
    normalized_movers = {
        "top": clean_text(str(movers.get("top", "No top mover summary available."))),
        "bottom": clean_text(
            str(movers.get("bottom", "No bottom mover summary available."))
        ),
    }

    buy_candidates_summary = clean_text(
        str(
            brief.get("buy_candidates_summary")
            or "These are heuristic buy candidates based on recent S&P 500 price action."
        )
    )

    return {
        "market_news_briefs": normalized_news,
        "digests": normalized_digests,
        "movers": normalized_movers,
        "buy_candidates_summary": buy_candidates_summary,
    }


def generate_ollama_brief(config: Config, llm_input: dict[str, Any]) -> dict[str, Any] | None:
    if config.llm_provider != "ollama":
        return None

    prompt = (
        "You write a concise U.S. morning market email. "
        "Use only the supplied data. Do not invent facts. "
        "Return valid JSON that matches the provided schema exactly. "
        "market_news_briefs must contain exactly 3 items. "
        "Each digest should be 1-2 concise sentences. "
        "The mover summaries should each be 2 concise sentences.\n\n"
        f"Input data:\n{json.dumps(llm_input, indent=2)}"
    )

    response = requests.post(
        config.ollama_url,
        json={
            "model": config.ollama_model,
            "prompt": prompt,
            "stream": False,
            "format": build_brief_schema(config.digest_tickers),
            "options": {"temperature": 0.2},
        },
        timeout=180,
    )
    response.raise_for_status()
    payload = response.json()
    raw_output = strip_code_fences(payload.get("response", ""))
    parsed = json.loads(raw_output)

    return normalize_brief_output(parsed, config.digest_tickers)


def generate_github_models_brief(
    config: Config, llm_input: dict[str, Any]
) -> dict[str, Any] | None:
    if config.llm_provider != "github_models":
        return None
    if not config.github_models_token:
        raise ValueError("Missing GITHUB_TOKEN or GITHUB_MODELS_TOKEN for GitHub Models.")

    schema = build_brief_schema(config.digest_tickers)
    prompt = (
        "You write a concise U.S. morning market email. "
        "Use only the supplied data. Do not invent facts. "
        "Return valid JSON that matches the requested schema exactly. "
        "market_news_briefs must contain exactly 3 items. "
        "Each digest should be 1-2 concise sentences. "
        "The mover summaries should each be 2 concise sentences.\n\n"
        f"Input data:\n{json.dumps(llm_input, indent=2)}"
    )

    response = requests.post(
        config.github_models_url,
        headers={
            "Authorization": f"Bearer {config.github_models_token}",
            "Content-Type": "application/json",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        json={
            "model": config.github_models_model,
            "messages": [
                {"role": "system", "content": "You are a concise financial email assistant."},
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
        },
        timeout=180,
    )
    response.raise_for_status()
    payload = response.json()
    raw_output = payload["choices"][0]["message"]["content"]
    parsed = json.loads(strip_code_fences(raw_output))

    return normalize_brief_output(parsed, config.digest_tickers)


def build_fallback_brief(
    general_news: list[dict[str, str]],
    focus_assets: pd.DataFrame,
    movers: dict[str, dict[str, Any]],
    focus_news: dict[str, list[dict[str, str]]],
    mover_news: dict[str, list[dict[str, str]]],
    buy_candidates: pd.DataFrame,
) -> dict[str, Any]:
    market_news_briefs = []
    for item in general_news[:3]:
        market_news_briefs.append(
            {
                "headline": item["title"],
                "takeaway": "Included directly from today's market-news feed.",
            }
        )

    while len(market_news_briefs) < 3:
        market_news_briefs.append(
            {
                "headline": "No additional market headline available",
                "takeaway": "The news feed returned fewer than three usable items.",
            }
        )

    digests: dict[str, str] = {}
    for row in focus_assets.to_dict(orient="records"):
        top_headline = "No recent headline found."
        if focus_news.get(row["Symbol"]):
            top_headline = focus_news[row["Symbol"]][0]["title"]
        digests[row["Symbol"]] = (
            f'{row["Name"]} closed at {format_price(row["LatestClose"])}, '
            f'up {format_pct(row["DailyReturn"])} on the day and '
            f'{format_pct(row["ThreeMonthReturn"])} over 3 months. '
            f"Headline: {top_headline}"
        )

    top_news = mover_news.get(movers["top"]["Symbol"], [])
    bottom_news = mover_news.get(movers["bottom"]["Symbol"], [])
    top_headline = top_news[0]["title"] if top_news else "No recent headline found."
    bottom_headline = (
        bottom_news[0]["title"] if bottom_news else "No recent headline found."
    )

    return {
        "market_news_briefs": market_news_briefs,
        "digests": digests,
        "movers": {
            "top": (
                f'{movers["top"]["Security"]} ({movers["top"]["Symbol"]}) led the S&P 500 '
                f'with a daily move of {format_pct(float(movers["top"]["DailyReturn"]))}. '
                f"Latest headline: {top_headline}"
            ),
            "bottom": (
                f'{movers["bottom"]["Security"]} ({movers["bottom"]["Symbol"]}) was the weakest '
                f'S&P 500 stock on the day at {format_pct(float(movers["bottom"]["DailyReturn"]))}. '
                f"Latest headline: {bottom_headline}"
            ),
        },
        "buy_candidates_summary": (
            "These are heuristic buy candidates chosen from S&P 500 names with weak "
            "3-month performance and either a positive 1-day reversal or deep recent weakness."
        ),
    }


def render_news_list(items: list[dict[str, str]]) -> str:
    if not items:
        return (
            "<li style='margin:0 0 10px; color:#667085;'>"
            "No recent news found."
            "</li>"
        )
    return "".join(
        (
            "<li style='margin:0 0 12px;'>"
            f"<a href='{html.escape(item['link'])}' "
            "style='color:#0f172a; text-decoration:none; font-weight:600;'>"
            f"{html.escape(item['title'])}</a><br/>"
            f"<span style='color:#667085; font-size:12px;'>{html.escape(item['pub_date'])}</span>"
            "</li>"
        )
        for item in items
    )


def render_html_report(
    generated_at: datetime,
    focus_assets: pd.DataFrame,
    general_news: list[dict[str, str]],
    focus_news: dict[str, list[dict[str, str]]],
    movers: dict[str, dict[str, Any]],
    mover_news: dict[str, list[dict[str, str]]],
    buy_candidates: pd.DataFrame,
    brief: dict[str, Any],
    llm_enabled: bool,
    llm_status: str,
) -> str:
    market_briefs_html = "".join(
        (
            "<li style='margin:0 0 14px; padding:14px 16px; border:1px solid #eaecf0; "
            "border-radius:14px; background:#ffffff;'>"
            f"<div style='font-size:15px; font-weight:700; color:#0f172a; margin-bottom:6px;'>{html.escape(item['headline'])}</div>"
            f"<div style='font-size:14px; color:#475467;'>{html.escape(item['takeaway'])}</div>"
            "</li>"
        )
        for item in brief["market_news_briefs"]
    )

    digest_sections = []
    for row in focus_assets.to_dict(orient="records"):
        headline_html = ""
        if focus_news.get(row["Symbol"]):
            headline_html = (
                "<div style='margin-top:12px; padding-top:12px; border-top:1px solid #eaecf0;'>"
                "<div style='font-size:12px; text-transform:uppercase; letter-spacing:0.05em; color:#667085; margin-bottom:6px;'>Headline</div>"
                f"<div style='font-size:14px; color:#344054;'>{html.escape(focus_news[row['Symbol']][0]['title'])}</div>"
                "</div>"
            )
        digest_sections.append(
            "<div style='margin-top:18px; padding:18px; border:1px solid #eaecf0; "
            "border-radius:18px; background:#ffffff;'>"
            f"<div style='font-size:16px; font-weight:700; color:#0f172a; margin:0 0 4px;'>{html.escape(row['Symbol'])} - {html.escape(row['Name'])}</div>"
            f"<div style='margin:0 0 12px; color:#475467; font-size:14px;'>{html.escape(brief['digests'].get(row['Symbol'], 'No digest available.'))}</div>"
            "<table role='presentation' cellspacing='0' cellpadding='0' style='border-collapse:collapse; width:100%;'>"
            "<tr>"
            "<td style='padding:0 8px 8px 0;'>"
            "<div style='display:inline-block; padding:8px 10px; border-radius:12px; background:#f8fafc;'>"
            "<div style='font-size:11px; color:#667085; text-transform:uppercase; letter-spacing:0.05em;'>Last close</div>"
            f"<div style='font-size:15px; font-weight:700; color:#101828;'>{format_price(row['LatestClose'])}</div>"
            "</div>"
            "</td>"
            "<td style='padding:0 8px 8px 0;'>"
            "<div style='display:inline-block; padding:8px 10px; border-radius:12px; background:#f8fafc;'>"
            "<div style='font-size:11px; color:#667085; text-transform:uppercase; letter-spacing:0.05em;'>1-day</div>"
            f"<div style='margin-top:4px;'>{render_return_badge(row['DailyReturn'])}</div>"
            "</div>"
            "</td>"
            "<td style='padding:0 0 8px 0;'>"
            "<div style='display:inline-block; padding:8px 10px; border-radius:12px; background:#f8fafc;'>"
            "<div style='font-size:11px; color:#667085; text-transform:uppercase; letter-spacing:0.05em;'>3-month</div>"
            f"<div style='margin-top:4px;'>{render_return_badge(row['ThreeMonthReturn'])}</div>"
            "</div>"
            "</td>"
            "</tr>"
            "</table>"
            f"{headline_html}"
            "</div>"
        )

    timestamp = generated_at.strftime("%Y-%m-%d %I:%M %p %Z")
    llm_label = (
        "AI-generated summary"
        if llm_enabled
        else f"Fallback summary used: {llm_status}"
    )
    top_mover = movers["top"]
    bottom_mover = movers["bottom"]
    buy_cards = []
    for row in buy_candidates.to_dict(orient="records"):
        buy_cards.append(
            (
                "<div style='margin-top:14px; padding:16px; border:1px solid #eaecf0; border-radius:18px; background:#fcfcfd;'>"
                f"<div style='font-size:17px; font-weight:800; color:#101828;'>{html.escape(row['Symbol'])} <span style='font-size:14px; font-weight:600; color:#475467;'>{html.escape(row['Security'])}</span></div>"
                f"<div style='font-size:13px; color:#667085; margin-top:4px;'>{html.escape(row['GICS Sector'])}</div>"
                "<div style='margin-top:12px;'>"
                f"<span style='display:inline-block; margin:0 8px 8px 0; padding:8px 10px; border-radius:12px; background:#f8fafc; font-weight:700; color:#101828;'>{format_price(float(row['LatestClose']))}</span>"
                f"<span style='display:inline-block; margin:0 8px 8px 0;'>{render_return_badge(float(row['DailyReturn']))}</span>"
                f"<span style='display:inline-block; margin:0 8px 8px 0;'>{render_return_badge(float(row['ThreeMonthReturn']))}</span>"
                "</div>"
                "<div style='margin-top:10px; font-size:12px; text-transform:uppercase; letter-spacing:0.05em; color:#667085;'>Reason</div>"
                f"<div style='margin-top:4px; font-size:14px; color:#344054; line-height:1.55;'>{html.escape(row['Reason'])}</div>"
                "</div>"
            )
        )

    summary_cards = (
        "<tr>"
        "<td style='padding:0 8px 0 0; width:50%; vertical-align:top;'>"
        "<div style='background:#ffffff; border:1px solid #dbe4ff; border-radius:18px; padding:18px;'>"
        "<div style='font-size:12px; text-transform:uppercase; letter-spacing:0.08em; color:#667085; margin-bottom:8px;'>Top mover</div>"
        f"<div style='font-size:24px; font-weight:800; color:#101828;'>{html.escape(top_mover['Symbol'])}</div>"
        f"<div style='font-size:14px; color:#475467; margin:4px 0 10px;'>{html.escape(top_mover['Security'])}</div>"
        f"{render_return_badge(float(top_mover['DailyReturn']))}"
        "</div>"
        "</td>"
        "<td style='padding:0 0 0 8px; width:50%; vertical-align:top;'>"
        "<div style='background:#ffffff; border:1px solid #f2d6db; border-radius:18px; padding:18px;'>"
        "<div style='font-size:12px; text-transform:uppercase; letter-spacing:0.08em; color:#667085; margin-bottom:8px;'>Bottom mover</div>"
        f"<div style='font-size:24px; font-weight:800; color:#101828;'>{html.escape(bottom_mover['Symbol'])}</div>"
        f"<div style='font-size:14px; color:#475467; margin:4px 0 10px;'>{html.escape(bottom_mover['Security'])}</div>"
        f"{render_return_badge(float(bottom_mover['DailyReturn']))}"
        "</div>"
        "</td>"
        "</tr>"
    )

    return f"""
    <html>
      <body style="margin:0; padding:24px 0; background:#f5f7fb; font-family:-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; color:#111827; line-height:1.5;">
        <div style="max-width:760px; margin:0 auto; padding:0 16px;">
          <div style="background:linear-gradient(135deg, #0f172a 0%, #1d4ed8 100%); border-radius:24px; padding:28px 24px; color:#ffffff; box-shadow:0 20px 40px rgba(15, 23, 42, 0.16);">
            <div style="font-size:12px; text-transform:uppercase; letter-spacing:0.12em; opacity:0.8;">Daily Market Digest</div>
            <h1 style="margin:10px 0 10px; font-size:30px; line-height:1.15;">Daily market digest for your core watchlist</h1>
            <p style="margin:0; font-size:14px; opacity:0.92;">Generated at {html.escape(timestamp)}</p>
            <div style="margin-top:12px; display:inline-block; padding:6px 10px; border-radius:999px; background:rgba(255,255,255,0.14); font-size:13px;">
              {html.escape(llm_label)}
            </div>
          </div>

          <div style="margin-top:18px;">
            <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="border-collapse:collapse;">
              {summary_cards}
            </table>
          </div>

          <div style="margin-top:18px; background:#ffffff; border:1px solid #eaecf0; border-radius:22px; padding:22px;">
            <div style="font-size:12px; text-transform:uppercase; letter-spacing:0.08em; color:#667085; margin-bottom:8px;">Section 1</div>
            <h2 style="margin:0 0 14px; font-size:24px; color:#101828;">Three brief market news items</h2>
            <ul style="margin:0; padding-left:0; list-style:none;">
              {market_briefs_html}
            </ul>
          </div>

          <div style="margin-top:18px; background:#ffffff; border:1px solid #eaecf0; border-radius:22px; padding:22px;">
            <div style="font-size:12px; text-transform:uppercase; letter-spacing:0.08em; color:#667085; margin-bottom:8px;">Section 2</div>
            <h2 style="margin:0 0 14px; font-size:24px; color:#101828;">Daily digest: VOO, QQQM, VGT, KLAC, NVDA, JPM</h2>
            <p style="margin:0 0 10px; color:#667085; font-size:14px;">Each card below shows the last close, 1-day move, and 3-month move for that ticker.</p>
            {''.join(digest_sections)}
          </div>

          <div style="margin-top:18px; background:#ffffff; border:1px solid #eaecf0; border-radius:22px; padding:22px;">
            <div style="font-size:12px; text-transform:uppercase; letter-spacing:0.08em; color:#667085; margin-bottom:8px;">Section 3</div>
            <h2 style="margin:0 0 14px; font-size:24px; color:#101828;">Top and bottom movers in the S&amp;P 500</h2>
            <div style="border:1px solid #eaecf0; border-radius:18px; overflow:hidden;">
              <table width="100%" cellspacing="0" cellpadding="0" style="border-collapse:collapse; background:#ffffff;">
                <thead>
                  <tr style="background:#f8fafc;">
                    <th align="left" style="padding:12px 10px; font-size:12px; text-transform:uppercase; letter-spacing:0.05em; color:#667085;">Type</th>
                    <th align="left" style="padding:12px 10px; font-size:12px; text-transform:uppercase; letter-spacing:0.05em; color:#667085;">Ticker</th>
                    <th align="left" style="padding:12px 10px; font-size:12px; text-transform:uppercase; letter-spacing:0.05em; color:#667085;">Company</th>
                    <th align="left" style="padding:12px 10px; font-size:12px; text-transform:uppercase; letter-spacing:0.05em; color:#667085;">Sector</th>
                    <th align="left" style="padding:12px 10px; font-size:12px; text-transform:uppercase; letter-spacing:0.05em; color:#667085;">Last close</th>
                    <th align="left" style="padding:12px 10px; font-size:12px; text-transform:uppercase; letter-spacing:0.05em; color:#667085;">1-day</th>
                    <th align="left" style="padding:12px 10px; font-size:12px; text-transform:uppercase; letter-spacing:0.05em; color:#667085;">3-month</th>
                  </tr>
                </thead>
                <tbody>
                  <tr style='border-top:1px solid #eaecf0;'>
                    <td style='padding:12px 10px; font-weight:700;'>Top mover</td>
                    <td style='padding:12px 10px; font-weight:700; color:#0f172a;'>{html.escape(top_mover['Symbol'])}</td>
                    <td style='padding:12px 10px; color:#344054;'>{html.escape(top_mover['Security'])}</td>
                    <td style='padding:12px 10px; color:#344054;'>{html.escape(top_mover['GICS Sector'])}</td>
                    <td style='padding:12px 10px; font-weight:600;'>{format_price(float(top_mover['LatestClose']))}</td>
                    <td style='padding:12px 10px;'>{render_return_badge(float(top_mover['DailyReturn']))}</td>
                    <td style='padding:12px 10px;'>{render_return_badge(float(top_mover['ThreeMonthReturn']))}</td>
                  </tr>
                  <tr style='border-top:1px solid #eaecf0;'>
                    <td style='padding:12px 10px; font-weight:700;'>Bottom mover</td>
                    <td style='padding:12px 10px; font-weight:700; color:#0f172a;'>{html.escape(bottom_mover['Symbol'])}</td>
                    <td style='padding:12px 10px; color:#344054;'>{html.escape(bottom_mover['Security'])}</td>
                    <td style='padding:12px 10px; color:#344054;'>{html.escape(bottom_mover['GICS Sector'])}</td>
                    <td style='padding:12px 10px; font-weight:600;'>{format_price(float(bottom_mover['LatestClose']))}</td>
                    <td style='padding:12px 10px;'>{render_return_badge(float(bottom_mover['DailyReturn']))}</td>
                    <td style='padding:12px 10px;'>{render_return_badge(float(bottom_mover['ThreeMonthReturn']))}</td>
                  </tr>
                </tbody>
              </table>
            </div>

            <div style="margin-top:18px; padding:18px; border:1px solid #eaecf0; border-radius:18px; background:#fcfcfd;">
              <h3 style="margin:0 0 8px; font-size:18px; color:#101828;">Top mover summary</h3>
              <p style="margin:0 0 12px; color:#475467;">{html.escape(brief['movers']['top'])}</p>
              <ul style="margin:0; padding-left:20px;">
                {render_news_list(mover_news.get(top_mover['Symbol'], []))}
              </ul>
            </div>

            <div style="margin-top:18px; padding:18px; border:1px solid #eaecf0; border-radius:18px; background:#fcfcfd;">
              <h3 style="margin:0 0 8px; font-size:18px; color:#101828;">Bottom mover summary</h3>
              <p style="margin:0 0 12px; color:#475467;">{html.escape(brief['movers']['bottom'])}</p>
              <ul style="margin:0; padding-left:20px;">
                {render_news_list(mover_news.get(bottom_mover['Symbol'], []))}
              </ul>
            </div>
          </div>

          <div style="margin-top:18px; background:#ffffff; border:1px solid #eaecf0; border-radius:22px; padding:22px;">
            <div style="font-size:12px; text-transform:uppercase; letter-spacing:0.08em; color:#667085; margin-bottom:8px;">Section 4</div>
            <h2 style="margin:0 0 8px; font-size:24px; color:#101828;">S&amp;P 500 buy candidates</h2>
            <p style="margin:0 0 14px; color:#475467;">{html.escape(brief['buy_candidates_summary'])}</p>
            <p style="margin:0 0 14px; color:#667085; font-size:13px;">Heuristic only, not investment advice.</p>
            {''.join(buy_cards)}
          </div>
        </div>
      </body>
    </html>
    """.strip()


def send_email(config: Config, subject: str, html_body: str) -> None:
    message = MIMEMultipart("alternative")
    message["Subject"] = subject
    message["From"] = config.email_from
    message["To"] = ", ".join(config.email_to)
    message.attach(MIMEText(html_body, "html"))

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(config.smtp_host, config.smtp_port, context=context) as server:
        server.login(config.smtp_username, config.smtp_password)
        server.sendmail(config.email_from, config.email_to, message.as_string())


def build_report_payload(config: Config) -> dict[str, Any]:
    now = datetime.now(ZoneInfo(config.timezone))
    start = now - timedelta(days=config.lookback_days)
    end = now + timedelta(days=1)

    sp500_table = get_sp500_table()
    sp500_tickers = sp500_table["Symbol"].tolist()
    sp500_raw = download_price_history(sp500_tickers, start=start, end=end)
    sp500_close = extract_close_frame(sp500_raw, tickers=sp500_tickers)
    performance_table = build_performance_table(sp500_table, sp500_close)
    buy_candidates = select_buy_candidates(
        performance_table, candidate_count=config.buy_candidate_count
    )

    focus_assets = build_focus_asset_table(config.digest_tickers, start=start, end=end)
    focus_names = {
        row["Symbol"]: row["Name"] for row in focus_assets.to_dict(orient="records")
    }
    focus_news = build_focus_news(focus_names, config.digest_total_news_count)

    general_news = build_general_news(config.market_news_count)
    movers = get_daily_movers(performance_table)
    mover_names = {
        movers["top"]["Symbol"]: movers["top"]["Security"],
        movers["bottom"]["Symbol"]: movers["bottom"]["Security"],
    }
    mover_news = build_ticker_news(mover_names, 2)

    llm_input = build_llm_input(
        generated_at=now,
        general_news=general_news,
        focus_assets=focus_assets,
        focus_news=focus_news,
        movers=movers,
        mover_news=mover_news,
        buy_candidates=buy_candidates,
    )

    llm_enabled = config.llm_provider in {"ollama", "github_models"}
    llm_status = "LLM provider not configured"
    try:
        brief = (
            generate_ollama_brief(config, llm_input)
            or generate_github_models_brief(config, llm_input)
            or build_fallback_brief(
                general_news=general_news,
                focus_assets=focus_assets,
                movers=movers,
                focus_news=focus_news,
                mover_news=mover_news,
                buy_candidates=buy_candidates,
            )
        )
        if config.llm_provider == "ollama":
            llm_status = f"Ollama summary generated with {config.ollama_model}"
        elif config.llm_provider == "github_models":
            llm_status = (
                f"GitHub Models summary generated with {config.github_models_model}"
            )
    except Exception as exc:
        llm_enabled = False
        llm_status = clean_text(str(exc))[:160] or "LLM request failed"
        brief = build_fallback_brief(
            general_news=general_news,
            focus_assets=focus_assets,
            movers=movers,
            focus_news=focus_news,
            mover_news=mover_news,
            buy_candidates=buy_candidates,
        )

    return {
        "generated_at": now,
        "focus_assets": focus_assets,
        "general_news": general_news,
        "focus_news": focus_news,
        "movers": movers,
        "mover_news": mover_news,
        "buy_candidates": buy_candidates,
        "brief": brief,
        "llm_enabled": llm_enabled,
        "llm_status": llm_status,
    }


def main() -> None:
    config = load_config()
    payload = build_report_payload(config)
    report_html = render_html_report(**payload)

    subject_date = payload["generated_at"].strftime("%Y-%m-%d")
    subject = f"Daily Market Digest - {subject_date}"
    send_email(config, subject, report_html)
    print(f"Report sent to {', '.join(config.email_to)}")


if __name__ == "__main__":
    main()
