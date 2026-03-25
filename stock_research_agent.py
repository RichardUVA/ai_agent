from __future__ import annotations

import html
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import requests

import stock_agent
from stock_agent import ZoneInfo


REPORT_DIR = Path("research_reports")


def build_research_universe(
    config: stock_agent.Config, performance_table, buy_candidates
) -> list[str]:
    base = list(config.digest_tickers)
    candidate_symbols = buy_candidates["Symbol"].head(3).tolist()
    ordered = []
    for ticker in base + candidate_symbols:
        if ticker not in ordered:
            ordered.append(ticker)
    return ordered


def build_research_input(
    generated_at: datetime,
    config: stock_agent.Config,
    research_assets,
    research_news: dict[str, list[dict[str, str]]],
    buy_candidates,
    market_news: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "generated_at": generated_at.isoformat(),
        "timezone": config.timezone,
        "research_assets": research_assets.to_dict(orient="records"),
        "research_news": research_news,
        "buy_candidates": buy_candidates.to_dict(orient="records"),
        "market_news": market_news,
    }


def generate_research_html(config: stock_agent.Config, research_input: dict[str, Any]) -> str:
    if config.llm_provider == "github_models":
        return generate_github_models_research_html(config, research_input)
    if config.llm_provider == "ollama":
        return generate_ollama_research_html(config, research_input)
    return build_fallback_research_html(research_input)


def generate_github_models_research_html(
    config: stock_agent.Config, research_input: dict[str, Any]
) -> str:
    if not config.github_models_token:
        raise ValueError("Missing GITHUB_TOKEN or GITHUB_MODELS_TOKEN for GitHub Models.")

    prompt = (
        "You are a stock research assistant writing a deeper end-of-day research memo. "
        "Use only the supplied data. Do not invent valuation metrics, earnings dates, or facts. "
        "Return HTML only, as a fragment for an email body. "
        "Structure it with these sections: "
        "Executive Summary, Market Context, Ticker Deep Dives, Buy Candidates, What To Watch Next. "
        "For each ticker, include: what happened, bull case, bear case, key risks, and what to watch next. "
        "Keep the tone analytical and concise.\n\n"
        f"Input data:\n{json.dumps(research_input, indent=2)}"
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
                {"role": "system", "content": "You write concise HTML stock research memos."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        },
        timeout=240,
    )
    response.raise_for_status()
    payload = response.json()
    return stock_agent.strip_code_fences(payload["choices"][0]["message"]["content"])


def generate_ollama_research_html(
    config: stock_agent.Config, research_input: dict[str, Any]
) -> str:
    prompt = (
        "You are a stock research assistant writing a deeper end-of-day research memo. "
        "Use only the supplied data. Do not invent valuation metrics, earnings dates, or facts. "
        "Return HTML only, as a fragment for an email body. "
        "Structure it with these sections: "
        "Executive Summary, Market Context, Ticker Deep Dives, Buy Candidates, What To Watch Next. "
        "For each ticker, include: what happened, bull case, bear case, key risks, and what to watch next. "
        "Keep the tone analytical and concise.\n\n"
        f"Input data:\n{json.dumps(research_input, indent=2)}"
    )
    response = requests.post(
        config.ollama_url,
        json={
            "model": config.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2},
        },
        timeout=240,
    )
    response.raise_for_status()
    payload = response.json()
    return stock_agent.strip_code_fences(payload.get("response", ""))


def build_fallback_research_html(research_input: dict[str, Any]) -> str:
    sections = []
    for asset in research_input["research_assets"]:
        headlines = research_input["research_news"].get(asset["Symbol"], [])
        top_headline = headlines[0]["title"] if headlines else "No recent headline found."
        sections.append(
            "<div style='margin-top:18px; padding:18px; border:1px solid #eaecf0; border-radius:18px;'>"
            f"<h3 style='margin:0 0 8px;'>{html.escape(asset['Symbol'])} - {html.escape(asset['Name'])}</h3>"
            f"<p><strong>Price action:</strong> Last close {stock_agent.format_price(asset['LatestClose'])}, "
            f"1-day {stock_agent.format_pct(asset['DailyReturn'])}, "
            f"3-month {stock_agent.format_pct(asset['ThreeMonthReturn'])}.</p>"
            f"<p><strong>Headline:</strong> {html.escape(top_headline)}</p>"
            "<p><strong>What to watch:</strong> Follow whether the recent price move holds and whether headline flow stays supportive.</p>"
            "</div>"
        )

    return (
        "<h1>Daily Stock Research Memo</h1>"
        "<p>This is a fallback research memo generated without the LLM path.</p>"
        "<h2>Ticker Deep Dives</h2>"
        f"{''.join(sections)}"
    )


def wrap_research_email(generated_at: datetime, html_fragment: str) -> str:
    timestamp = generated_at.strftime("%Y-%m-%d %I:%M %p %Z")
    return f"""
    <html>
      <body style="margin:0; padding:24px 0; background:#f5f7fb; font-family:-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; color:#111827; line-height:1.6;">
        <div style="max-width:820px; margin:0 auto; padding:0 16px;">
          <div style="background:linear-gradient(135deg, #111827 0%, #1d4ed8 100%); border-radius:24px; padding:28px 24px; color:#ffffff;">
            <div style="font-size:12px; text-transform:uppercase; letter-spacing:0.12em; opacity:0.8;">Daily Stock Research</div>
            <h1 style="margin:10px 0;">Deeper research memo</h1>
            <p style="margin:0; font-size:14px; opacity:0.9;">Generated at {html.escape(timestamp)}</p>
          </div>
          <div style="margin-top:18px; background:#ffffff; border:1px solid #eaecf0; border-radius:22px; padding:24px;">
            {html_fragment}
          </div>
        </div>
      </body>
    </html>
    """.strip()


def save_report(generated_at: datetime, report_html: str) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    filename = generated_at.strftime("%Y-%m-%d-stock-research.html")
    report_path = REPORT_DIR / filename
    report_path.write_text(report_html)
    return report_path


def main() -> None:
    config = stock_agent.load_config()
    now = datetime.now(ZoneInfo(config.timezone))
    start = now - timedelta(days=config.lookback_days)
    end = now + timedelta(days=1)

    sp500_table = stock_agent.get_sp500_table()
    sp500_tickers = sp500_table["Symbol"].tolist()
    sp500_raw = stock_agent.download_price_history(sp500_tickers, start=start, end=end)
    sp500_close = stock_agent.extract_close_frame(sp500_raw, tickers=sp500_tickers)
    performance_table = stock_agent.build_performance_table(sp500_table, sp500_close)
    buy_candidates = stock_agent.select_buy_candidates(
        performance_table, candidate_count=config.buy_candidate_count
    )

    research_universe = build_research_universe(config, performance_table, buy_candidates)
    research_assets = stock_agent.build_focus_asset_table(research_universe, start, end)
    research_names = {
        row["Symbol"]: row["Name"] for row in research_assets.to_dict(orient="records")
    }
    research_news = stock_agent.build_ticker_news(research_names, 3)
    market_news = stock_agent.build_general_news(6)

    research_input = build_research_input(
        generated_at=now,
        config=config,
        research_assets=research_assets,
        research_news=research_news,
        buy_candidates=buy_candidates,
        market_news=market_news,
    )
    research_html = generate_research_html(config, research_input)
    email_html = wrap_research_email(now, research_html)
    report_path = save_report(now, email_html)

    subject = f"Daily Stock Research - {now.strftime('%Y-%m-%d')}"
    stock_agent.send_email(config, subject, email_html)
    print(f"Research report sent to {', '.join(config.email_to)}")
    print(f"Saved research report to {report_path}")


if __name__ == "__main__":
    main()
