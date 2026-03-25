from __future__ import annotations

import html
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import requests

import stock_agent
from stock_agent import ZoneInfo


REPORT_DIR = Path("research_reports")
MEMORY_DIR = Path("research_memory")
PROGRAM_PATH = Path("program.md")


def load_program() -> str:
    if not PROGRAM_PATH.exists():
        return ""
    return PROGRAM_PATH.read_text().strip()


def load_memory(ticker: str) -> dict[str, Any]:
    path = MEMORY_DIR / f"{ticker}.json"
    if not path.exists():
        return {
            "ticker": ticker,
            "thesis": "",
            "bull_case": "",
            "bear_case": "",
            "risks": [],
            "catalysts": [],
            "open_questions": [],
            "what_changed": "",
            "confidence": "low",
        }
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {
            "ticker": ticker,
            "thesis": "",
            "bull_case": "",
            "bear_case": "",
            "risks": [],
            "catalysts": [],
            "open_questions": [],
            "what_changed": "",
            "confidence": "low",
        }


def save_memory(ticker: str, payload: dict[str, Any]) -> Path:
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    path = MEMORY_DIR / f"{ticker}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def build_research_universe(
    config: stock_agent.Config, buy_candidates
) -> list[str]:
    ordered = []
    for ticker in list(config.digest_tickers) + buy_candidates["Symbol"].head(3).tolist():
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
    program_text: str,
    prior_memory: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "generated_at": generated_at.isoformat(),
        "timezone": config.timezone,
        "program": program_text,
        "research_assets": research_assets.to_dict(orient="records"),
        "research_news": research_news,
        "buy_candidates": buy_candidates.to_dict(orient="records"),
        "market_news": market_news,
        "prior_memory": prior_memory,
    }


def build_memory_update_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "portfolio_summary": {"type": "string"},
            "change_log": {"type": "string"},
            "ticker_updates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string"},
                        "thesis": {"type": "string"},
                        "bull_case": {"type": "string"},
                        "bear_case": {"type": "string"},
                        "risks": {"type": "array", "items": {"type": "string"}},
                        "catalysts": {"type": "array", "items": {"type": "string"}},
                        "open_questions": {"type": "array", "items": {"type": "string"}},
                        "what_changed": {"type": "string"},
                        "confidence": {"type": "string"},
                    },
                    "required": [
                        "ticker",
                        "thesis",
                        "bull_case",
                        "bear_case",
                        "risks",
                        "catalysts",
                        "open_questions",
                        "what_changed",
                        "confidence",
                    ],
                },
            },
        },
        "required": ["portfolio_summary", "change_log", "ticker_updates"],
    }


def normalize_memory_response(
    response: dict[str, Any], research_assets
) -> dict[str, Any]:
    asset_rows = {row["Symbol"]: row for row in research_assets.to_dict(orient="records")}
    normalized_updates = []

    for update in response.get("ticker_updates", []):
        if not isinstance(update, dict):
            continue
        ticker = stock_agent.clean_text(str(update.get("ticker", "")).upper())
        if ticker not in asset_rows:
            continue
        normalized_updates.append(
            {
                "ticker": ticker,
                "name": asset_rows[ticker]["Name"],
                "last_updated": datetime.utcnow().isoformat() + "Z",
                "latest_close": asset_rows[ticker]["LatestClose"],
                "daily_return": asset_rows[ticker]["DailyReturn"],
                "three_month_return": asset_rows[ticker]["ThreeMonthReturn"],
                "thesis": stock_agent.clean_text(str(update.get("thesis", ""))),
                "bull_case": stock_agent.clean_text(str(update.get("bull_case", ""))),
                "bear_case": stock_agent.clean_text(str(update.get("bear_case", ""))),
                "risks": [stock_agent.clean_text(str(x)) for x in update.get("risks", [])[:5]],
                "catalysts": [
                    stock_agent.clean_text(str(x)) for x in update.get("catalysts", [])[:5]
                ],
                "open_questions": [
                    stock_agent.clean_text(str(x))
                    for x in update.get("open_questions", [])[:5]
                ],
                "what_changed": stock_agent.clean_text(
                    str(update.get("what_changed", "No prior change log available."))
                ),
                "confidence": stock_agent.clean_text(
                    str(update.get("confidence", "medium")).lower()
                ),
            }
        )

    return {
        "portfolio_summary": stock_agent.clean_text(
            str(response.get("portfolio_summary", "No portfolio summary available."))
        ),
        "change_log": stock_agent.clean_text(
            str(response.get("change_log", "No change log available."))
        ),
        "ticker_updates": normalized_updates,
    }


def generate_research_updates(
    config: stock_agent.Config, research_input: dict[str, Any], research_assets
) -> dict[str, Any]:
    if config.llm_provider == "github_models":
        return generate_github_models_research_updates(config, research_input, research_assets)
    if config.llm_provider == "ollama":
        return generate_ollama_research_updates(config, research_input, research_assets)
    return build_fallback_research_updates(research_input, research_assets)


def generate_github_models_research_updates(
    config: stock_agent.Config, research_input: dict[str, Any], research_assets
) -> dict[str, Any]:
    if not config.github_models_token:
        raise ValueError("Missing GITHUB_TOKEN or GITHUB_MODELS_TOKEN for GitHub Models.")

    prompt = (
        "You are an autonomous stock research analyst updating persistent ticker memory. "
        "Follow the supplied program.md guidance. "
        "Use only the supplied data. Do not invent valuation metrics, earnings dates, or facts. "
        "Update the thesis for each ticker based on what changed, risks, catalysts, and open questions. "
        "Return valid JSON matching the requested schema exactly.\n\n"
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
                {
                    "role": "system",
                    "content": (
                        "You maintain structured stock research memory and write concise "
                        "institutional-style updates."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
        },
        timeout=240,
    )
    response.raise_for_status()
    payload = response.json()
    parsed = json.loads(
        stock_agent.strip_code_fences(payload["choices"][0]["message"]["content"])
    )
    return normalize_memory_response(parsed, research_assets)


def generate_ollama_research_updates(
    config: stock_agent.Config, research_input: dict[str, Any], research_assets
) -> dict[str, Any]:
    prompt = (
        "You are an autonomous stock research analyst updating persistent ticker memory. "
        "Follow the supplied program.md guidance. "
        "Use only the supplied data. Do not invent valuation metrics, earnings dates, or facts. "
        "Update the thesis for each ticker based on what changed, risks, catalysts, and open questions. "
        "Return valid JSON matching the provided schema exactly.\n\n"
        f"Input data:\n{json.dumps(research_input, indent=2)}"
    )
    response = requests.post(
        config.ollama_url,
        json={
            "model": config.ollama_model,
            "prompt": prompt,
            "stream": False,
            "format": build_memory_update_schema(),
            "options": {"temperature": 0.2},
        },
        timeout=240,
    )
    response.raise_for_status()
    payload = response.json()
    parsed = json.loads(stock_agent.strip_code_fences(payload.get("response", "")))
    return normalize_memory_response(parsed, research_assets)


def build_fallback_research_updates(research_input: dict[str, Any], research_assets) -> dict[str, Any]:
    updates = []
    for asset in research_assets.to_dict(orient="records"):
        headlines = research_input["research_news"].get(asset["Symbol"], [])
        top_headline = headlines[0]["title"] if headlines else "No recent headline found."
        updates.append(
            {
                "ticker": asset["Symbol"],
                "name": asset["Name"],
                "last_updated": datetime.utcnow().isoformat() + "Z",
                "latest_close": asset["LatestClose"],
                "daily_return": asset["DailyReturn"],
                "three_month_return": asset["ThreeMonthReturn"],
                "thesis": (
                    f"{asset['Name']} remains on the research list because price action and "
                    "headline flow may create a new entry or exit setup."
                ),
                "bull_case": "Positive momentum continuation could improve market sentiment.",
                "bear_case": "Weak follow-through or negative headline flow could pressure the stock.",
                "risks": ["Headline-driven volatility", "Sector rotation"],
                "catalysts": [top_headline],
                "open_questions": ["Will the latest move sustain over the next week?"],
                "what_changed": "Memory initialized from fallback research logic.",
                "confidence": "low",
            }
        )
    return {
        "portfolio_summary": "Fallback research mode was used for this run.",
        "change_log": "Initial ticker memory was created without the research LLM path.",
        "ticker_updates": updates,
    }


def render_ticker_memory_card(memory: dict[str, Any], news_items: list[dict[str, str]]) -> str:
    risks = "".join(f"<li>{html.escape(item)}</li>" for item in memory["risks"]) or "<li>None noted.</li>"
    catalysts = "".join(f"<li>{html.escape(item)}</li>" for item in memory["catalysts"]) or "<li>None noted.</li>"
    questions = "".join(f"<li>{html.escape(item)}</li>" for item in memory["open_questions"]) or "<li>None noted.</li>"
    headlines = stock_agent.render_news_list(news_items[:3])
    return (
        "<div style='margin-top:18px; padding:18px; border:1px solid #eaecf0; border-radius:18px; background:#ffffff;'>"
        f"<h3 style='margin:0 0 6px;'>{html.escape(memory['ticker'])} - {html.escape(memory['name'])}</h3>"
        f"<p style='margin:0 0 10px; color:#475467;'><strong>Thesis:</strong> {html.escape(memory['thesis'])}</p>"
        f"<p style='margin:0 0 8px; color:#475467;'><strong>Bull case:</strong> {html.escape(memory['bull_case'])}</p>"
        f"<p style='margin:0 0 8px; color:#475467;'><strong>Bear case:</strong> {html.escape(memory['bear_case'])}</p>"
        f"<p style='margin:0 0 8px; color:#475467;'><strong>What changed:</strong> {html.escape(memory['what_changed'])}</p>"
        f"<p style='margin:0 0 12px; color:#475467;'><strong>Confidence:</strong> {html.escape(memory['confidence'])}</p>"
        "<div style='margin-top:12px;'><strong>Risks</strong><ul style='margin:6px 0 10px;'>" + risks + "</ul></div>"
        "<div><strong>Catalysts</strong><ul style='margin:6px 0 10px;'>" + catalysts + "</ul></div>"
        "<div><strong>Open questions</strong><ul style='margin:6px 0 10px;'>" + questions + "</ul></div>"
        "<div><strong>Recent headlines</strong><ul style='margin:6px 0 0;'>" + headlines + "</ul></div>"
        "</div>"
    )


def wrap_research_email(
    generated_at: datetime,
    updates: dict[str, Any],
    research_news: dict[str, list[dict[str, str]]],
) -> str:
    timestamp = generated_at.strftime("%Y-%m-%d %I:%M %p %Z")
    cards = []
    for update in updates["ticker_updates"]:
        cards.append(render_ticker_memory_card(update, research_news.get(update["ticker"], [])))

    return f"""
    <html>
      <body style="margin:0; padding:24px 0; background:#f5f7fb; font-family:-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; color:#111827; line-height:1.6;">
        <div style="max-width:860px; margin:0 auto; padding:0 16px;">
          <div style="background:linear-gradient(135deg, #111827 0%, #1d4ed8 100%); border-radius:24px; padding:28px 24px; color:#ffffff;">
            <div style="font-size:12px; text-transform:uppercase; letter-spacing:0.12em; opacity:0.8;">Stock Research Autoresearch Loop</div>
            <h1 style="margin:10px 0;">Persistent ticker research memo</h1>
            <p style="margin:0; font-size:14px; opacity:0.9;">Generated at {html.escape(timestamp)}</p>
          </div>
          <div style="margin-top:18px; background:#ffffff; border:1px solid #eaecf0; border-radius:22px; padding:24px;">
            <h2 style="margin:0 0 8px;">Executive Summary</h2>
            <p style="margin:0 0 12px; color:#475467;">{html.escape(updates['portfolio_summary'])}</p>
            <h2 style="margin:18px 0 8px;">Research Change Log</h2>
            <p style="margin:0; color:#475467;">{html.escape(updates['change_log'])}</p>
          </div>
          <div style="margin-top:18px;">
            {''.join(cards)}
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

    research_universe = build_research_universe(config, buy_candidates)
    research_assets = stock_agent.build_focus_asset_table(research_universe, start, end)
    research_names = {
        row["Symbol"]: row["Name"] for row in research_assets.to_dict(orient="records")
    }
    research_news = stock_agent.build_ticker_news(research_names, 3)
    market_news = stock_agent.build_general_news(6)
    program_text = load_program()
    prior_memory = {ticker: load_memory(ticker) for ticker in research_universe}

    research_input = build_research_input(
        generated_at=now,
        config=config,
        research_assets=research_assets,
        research_news=research_news,
        buy_candidates=buy_candidates,
        market_news=market_news,
        program_text=program_text,
        prior_memory=prior_memory,
    )

    updates = generate_research_updates(config, research_input, research_assets)
    for update in updates["ticker_updates"]:
        save_memory(update["ticker"], update)

    email_html = wrap_research_email(now, updates, research_news)
    report_path = save_report(now, email_html)

    subject = f"Daily Stock Research - {now.strftime('%Y-%m-%d')}"
    stock_agent.send_email(config, subject, email_html)
    print(f"Research report sent to {', '.join(config.email_to)}")
    print(f"Saved research report to {report_path}")
    print(f"Updated ticker memory in {MEMORY_DIR}")


if __name__ == "__main__":
    main()
