from __future__ import annotations

import html
import json
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

import requests
from dotenv import load_dotenv

import stock_agent
from stock_agent import ZoneInfo


BASE_DIR = Path(__file__).resolve().parent
GOOGLE_NEWS_RSS = (
    "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
)
USER_AGENT = stock_agent.USER_AGENT

DEFAULT_TOPICS = [
    "semiconductor industry",
    "NVIDIA AI chips",
    "TSMC foundry",
    "ASML lithography",
    "semiconductor equipment",
    "memory chips DRAM NAND",
    "chip export controls",
    "advanced packaging chiplets",
]
DEFAULT_OBSIDIAN_ROOT = BASE_DIR / "obsidian" / "Semiconductor News"


@dataclass
class SemiconductorConfig:
    timezone: str
    llm_provider: str
    ollama_url: str
    ollama_model: str
    github_models_url: str
    github_models_model: str
    github_models_token: str | None
    email_enabled: bool
    obsidian_root: Path
    rss_topics: list[str]
    rss_days: int
    per_topic_limit: int
    max_articles_for_ai: int


def load_config() -> SemiconductorConfig:
    load_dotenv(BASE_DIR / ".env")
    return SemiconductorConfig(
        timezone=os.getenv("TIMEZONE", "America/Detroit"),
        llm_provider=os.getenv("LLM_PROVIDER", "ollama").strip().lower(),
        ollama_url=os.getenv("OLLAMA_URL", stock_agent.OLLAMA_API_URL),
        ollama_model=os.getenv("OLLAMA_MODEL", "llama3:latest"),
        github_models_url=os.getenv(
            "GITHUB_MODELS_URL", stock_agent.GITHUB_MODELS_API_URL
        ),
        github_models_model=os.getenv("GITHUB_MODELS_MODEL", "openai/gpt-4.1"),
        github_models_token=os.getenv("GITHUB_MODELS_TOKEN") or os.getenv("GITHUB_TOKEN"),
        email_enabled=os.getenv("SEMICONDUCTOR_EMAIL_ENABLED", "true").lower()
        not in {"0", "false", "no"},
        obsidian_root=Path(
            os.getenv("SEMICONDUCTOR_OBSIDIAN_ROOT", str(DEFAULT_OBSIDIAN_ROOT))
        ).expanduser(),
        rss_topics=parse_topic_env("SEMICONDUCTOR_RSS_TOPICS", DEFAULT_TOPICS),
        rss_days=int(os.getenv("SEMICONDUCTOR_RSS_DAYS", "7")),
        per_topic_limit=int(os.getenv("SEMICONDUCTOR_RSS_PER_TOPIC_LIMIT", "8")),
        max_articles_for_ai=int(os.getenv("SEMICONDUCTOR_MAX_ARTICLES_FOR_AI", "35")),
    )


def parse_topic_env(name: str, default: list[str]) -> list[str]:
    raw_value = os.getenv(name, "")
    if not raw_value.strip():
        return default
    return [item.strip() for item in raw_value.split(";") if item.strip()]


def fetch_rss_news(topic: str, days: int, limit: int) -> list[dict[str, str]]:
    query = f"{topic} when:{days}d"
    rss_url = GOOGLE_NEWS_RSS.format(query=quote_plus(query))
    response = requests.get(rss_url, headers=USER_AGENT, timeout=30)
    response.raise_for_status()
    root = ET.fromstring(response.content)

    items = []
    for item in root.findall("./channel/item"):
        title = stock_agent.clean_text(item.findtext("title", default=""))
        link = item.findtext("link", default="")
        pub_date = item.findtext("pubDate", default="")
        source = ""
        source_node = item.find("source")
        if source_node is not None and source_node.text:
            source = stock_agent.clean_text(source_node.text)
        if title and link:
            items.append(
                {
                    "topic": topic,
                    "title": title,
                    "link": link,
                    "pub_date": pub_date,
                    "source": source,
                }
            )
        if len(items) >= limit:
            break
    return items


def collect_news(config: SemiconductorConfig) -> list[dict[str, str]]:
    articles_by_key: dict[str, dict[str, str]] = {}
    for topic in config.rss_topics:
        try:
            for item in fetch_rss_news(topic, config.rss_days, config.per_topic_limit):
                key = normalize_article_key(item["title"])
                articles_by_key.setdefault(key, item)
        except (requests.RequestException, ET.ParseError):
            continue
    return list(articles_by_key.values())


def normalize_article_key(title: str) -> str:
    cleaned = re.sub(r"\s+-\s+[^-]+$", "", title.lower())
    return re.sub(r"[^a-z0-9]+", " ", cleaned).strip()


def build_ai_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "executive_summary": {"type": "string"},
            "key_developments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "headline": {"type": "string"},
                        "why_it_matters": {"type": "string"},
                    },
                    "required": ["headline", "why_it_matters"],
                },
            },
            "trends": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "trend": {"type": "string"},
                        "evidence": {"type": "string"},
                        "watch_next": {"type": "string"},
                    },
                    "required": ["trend", "evidence", "watch_next"],
                },
            },
            "company_watchlist": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "note": {"type": "string"},
                    },
                    "required": ["name", "note"],
                },
            },
            "open_questions": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "executive_summary",
            "key_developments",
            "trends",
            "company_watchlist",
            "open_questions",
        ],
    }


def build_ai_input(
    generated_at: datetime, config: SemiconductorConfig, articles: list[dict[str, str]]
) -> dict[str, Any]:
    return {
        "generated_at": generated_at.isoformat(),
        "timezone": config.timezone,
        "coverage_window_days": config.rss_days,
        "topics": config.rss_topics,
        "articles": articles[: config.max_articles_for_ai],
    }


def generate_ai_summary(
    config: SemiconductorConfig, ai_input: dict[str, Any]
) -> dict[str, Any]:
    if config.llm_provider == "github_models":
        return generate_github_models_summary(config, ai_input)
    if config.llm_provider == "ollama":
        return generate_ollama_summary(config, ai_input)
    return build_fallback_summary(ai_input["articles"])


def generate_ollama_summary(
    config: SemiconductorConfig, ai_input: dict[str, Any]
) -> dict[str, Any]:
    prompt = build_prompt(ai_input)
    response = requests.post(
        config.ollama_url,
        json={
            "model": config.ollama_model,
            "prompt": prompt,
            "stream": False,
            "format": build_ai_schema(),
            "options": {"temperature": 0.2},
        },
        timeout=240,
    )
    response.raise_for_status()
    payload = response.json()
    parsed = json.loads(stock_agent.strip_code_fences(payload.get("response", "")))
    return normalize_summary(parsed, ai_input["articles"])


def generate_github_models_summary(
    config: SemiconductorConfig, ai_input: dict[str, Any]
) -> dict[str, Any]:
    if not config.github_models_token:
        raise ValueError("Missing GITHUB_TOKEN or GITHUB_MODELS_TOKEN for GitHub Models.")

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
                        "You write concise semiconductor industry intelligence reports. "
                        "Use only the supplied RSS headlines and links."
                    ),
                },
                {"role": "user", "content": build_prompt(ai_input)},
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
    return normalize_summary(parsed, ai_input["articles"])


def build_prompt(ai_input: dict[str, Any]) -> str:
    return (
        "Create a weekly semiconductor news intelligence report from RSS headlines. "
        "Use only the supplied data. Do not invent facts, financial metrics, or dates. "
        "Identify cross-article trends, important company or supply-chain signals, and "
        "what to watch next. Return valid JSON matching the requested schema exactly.\n\n"
        f"Input data:\n{json.dumps(ai_input, indent=2)}"
    )


def normalize_summary(summary: dict[str, Any], articles: list[dict[str, str]]) -> dict[str, Any]:
    def clean_list(value: Any, limit: int) -> list[Any]:
        return value[:limit] if isinstance(value, list) else []

    key_developments = []
    for item in clean_list(summary.get("key_developments"), 8):
        if isinstance(item, dict):
            key_developments.append(
                {
                    "headline": stock_agent.clean_text(str(item.get("headline", ""))),
                    "why_it_matters": stock_agent.clean_text(
                        str(item.get("why_it_matters", ""))
                    ),
                }
            )

    trends = []
    for item in clean_list(summary.get("trends"), 6):
        if isinstance(item, dict):
            trends.append(
                {
                    "trend": stock_agent.clean_text(str(item.get("trend", ""))),
                    "evidence": stock_agent.clean_text(str(item.get("evidence", ""))),
                    "watch_next": stock_agent.clean_text(str(item.get("watch_next", ""))),
                }
            )

    watchlist = []
    for item in clean_list(summary.get("company_watchlist"), 10):
        if isinstance(item, dict):
            watchlist.append(
                {
                    "name": stock_agent.clean_text(str(item.get("name", ""))),
                    "note": stock_agent.clean_text(str(item.get("note", ""))),
                }
            )

    open_questions = [
        stock_agent.clean_text(str(item))
        for item in clean_list(summary.get("open_questions"), 8)
    ]

    fallback = build_fallback_summary(articles)
    return {
        "executive_summary": stock_agent.clean_text(
            str(summary.get("executive_summary") or fallback["executive_summary"])
        ),
        "key_developments": key_developments or fallback["key_developments"],
        "trends": trends or fallback["trends"],
        "company_watchlist": watchlist or fallback["company_watchlist"],
        "open_questions": open_questions or fallback["open_questions"],
    }


def build_fallback_summary(articles: list[dict[str, str]]) -> dict[str, Any]:
    top_articles = articles[:8]
    topics = sorted({item["topic"] for item in articles})
    key_developments = [
        {
            "headline": item["title"],
            "why_it_matters": f"Included from the {item['topic']} RSS search.",
        }
        for item in top_articles
    ]
    return {
        "executive_summary": (
            f"Fallback mode collected {len(articles)} semiconductor RSS headlines "
            f"across {len(topics)} topics. AI summarization was unavailable."
        ),
        "key_developments": key_developments,
        "trends": [
            {
                "trend": "Headline flow is clustered by monitored topic.",
                "evidence": ", ".join(topics[:8]) or "No RSS topics returned news.",
                "watch_next": "Review repeated companies, supply-chain constraints, and policy headlines.",
            }
        ],
        "company_watchlist": [],
        "open_questions": ["Which repeated headline themes deserve deeper follow-up?"],
    }


def render_markdown_report(
    generated_at: datetime,
    config: SemiconductorConfig,
    articles: list[dict[str, str]],
    summary: dict[str, Any],
    llm_status: str,
) -> str:
    date_string = generated_at.strftime("%Y-%m-%d")
    title = f"Semiconductor News Report - {date_string}"
    lines = [
        "---",
        f'title: "{title}"',
        f"date: {date_string}",
        "type: semiconductor-news-report",
        "tags:",
        "  - semiconductor",
        "  - news",
        "  - ai-agent",
        "---",
        "",
        f"# {title}",
        "",
        f"Generated: {generated_at.strftime('%Y-%m-%d %I:%M %p %Z')}",
        f"Coverage window: last {config.rss_days} days",
        f"AI status: {llm_status}",
        "",
        "## Executive Summary",
        "",
        summary["executive_summary"],
        "",
        "## Key Developments",
        "",
    ]

    for item in summary["key_developments"]:
        lines.extend(
            [
                f"- **{item['headline']}**",
                f"  - Why it matters: {item['why_it_matters']}",
            ]
        )

    lines.extend(["", "## Trends", ""])
    for item in summary["trends"]:
        lines.extend(
            [
                f"### {item['trend']}",
                "",
                f"- Evidence: {item['evidence']}",
                f"- Watch next: {item['watch_next']}",
                "",
            ]
        )

    lines.extend(["## Company Watchlist", ""])
    if summary["company_watchlist"]:
        for item in summary["company_watchlist"]:
            lines.append(f"- **{item['name']}**: {item['note']}")
    else:
        lines.append("- No company watchlist generated.")

    lines.extend(["", "## Open Questions", ""])
    for question in summary["open_questions"]:
        lines.append(f"- {question}")

    lines.extend(["", "## Source Headlines", ""])
    for item in articles:
        source = f" - {item['source']}" if item.get("source") else ""
        lines.append(
            f"- [{item['title']}]({item['link']}){source} ({item['topic']}; {item['pub_date']})"
        )

    return "\n".join(lines).strip() + "\n"


def save_obsidian_report(
    generated_at: datetime, config: SemiconductorConfig, markdown: str
) -> Path:
    year = generated_at.strftime("%Y")
    month = generated_at.strftime("%m-%B")
    report_dir = config.obsidian_root / year / month
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / generated_at.strftime("%Y-%m-%d-semiconductor-news.md")
    report_path.write_text(markdown)
    return report_path


def render_email_report(
    generated_at: datetime,
    config: SemiconductorConfig,
    articles: list[dict[str, str]],
    summary: dict[str, Any],
    llm_status: str,
    report_path: Path,
) -> str:
    date_string = generated_at.strftime("%b %-d, %Y")
    generated_string = generated_at.strftime("%Y-%m-%d %I:%M %p %Z")
    topic_count = len(config.rss_topics)
    article_count = len(articles)
    source_count = len({item.get("source", "") for item in articles if item.get("source")})
    topic_chips = "".join(
        f"<span style='display:inline-block; margin:0 6px 6px 0; padding:6px 10px; border-radius:999px; background:#eef2ff; color:#3730a3; font-size:12px; font-weight:700;'>{html.escape(topic)}</span>"
        for topic in config.rss_topics[:8]
    )
    developments_html = "".join(
        (
            "<tr>"
            "<td style='padding:0 0 12px;'>"
            "<div style='border:1px solid #e5e7eb; border-left:4px solid #2563eb; border-radius:12px; padding:14px 16px; background:#ffffff;'>"
            f"<div style='font-size:15px; font-weight:800; color:#111827; margin-bottom:6px;'>{html.escape(item['headline'])}</div>"
            f"<div style='font-size:14px; color:#4b5563;'>{html.escape(item['why_it_matters'])}</div>"
            "</div>"
            "</td>"
            "</tr>"
        )
        for item in summary["key_developments"][:8]
    )
    trends_html = "".join(
        (
            "<tr>"
            "<td style='padding:0 0 14px;'>"
            "<div style='border:1px solid #dbe4e8; border-radius:14px; padding:16px; background:#f8fafc;'>"
            f"<div style='font-size:16px; font-weight:800; color:#0f172a; margin-bottom:10px;'>{html.escape(item['trend'])}</div>"
            "<table role='presentation' width='100%' cellspacing='0' cellpadding='0' style='border-collapse:collapse;'>"
            "<tr>"
            "<td style='padding:0 12px 0 0; width:50%; vertical-align:top;'>"
            "<div style='font-size:11px; text-transform:uppercase; letter-spacing:0.08em; color:#64748b; font-weight:800; margin-bottom:4px;'>Evidence</div>"
            f"<div style='font-size:14px; color:#334155;'>{html.escape(item['evidence'])}</div>"
            "</td>"
            "<td style='padding:0 0 0 12px; width:50%; vertical-align:top; border-left:1px solid #e2e8f0;'>"
            "<div style='font-size:11px; text-transform:uppercase; letter-spacing:0.08em; color:#64748b; font-weight:800; margin-bottom:4px;'>Watch next</div>"
            f"<div style='font-size:14px; color:#334155;'>{html.escape(item['watch_next'])}</div>"
            "</td>"
            "</tr>"
            "</table>"
            "</div>"
            "</td>"
            "</tr>"
        )
        for item in summary["trends"][:6]
    )
    company_html = render_company_watchlist(summary["company_watchlist"])
    questions_html = render_question_list(summary["open_questions"])
    source_html = render_source_headlines(articles[:12])
    return f"""
    <html>
      <body style="margin:0; padding:0; background:#eef2f6; font-family:-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; color:#111827; line-height:1.55;">
        <div style="display:none; overflow:hidden; line-height:1px; opacity:0; max-height:0; max-width:0;">Weekly semiconductor trends, key developments, and source headlines.</div>
        <div style="max-width:860px; margin:0 auto; padding:24px 14px;">
          <div style="background:#0f172a; border-radius:20px 20px 0 0; padding:28px 28px 24px; color:#ffffff;">
            <div style="font-size:12px; text-transform:uppercase; letter-spacing:0.14em; color:#93c5fd; font-weight:800;">Semiconductor Briefing</div>
            <h1 style="margin:10px 0 8px; font-size:30px; line-height:1.15; font-weight:850;">Weekly industry signal report</h1>
            <p style="margin:0; color:#cbd5e1; font-size:14px;">{html.escape(date_string)} - generated {html.escape(generated_string)}</p>
            <div style="margin-top:18px;">
              <span style="display:inline-block; margin:0 8px 8px 0; padding:8px 11px; border-radius:10px; background:#1e293b; color:#e0f2fe; font-size:13px; font-weight:800;">{article_count} headlines</span>
              <span style="display:inline-block; margin:0 8px 8px 0; padding:8px 11px; border-radius:10px; background:#1e293b; color:#e0f2fe; font-size:13px; font-weight:800;">{topic_count} topics</span>
              <span style="display:inline-block; margin:0 8px 8px 0; padding:8px 11px; border-radius:10px; background:#1e293b; color:#e0f2fe; font-size:13px; font-weight:800;">{source_count} sources</span>
            </div>
          </div>

          <div style="background:#ffffff; border:1px solid #d8dee8; border-top:0; border-radius:0 0 20px 20px; overflow:hidden;">
            <div style="padding:24px 28px; border-bottom:1px solid #e5e7eb;">
              <div style="font-size:12px; text-transform:uppercase; letter-spacing:0.1em; color:#64748b; font-weight:900; margin-bottom:8px;">Executive summary</div>
              <p style="margin:0; font-size:17px; color:#1f2937; line-height:1.65;">{html.escape(summary['executive_summary'])}</p>
              <div style="margin-top:16px;">{topic_chips}</div>
              <p style="margin:10px 0 0; color:#6b7280; font-size:12px;">{html.escape(llm_status)}</p>
            </div>

            <div style="padding:24px 28px 12px; border-bottom:1px solid #e5e7eb;">
              <h2 style="margin:0 0 16px; font-size:21px; color:#111827;">Key developments</h2>
              <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="border-collapse:collapse;">{developments_html}</table>
            </div>

            <div style="padding:24px 28px 10px; border-bottom:1px solid #e5e7eb;">
              <h2 style="margin:0 0 16px; font-size:21px; color:#111827;">Trend map</h2>
              <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="border-collapse:collapse;">{trends_html}</table>
            </div>

            <div style="padding:24px 28px; border-bottom:1px solid #e5e7eb;">
              <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="border-collapse:collapse;">
                <tr>
                  <td style="width:50%; padding:0 12px 0 0; vertical-align:top;">
                    <h2 style="margin:0 0 12px; font-size:19px; color:#111827;">Company watchlist</h2>
                    {company_html}
                  </td>
                  <td style="width:50%; padding:0 0 0 12px; vertical-align:top;">
                    <h2 style="margin:0 0 12px; font-size:19px; color:#111827;">Open questions</h2>
                    {questions_html}
                  </td>
                </tr>
              </table>
            </div>

            <div style="padding:24px 28px;">
              <h2 style="margin:0 0 14px; font-size:21px; color:#111827;">Source headlines</h2>
              {source_html}
              <p style="margin:22px 0 0; color:#6b7280; font-size:12px;">Saved to Obsidian path: {html.escape(str(report_path))}</p>
            </div>
          </div>
        </div>
      </body>
    </html>
    """.strip()


def render_company_watchlist(items: list[dict[str, str]]) -> str:
    if not items:
        return (
            "<div style='border:1px dashed #cbd5e1; border-radius:12px; padding:14px; color:#64748b; font-size:14px;'>"
            "No company watchlist generated."
            "</div>"
        )
    return "".join(
        (
            "<div style='margin:0 0 10px; padding:12px; border:1px solid #e5e7eb; border-radius:12px; background:#fcfcfd;'>"
            f"<div style='font-size:14px; font-weight:800; color:#111827;'>{html.escape(item['name'])}</div>"
            f"<div style='font-size:13px; color:#4b5563; margin-top:4px;'>{html.escape(item['note'])}</div>"
            "</div>"
        )
        for item in items[:8]
    )


def render_question_list(items: list[str]) -> str:
    if not items:
        return (
            "<div style='border:1px dashed #cbd5e1; border-radius:12px; padding:14px; color:#64748b; font-size:14px;'>"
            "No open questions generated."
            "</div>"
        )
    rows = "".join(
        f"<li style='margin:0 0 10px; color:#334155;'>{html.escape(item)}</li>"
        for item in items[:8]
    )
    return f"<ul style='margin:0; padding-left:20px; font-size:14px;'>{rows}</ul>"


def render_source_headlines(items: list[dict[str, str]]) -> str:
    if not items:
        return (
            "<div style='border:1px dashed #cbd5e1; border-radius:12px; padding:14px; color:#64748b; font-size:14px;'>"
            "No RSS headlines were collected."
            "</div>"
        )
    rows = []
    for item in items:
        source = item.get("source") or item["topic"]
        meta = f"{source} - {item['topic']}"
        rows.append(
            (
                "<tr>"
                "<td style='padding:12px 0; border-top:1px solid #eef2f7;'>"
                f"<a href='{html.escape(item['link'])}' style='color:#0f172a; font-weight:800; text-decoration:none;'>{html.escape(item['title'])}</a>"
                f"<div style='margin-top:5px; color:#64748b; font-size:12px;'>{html.escape(meta)}"
                f"{' - ' + html.escape(item['pub_date']) if item.get('pub_date') else ''}</div>"
                "</td>"
                "</tr>"
            )
        )
    return (
        "<table role='presentation' width='100%' cellspacing='0' cellpadding='0' style='border-collapse:collapse;'>"
        + "".join(rows)
        + "</table>"
    )


def main() -> None:
    config = load_config()
    now = datetime.now(ZoneInfo(config.timezone))

    articles = collect_news(config)
    ai_input = build_ai_input(now, config, articles)
    llm_status = "Fallback summary used: LLM provider not configured"
    try:
        summary = generate_ai_summary(config, ai_input)
        if config.llm_provider == "ollama":
            llm_status = f"Ollama summary generated with {config.ollama_model}"
        elif config.llm_provider == "github_models":
            llm_status = (
                f"GitHub Models summary generated with {config.github_models_model}"
            )
    except Exception as exc:
        llm_status = f"Fallback summary used: {stock_agent.clean_text(str(exc))[:160]}"
        summary = build_fallback_summary(articles)

    markdown = render_markdown_report(
        generated_at=now,
        config=config,
        articles=articles,
        summary=summary,
        llm_status=llm_status,
    )
    report_path = save_obsidian_report(now, config, markdown)

    if config.email_enabled:
        email_config = stock_agent.load_config()
        subject = f"Weekly Semiconductor News - {now.strftime('%Y-%m-%d')}"
        email_html = render_email_report(
            generated_at=now,
            config=config,
            articles=articles,
            summary=summary,
            llm_status=llm_status,
            report_path=report_path,
        )
        stock_agent.send_email(email_config, subject, email_html)
        print(f"Semiconductor report sent to {', '.join(email_config.email_to)}")
    else:
        print("Email disabled by SEMICONDUCTOR_EMAIL_ENABLED=false")
    print(f"Saved semiconductor report to {report_path}")


if __name__ == "__main__":
    main()
