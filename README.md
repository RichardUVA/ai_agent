# Daily S&P 500 Weak-Stock Agent

This project builds a simple local agent for your 2022 M1 Pro MacBook Pro. Each trading day it:

- pulls the current S&P 500 constituent list
- finds the daily top mover and bottom mover in the index
- gathers broad market news
- builds daily digests for `VOO`, `QQQM`, `VGT`, `KLAC`, `NVDA`, and `JPM`
- uses either a local Ollama model or GitHub Models to turn the raw data into a concise HTML end-of-day email

## Files

- `stock_agent.py`: main script
- `.env.example`: environment variables you should copy into `.env`
- `requirements.txt`: Python dependencies
- `launchd/com.richardwang.stockagent.plist.template`: macOS scheduler template
- `.github/workflows/daily-market-agent.yml`: GitHub Actions cloud scheduler

## 1. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Configure email

Copy `.env.example` to `.env` and fill in your email settings plus your Ollama settings.

```bash
cp .env.example .env
```

If you use Gmail:

- keep `SMTP_HOST=smtp.gmail.com`
- keep `SMTP_PORT=465`
- use a Gmail app password, not your normal password

For the LLM summary:

- keep `LLM_PROVIDER=ollama`
- keep `OLLAMA_URL=http://127.0.0.1:11434/api/generate`
- keep `OLLAMA_MODEL=llama3:latest` unless you want a different local model
- if Ollama is unavailable, the script still sends the email with a non-LLM fallback summary

For GitHub Actions with GitHub Models:

- set `LLM_PROVIDER=github_models` in the workflow or environment
- keep `GITHUB_MODELS_URL=https://models.github.ai/inference/chat/completions`
- keep `GITHUB_MODELS_MODEL=openai/gpt-4.1` unless you want a different GitHub Models entry
- the script reads `GITHUB_TOKEN` automatically in GitHub Actions

## 3. Run the agent manually

```bash
source .venv/bin/activate
python stock_agent.py
```

If everything is configured correctly, it will email the report to all addresses in `EMAIL_TO`.

## 4. Schedule it on macOS

Copy the template into LaunchAgents:

```bash
mkdir -p ~/Library/LaunchAgents
cp launchd/com.richardwang.stockagent.plist.template ~/Library/LaunchAgents/com.richardwang.stockagent.plist
launchctl unload ~/Library/LaunchAgents/com.richardwang.stockagent.plist 2>/dev/null || true
launchctl load ~/Library/LaunchAgents/com.richardwang.stockagent.plist
```

The template is set to run every day at 4:00 PM local time.
It points to `/Users/richardwang/Documents/ai_agent/.venv/bin/python`, so create the virtualenv first.

To change the schedule, edit these keys inside the plist:

- `Hour`
- `Minute`

## 5. Run it in GitHub Actions instead of your laptop

This repo now includes `.github/workflows/daily-market-agent.yml`.

The workflow:

- runs on GitHub, not your MacBook
- uses GitHub Models instead of local Ollama
- checks Detroit time and only sends the report at 4:00 PM `America/Detroit`
- can also be triggered manually with `workflow_dispatch`

Add these GitHub repository secrets before enabling the workflow:

- `SMTP_HOST`
- `SMTP_PORT`
- `SMTP_USERNAME`
- `SMTP_PASSWORD`
- `EMAIL_FROM`
- `EMAIL_TO`

The workflow already gets `GITHUB_TOKEN` automatically and requests `models: read` permission.

## Report behavior

By default the report includes:

- 3 brief market-news items
- daily digests for `VOO`, `QQQM`, `VGT`, `KLAC`, `NVDA`, and `JPM`
- 3 ticker-news headlines total across the digest section
- the top and bottom daily movers in the S&P 500
- a heuristic S&P 500 buy-candidate section

You can change those with these `.env` values:

- `LOOKBACK_DAYS`
- `MARKET_NEWS_COUNT`
- `DIGEST_TOTAL_NEWS_COUNT`
- `DIGEST_TICKERS`
- `OLLAMA_MODEL`
- `GITHUB_MODELS_MODEL`
- `BUY_CANDIDATE_COUNT`

## Notes

- This uses `yfinance` for market data, Google News RSS for headlines, and either Ollama or GitHub Models for the written daily digest.
- Yahoo Finance and news feeds can occasionally rate-limit or change format.
- This is a monitoring tool, not investment advice.
