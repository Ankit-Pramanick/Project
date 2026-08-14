# SkillSift AI

**Public web tool:** https://skillsiftai.onrender.com

Upload any resume (PDF/DOCX) and SkillSift AI will extract the candidate's skills using Google Gemini, generate a timed technical assessment, and score it — all in about a minute. Free for up to 5 runs per day, no sign-in required.

## What the tool does

1. **Parse** — PDF or DOCX text extraction server-side (PyPDF2 / python-docx, max 5 MB).
2. **Extract skills** — Gemini returns a clean, de-duplicated list of technologies and qualifications.
3. **Generate assessment** — 5 to 20 questions (default 15: 10 MCQ + 5 case-based), created from the extracted skills. You choose the count before generating.
4. **Score server-side** — 10-minute timed test, answer key never ships to the browser, per-question review after scoring.

## How to use it

1. Visit the app and either drag a resume onto the upload zone (or click to browse), or click **Try Sample Resume**.
2. Optionally change the question count (10 / 15 / 20) using the dropdown.
3. Click **Generate Assessment**. A progress indicator shows each pipeline stage (~1 minute).
4. Once loaded, answer the questions and click **Submit Test** to see your score.
5. Use the **Copy** / **Export JSON** / **Clear** buttons in the output panel.

## Architecture

```
Resume (PDF/DOCX)
      │  PyPDF2 / python-docx text extraction
      ▼
Resume Text
      │  Gemini: skill extraction (JSON)
      ▼
Extracted Skills
      │  Gemini: MCQ + case-based question generation (JSON)
      ▼
Technical Assessment (configurable count)   ──►  Answer key stored server-side (test ID)
      │  Client-side timer + submission
      ▼
Server-side scoring (/submit_test)
      ▼
Instant Results + Answer Review
```

### Reliability engineering

- **10-model Gemini fallback chain:** on quota/rate-limit (429/RESOURCE_EXHAUSTED) errors the app automatically rotates through a preference-ordered model list with exponential backoff, keeping the service online when a single model is exhausted.
- **Resilient JSON parsing:** strips markdown code fences and falls back to fuzzy matching when the correct answer text doesn't exactly match an option.
- **Graceful error handling:** per-format parsing failures and empty-extraction cases return clear client messages instead of crashing.

## Tech stack

| Layer | Tool |
|---|---|
| Backend | Python 3, Flask, Gunicorn |
| AI | Google Gemini API (`google-generativeai`) |
| Parsing | PyPDF2, python-docx |
| Frontend | Tailwind CSS 3 (compiled, no CDN), vanilla JS |
| Hosting | Render (free tier) |

## Getting started

```bash
git clone https://github.com/Ankit-Pramanick/Project.git
cd Project

python -m venv venv
# Windows: venv\Scripts\activate  |  macOS/Linux: source venv/bin/activate

pip install -r requirements.txt

# Create .env:
# GEMINI_API_KEY=your_key_here
# PORT=5000

python app.py
# → open http://localhost:5000
```

The committed `static/styles.css` is pre-built. To rebuild after editing Tailwind markup:
`npm install && npm run build:css` (dev-only, requires Node — output is committed).

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Public tool dashboard |
| GET | `/healthz` | Health check for keep-alive pingers (no AI calls) |
| GET | `/api/usage` | Returns `{usage: {used, remaining, limit, date}}` for the caller's anonymous key |
| POST | `/upload_resume` | Multipart: `file` (PDF/DOCX), optional `question_count`. Returns `{skills, test, test_id, usage}` |
| POST | `/sample_demo` | One-click demo on a bundled sample resume. Returns same shape as above |
| POST | `/submit_test` | JSON `{test_id, selected: {qIndex: chosen}}`. Returns `{score, total, percentage, answers}` |

All generation endpoints enforce a **5-run daily limit** per anonymous user (keyed by `X-Anon-Key` header; falls back to IP). When the limit is reached the endpoint returns 429 with the usage object so the UI can display it.

## Keeping the free tier warm

Render's free tier sleeps after ~15 minutes of inactivity (cold-start takes ~30-60 s). Add a free [UptimeRobot](https://uptimerobot.com) monitor pointing at `https://skillsiftai.onrender.com/healthz` (≤ 5 min interval) to keep the instance alive.

## Roadmap

- Full account system with per-user history, CSV export, and team dashboards
- Automated question-difficulty calibration per skill level
- Per-question explanations returned after scoring
- Webhook / email integration for hiring-team workflows
- Dark-mode auto-detection via `prefers-color-scheme` (already implemented client-side)

## License

MIT
