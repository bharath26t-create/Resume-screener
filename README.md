[README.md](https://github.com/user-attachments/files/31354118/README.md)
# Smart Resume Screener

Parses resumes (PDF/text), extracts structured data with an LLM, and scores
candidate fit against a job description with a justification — matching the
brief: extract skills/experience/education, compute an LLM match score,
show shortlisted candidates with reasoning.

## Architecture

```
Browser (templates/index.html)
   |  upload resume + job description
   v
Flask API (app.py)
   |
   |--> resume_parser.py         -- pulls raw text out of PDF or .txt
   |--> matcher.py                -- 2 LLM calls: extract structured data, then score fit
   |--> category_classifier.py    -- free local TF-IDF + LinearSVC model (no API call)
   |--> db.py                     -- SQLite storage of every screened candidate
   v
SQLite (screener.db)
```

## Category classifier (local, free, no LLM)

Trained on the Kaggle "Resume Dataset" (`data/Resume.csv` — 2,484 resumes
across 24 job categories) with `train_category_classifier.py`. This is a
TF-IDF vectorizer + Linear SVM, run entirely on your machine — no API key,
no cost, no network call at inference time. It's a separate signal from the
GPT-based JD-fit score, not a replacement for it: the dataset only has
category labels (e.g. "HR", "ENGINEERING"), not job-description match
scores, so this model answers "what field is this resume in?" while the
GPT scoring call answers "how well does it fit *this specific* job
description?"

**Real result from training on this dataset** (80/20 train/test split,
stratified by category): **73.4% test accuracy** across 24 classes. Some
categories perform very well (DESIGNER: 93% F1, INFORMATION-TECHNOLOGY:
86% F1); a few perform poorly, notably BPO (0% F1) — that category only
has ~22 total examples in the whole dataset, not enough for the model to
learn it. This is a genuine limitation of the dataset size for that class,
not a bug — worth knowing before you rely on the classifier for
underrepresented categories.

To retrain (e.g. after adding more data):
```bash
py train_category_classifier.py
```
This regenerates `category_classifier.pkl` and prints a full
per-category precision/recall/F1 report so you can see exactly where it's
weak before trusting its output.

Two separate LLM calls, not one, on purpose:
1. **Extraction call** — turns raw resume text into structured JSON
   (skills / experience / education). This is the "extract structured data"
   requirement in the brief.
2. **Scoring call** — takes that structured data plus the job description
   and produces a 1-10 fit score with justification. This is the semantic
   matching step, using essentially the exact example prompt from the brief.

Splitting these into two calls (instead of one big prompt) makes each one
easier to debug and keeps the JSON output for each step simpler and more
reliable to parse.

## What each file does

- **`app.py`** — the Flask server. Three routes: `/` (dashboard), `POST /screen`
  (upload + score a resume), `GET /candidates` (list stored results).
- **`resume_parser.py`** — uses `pypdf` to extract raw text from a PDF, or
  just reads a `.txt` file directly.
- **`matcher.py`** — the two LLM calls described above, using OpenAI's
  `gpt-4o-mini`. Both prompts force JSON-only output so the response can be
  parsed directly without extra cleanup logic.
- **`db.py`** — SQLite (built into Python, no separate database server to
  install) storing every screened candidate: filename, extracted data, job
  description, score, and justification.
- **`templates/index.html`** — a single-page dashboard: upload a resume +
  paste a job description, see the score/justification, and see a table of
  every candidate screened so far.

## The actual prompts used

**Extraction prompt** (`matcher.py`, `EXTRACTION_PROMPT`):
```
Extract structured data from this resume. Return ONLY valid JSON...
{"skills": [...], "experience": [...], "education": [...]}
```

**Scoring prompt** (`matcher.py`, `SCORING_PROMPT`) — directly based on the
brief's example prompt:
```
Compare the following resume with this job description and rate fit on
1-10 with justification.
```

## Setup

```bash
py -m pip install -r requirements.txt
```

Train the local category classifier once before first run (needs
`data/Resume.csv` in place):
```bash
py train_category_classifier.py
```

Create a `.env` file in this folder with your OpenAI key:
```
OPENAI_API_KEY=sk-...
```

Run it:
```bash
py app.py
```

Open `http://127.0.0.1:5000` in your browser. Upload a resume PDF, paste a
job description, click "Screen Resume." Results also get saved to
`screener.db` and show up in the table below the form.

## What's genuinely "simple" here (and what you'd add for a real version)

- No authentication, no multi-user support — single local SQLite file.
- No frontend framework — plain HTML/JS, no React/build step, matching the
  brief's "optional frontend dashboard."
- No resume-format handling beyond PDF/text (no .docx support).
- Scoring is single-pass, no retry logic if the LLM returns malformed JSON
  (rare with `temperature=0` but not impossible — worth adding a retry
  wrapper if you extend this).

For the demo video deliverable: screen 2-3 different resumes against the
same job description on camera so the score differences are visible, then
show the `/candidates` table populating live.
