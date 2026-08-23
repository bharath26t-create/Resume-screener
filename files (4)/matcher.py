import json
from openai import OpenAI

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI()
    return _client

EXTRACTION_PROMPT = """Extract structured data from this resume. Return ONLY valid JSON, no other text, in this exact shape:

{{
  "skills": ["skill1", "skill2"],
  "experience": [{{"role": "...", "company": "...", "duration": "..."}}],
  "education": [{{"degree": "...", "institution": "...", "year": "..."}}]
}}

Resume text:
{resume_text}
"""

SCORING_PROMPT = """Compare the following resume with this job description and rate fit on 1-10 with justification.

Return ONLY valid JSON in this exact shape:
{{
  "score": <integer 1-10>,
  "justification": "<2-3 sentence explanation citing specific matches and gaps>"
}}

Resume (structured):
{resume_data}

Job Description:
{job_description}
"""


def extract_structured_data(resume_text):
    response = _get_client().chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": EXTRACTION_PROMPT.format(resume_text=resume_text)}],
        temperature=0,
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(raw)


def score_match(resume_data, job_description):
    response = _get_client().chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": SCORING_PROMPT.format(
                resume_data=json.dumps(resume_data),
                job_description=job_description,
            ),
        }],
        temperature=0,
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(raw)
