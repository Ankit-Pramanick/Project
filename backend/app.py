# app.py
from flask import Flask, request, jsonify, render_template
import os
import io
import json
import time
import uuid
import datetime
from functools import wraps
import google.generativeai as genai
import PyPDF2
import docx
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, template_folder='../templates', static_folder='../static')
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5 MB upload cap
app.config['MAX_FORM_MEMORY_SIZE'] = 1 * 1024 * 1024

# --- Anonymous daily usage quota (free tier) ---
MAX_FREE_DAILY_RUNS = 5
_usage = {}  # anon_key -> {'date': 'YYYY-MM-DD', 'count': int}

def _anon_key():
    return request.headers.get('X-Anon-Key') or request.remote_addr or 'unknown'

def _get_usage(key):
    today = datetime.date.today().isoformat()
    entry = _usage.get(key)
    if not entry or entry['date'] != today:
        entry = {'date': today, 'count': 0}
        _usage[key] = entry
    return entry

def _usage_payload(entry):
    return {
        'used': entry['count'],
        'limit': MAX_FREE_DAILY_RUNS,
        'remaining': max(0, MAX_FREE_DAILY_RUNS - entry['count']),
        'date': entry['date'],
    }

def _question_count_from_request():
    try:
        raw = request.form.get('question_count') or request.args.get('question_count') or 15
        count = int(raw)
    except (TypeError, ValueError):
        count = 15
    return max(5, min(20, count))

# --- Simple in-memory rate limiter for the Gemini-backed endpoints ---
RATE_LIMIT_SECONDS = {
    '/upload_resume': 30,
    '/sample_demo': 60,
}
_last_request = {}

def rate_limited(seconds):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = request.remote_addr or 'unknown'
            now = time.time()
            if now - _last_request.get(key, 0) < seconds:
                return jsonify({"error": "Too many requests. Please wait a moment and try again."}), 429
            _last_request[key] = now
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Set up Gemini API via Environment Variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# --- Text-generation models only ---
# Prefer ultra-fast text models with reliable json output.
MODEL_LIST = [
    'gemini-3.5-flash-lite',
    'gemini-3.1-flash-lite',
    'gemini-3.6-flash',
    'gemini-flash-lite-latest',
    'gemma-4-26b-a4b-it',
]

class GeminiQuotaError(Exception):
    pass

class GeminiGenerationError(Exception):
    pass


def generate_with_fallback(prompt, max_output_tokens=1800, timeout_seconds=45):
    """Generate text using compatible text models with a bounded request time."""
    last_error = None

    for model_name in MODEL_LIST:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=max_output_tokens,
                    response_mime_type="application/json",
                ),
                request_options={'timeout': timeout_seconds},
            )
            print(f"Success with model: {model_name}")
            return response
        except Exception as e:
            last_error = e
            err_str = str(e)
            lowered = err_str.lower()

            if '429' in err_str or 'resource_exhausted' in lowered or 'quota' in lowered:
                print(f"Quota exhausted for {model_name}, trying next model...")
                continue

            if '404' in err_str or 'not found' in lowered:
                print(f"Model {model_name} not found, trying next model...")
                continue

            if '504' in err_str or 'deadline exceeded' in lowered or 'timeout' in lowered:
                print(f"Timeout with {model_name}, trying next model...")
                continue

            print(f"Error with {model_name}: {e}")
            continue

    if last_error is not None:
        err_str = str(last_error)
        lowered = err_str.lower()
        if '429' in err_str or 'resource_exhausted' in lowered or 'quota' in lowered:
            raise GeminiQuotaError(
                "Gemini API quota is temporarily exhausted. Please try again later."
            ) from last_error

    raise GeminiGenerationError(
        f"Gemini text generation failed. Last error: {last_error}"
    ) from last_error


def detect_file_type(filename, data):
    """Validate a file by extension AND magic bytes. Returns 'pdf'|'docx'|None."""
    name = (filename or '').lower()
    if name.endswith('.pdf') and data[:5] == b'%PDF-':
        return 'pdf'
    if name.endswith('.docx') and data[:2] == b'PK':
        return 'docx'
    return None


def extract_text_from_pdf(file_bytes):
    text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    except Exception as e:
        print(f"Error extracting PDF text: {e}")
    return text


def extract_text_from_docx(file_bytes):
    text = ""
    try:
        doc = docx.Document(io.BytesIO(file_bytes))
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        print(f"Error extracting DOCX text: {e}")
    return text


def extract_skills_from_text(text):
    prompt = f"""
    Extract the key skills, technologies, and qualifications from this resume text.
    Return ONLY a JSON array of strings, with no preamble, explanation, or code fences.
    Example: ["Python", "Flask", "Docker", "PostgreSQL"]

    Resume text:
    {text[:20000]}
    """

    response = None
    try:
        response = generate_with_fallback(prompt, max_output_tokens=1000, timeout_seconds=30)
        skills_text = response.text.strip()

        try:
            cleaned = skills_text
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            if cleaned.startswith("["):
                skills = json.loads(cleaned)
                if isinstance(skills, list):
                    skills = [str(s).strip() for s in skills if str(s).strip()]
                    if skills:
                        return skills
        except Exception:
            pass

        skills = []
        for line in skills_text.split('\n'):
            line = line.strip()
            if line and not line.startswith('#') and not line.lower().startswith('skill'):
                cleaned_line = line
                if line.startswith('- '):
                    cleaned_line = line[2:]
                elif line.startswith('* '):
                    cleaned_line = line[2:]
                elif len(line) > 2 and line[0].isdigit() and line[1] == '.':
                    cleaned_line = line[2:].strip()

                if cleaned_line:
                    skills.append(cleaned_line)

        return skills
    except (GeminiQuotaError, GeminiGenerationError):
        raise
    except Exception as e:
        print(f"Error extracting skills: {e}")
        return []


def _parse_questions_response(response):
    questions_text = response.text.strip()
    if questions_text.startswith("```json"):
        questions_text = questions_text[7:]
    if questions_text.endswith("```"):
        questions_text = questions_text[:-3]
    questions_text = questions_text.strip()
    questions = json.loads(questions_text)

    if not isinstance(questions, list):
        raise ValueError("Gemini did not return a JSON array of questions.")

    formatted_questions = []
    correct_answer_indices = []

    for q in questions:
        options = q["options"]
        if not isinstance(options, list) or len(options) != 4:
            raise ValueError("Each question must contain exactly 4 options.")

        formatted_questions.append({
            "question": q["question"],
            "options": options,
        })

        correct_opt = str(q["correct_answer"]).strip()
        try:
            correct_index = options.index(q["correct_answer"])
        except ValueError:
            correct_index = None
            for i, option in enumerate(options):
                option_text = str(option).strip()
                if correct_opt == option_text or correct_opt in option_text or option_text in correct_opt:
                    correct_index = i
                    break

            if correct_index is None:
                raise ValueError(
                    f"Correct answer does not match any option: {q['correct_answer']}"
                )

        correct_answer_indices.append(correct_index)

    return formatted_questions, questions, correct_answer_indices


def generate_mcq_questions(skills, resume_text, question_count=15):
    """Generate questions in small batches to avoid long or truncated responses."""
    all_formatted = []
    all_full = []
    all_correct_indices = []
    remaining = question_count
    batch_size = 3
    batch_number = 1

    while remaining > 0:
        current_count = min(batch_size, remaining)
        num_case = 1 if current_count >= 3 else 0
        num_mcq = current_count - num_case

        prompt = f"""
        Create exactly {num_mcq} technical multiple-choice questions and {num_case} case-based multiple-choice question.
        Base the questions on the candidate resume and listed skills below.

        Resume text:
        {resume_text[:12000]}

        Key skills:
        {', '.join(skills[:25])}

        Requirements:
        1. Every question must have exactly 4 options.
        2. Exactly one option must be correct.
        3. Test real technical knowledge rather than repeating resume facts.
        4. Case-based questions must describe a realistic engineering situation.
        5. Keep each question and each option concise.
        6. Return ONLY a JSON array. No markdown or explanation.

        JSON structure:
        [
          {{
            "question": "Question text",
            "options": ["Option A", "Option B", "Option C", "Option D"],
            "correct_answer": "The exact correct option text"
          }}
        ]
        """

        try:
            response = generate_with_fallback(
                prompt,
                max_output_tokens=1200,
                timeout_seconds=35,
            )
            formatted, full, correct = _parse_questions_response(response)
            if len(formatted) != current_count:
                raise ValueError(
                    f"Expected {current_count} questions but received {len(formatted)}."
                )
            all_formatted.extend(formatted)
            all_full.extend(full)
            all_correct_indices.extend(correct)
            print(f"Question batch {batch_number} generated: {len(formatted)} questions")
        except (GeminiQuotaError, GeminiGenerationError):
            raise
        except Exception as e:
            print(f"Error generating question batch {batch_number}: {e}")
            return [], [], []

        remaining -= current_count
        batch_number += 1

    return all_formatted, all_full, all_correct_indices

SAMPLE_RESUME_TEXT = """Ankit Pramanick
Senior Backend Engineer

SUMMARY
Backend engineer with 4+ years of experience designing REST APIs, data pipelines, and scalable services. Track record of improving performance, automating deployments, and shipping reliable systems.

EXPERIENCE
Acme Corp - Senior Backend Engineer (2021-Present)
- Built a microservice processing 2M requests/day with a 99.9% uptime target.
- Cut API p95 latency from 800ms to 200ms via query optimization and caching.
- Containerized services with Docker and orchestrated deployments on Kubernetes.

Startup Ltd - Backend Engineer (2019-2021)
- Designed REST APIs consumed by a mobile app with 100K+ users.
- Set up CI/CD with GitHub Actions, cutting release time from hours to minutes.

SKILLS
Python, Flask, FastAPI, PostgreSQL, Redis, Docker, Kubernetes, AWS (EC2, S3), REST API design, CI/CD, pytest, Git

EDUCATION
B.Tech in Computer Science
"""

def build_assessment(resume_text, question_count=15):
    """Run the full extraction + generation pipeline and return the response payload."""
    skills = extract_skills_from_text(resume_text)
    if not skills:
        raise ValueError("Could not extract skills from the resume.")

    formatted_questions, full_questions, correct_indices = generate_mcq_questions(skills, resume_text, question_count)
    if not formatted_questions:
        raise ValueError("Could not generate questions from the resume.")

    test_id = _store_test(correct_indices)

    return {
        "skills": skills,
        "test": formatted_questions,
        "test_id": test_id
    }

# --- Server-side test store ---
_tests = {}
TEST_TTL_SECONDS = 3600

def _store_test(answers):
    test_id = uuid.uuid4().hex[:16]
    _tests[test_id] = {'answers': answers, 'created': time.time()}
    return test_id

def _prune_tests():
    now = time.time()
    stale = [k for k, v in _tests.items() if now - v['created'] > TEST_TTL_SECONDS]
    for k in stale:
        _tests.pop(k, None)

@app.route('/submit_test', methods=['POST'])
def submit_test():
    data = request.get_json(silent=True) or {}
    test_id = str(data.get('test_id', ''))
    selected = data.get('selected') or {}

    entry = _tests.get(test_id)
    if not entry:
        return jsonify({"error": "Test session not found or expired. Please regenerate the test."}), 400

    answers = entry['answers']
    score = 0
    for q_str, chosen in selected.items():
        try:
            q_idx = int(q_str)
            if 0 <= q_idx < len(answers) and int(chosen) == answers[q_idx]:
                score += 1
        except (ValueError, TypeError):
            continue

    total = len(answers)
    percentage = round((score / total) * 100) if total else 0

    _prune_tests()

    return jsonify({
        "score": score,
        "total": total,
        "percentage": percentage,
        "answers": answers
    })

@app.route('/healthz')
def healthz():
    return jsonify({"status": "ok"})

@app.after_request
def set_security_headers(resp):
    resp.headers['X-Content-Type-Options'] = 'nosniff'
    resp.headers['X-Frame-Options'] = 'SAMEORIGIN'
    resp.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    resp.headers['Permissions-Policy'] = 'camera=(), microphone=(), geolocation=()'
    if request.headers.get('X-Forwarded-Proto', '') == 'https':
        resp.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    return resp

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Maximum allowed size is 5 MB."}), 413

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/usage', methods=['GET'])
def usage():
    key = _anon_key()
    entry = _get_usage(key)
    return jsonify({"usage": _usage_payload(entry)})

@app.route('/upload_resume', methods=['POST'])
@rate_limited(RATE_LIMIT_SECONDS['/upload_resume'])
def upload_resume():
    key = _anon_key()
    entry = _get_usage(key)
    if entry['count'] >= MAX_FREE_DAILY_RUNS:
        return jsonify({
            "error": "You've used all 5 free daily runs. Come back tomorrow or sign in (coming soon) for more.",
            "usage": _usage_payload(entry)
        }), 429

    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        filename = secure_filename(file.filename)
        file_bytes = file.read()

        file_kind = detect_file_type(filename, file_bytes)
        if not file_kind:
            return jsonify({"error": "Unsupported or invalid file. Please upload a valid PDF or DOCX."}), 400

        if file_kind == 'pdf':
            resume_text = extract_text_from_pdf(file_bytes)
        else:
            resume_text = extract_text_from_docx(file_bytes)

        if not resume_text.strip():
            return jsonify({"error": "Could not extract text from the uploaded file."}), 400

        question_count = _question_count_from_request()
        payload = build_assessment(resume_text, question_count)
        entry['count'] += 1
        payload['usage'] = _usage_payload(entry)
        return jsonify(payload)

    except GeminiQuotaError as e:
        print(f"Gemini quota error: {e}")
        return jsonify({
            "error": str(e),
            "usage": _usage_payload(entry)
        }), 429
    except GeminiGenerationError as e:
        print(f"Gemini generation error: {e}")
        return jsonify({"error": "AI generation failed. Please try again shortly."}), 502
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        print(f"Error processing file: {e}")
        return jsonify({"error": "Error processing file. Please try again."}), 500

@app.route('/sample_demo', methods=['POST'])
@rate_limited(RATE_LIMIT_SECONDS['/sample_demo'])
def sample_demo():
    """Runs the full pipeline on a bundled sample resume for one-click demoing."""
    key = _anon_key()
    entry = _get_usage(key)
    if entry['count'] >= MAX_FREE_DAILY_RUNS:
        return jsonify({
            "error": "You've used all 5 free daily runs. Come back tomorrow or sign in (coming soon) for more.",
            "usage": _usage_payload(entry)
        }), 429

    try:
        question_count = _question_count_from_request()
        payload = build_assessment(SAMPLE_RESUME_TEXT, question_count)
        entry['count'] += 1
        payload['usage'] = _usage_payload(entry)
        return jsonify(payload)
    except GeminiQuotaError as e:
        print(f"Gemini quota error: {e}")
        return jsonify({
            "error": str(e),
            "usage": _usage_payload(entry)
        }), 429
    except GeminiGenerationError as e:
        print(f"Gemini generation error: {e}")
        return jsonify({"error": "AI generation failed. Please try again shortly."}), 502
    except ValueError as e:
        print(f"Error generating sample demo: {e}")
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        print(f"Error generating sample demo: {e}")
        return jsonify({"error": "Could not run the sample demo right now. Please try again shortly."}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
