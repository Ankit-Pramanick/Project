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
# Public users get a small number of free generations per day. A client-side
# anonymous ID (X-Anon-Key header) is used so one visitor can't exhaust the
# Gemini quota; the counter resets daily.
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
# Each request triggers up to 2 paid API calls, so unthrottled endpoints
# could exhaust the free-tier Gemini quota in seconds. Keep the public
# demo usable by spacing out requests per IP.
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

# Set up Gemini API via Environment Variable with default fallback
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# --- Model fallback list (ordered by preference) ---
# We build a ranked list of available models at startup.
# When generating content, we try each model in order,
# automatically skipping models whose quota is exhausted.

PREFERRED_MODEL_KEYWORDS = [
    'gemini-2.5-flash',
    'gemini-2.0-flash',
    'gemini-flash-latest',
    'gemini-2.5-flash-lite',
    'gemini-2.0-flash-lite',
    'gemini-2.5-pro',
    'gemini-3.5-flash',
    'gemini-3.6-flash',
    'gemini-3.1-flash-lite',
    'gemini-3.1-pro-preview',
]

def _build_model_list():
    """Return a list of model name strings, ordered by preference."""
    try:
        all_models = genai.list_models()
        supported = [
            m.name for m in all_models
            if 'generateContent' in getattr(m, 'supported_generation_methods', [])
        ]
        print("Available generateContent models:", supported)
    except Exception as e:
        print(f"Error listing models: {e}")
        supported = []

    ordered = []
    seen = set()
    # First pass: add models that match our preference keywords, in order.
    # Exact model-name matching only: 'gemini-2.5-flash' must not match
    # 'gemini-2.5-flash-lite', '-image', or '-preview-tts' variants.
    for keyword in PREFERRED_MODEL_KEYWORDS:
        for full_name in supported:
            if full_name == 'models/' + keyword and full_name not in seen:
                ordered.append(full_name)
                seen.add(full_name)
    # Second pass: add any remaining models we haven't added yet
    for full_name in supported:
        if full_name not in seen:
            ordered.append(full_name)
            seen.add(full_name)

    if not ordered:
        # Absolute last resort
        ordered = ['models/gemini-2.0-flash-lite']

    print("Model fallback order:", ordered)
    return ordered

MODEL_LIST = _build_model_list()

def generate_with_fallback(prompt, max_retries=2, initial_delay=10):
    """Try to generate content, falling back across models on quota errors."""
    last_error = None
    for model_name in MODEL_LIST:
        model = genai.GenerativeModel(model_name)
        for attempt in range(max_retries):
            try:
                response = model.generate_content(prompt)
                return response  # success
            except Exception as e:
                last_error = e
                err_str = str(e)
                if '429' in err_str or 'RESOURCE_EXHAUSTED' in err_str or 'quota' in err_str.lower():
                    if attempt < max_retries - 1:
                        wait = initial_delay * (attempt + 1)
                        print(f"Rate limited on {model_name}, retrying in {wait}s (attempt {attempt+1})...")
                        time.sleep(wait)
                    else:
                        print(f"Quota exhausted for {model_name}, trying next model...")
                        break  # move to next model
                elif '404' in err_str or 'not found' in err_str.lower():
                    print(f"Model {model_name} not found, skipping...")
                    break  # move to next model
                else:
                    print(f"Unexpected error with {model_name}: {e}")
                    break  # move to next model

    raise Exception(f"All models exhausted. Last error: {last_error}")

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
    {text}
    """
    
    try:
        response = generate_with_fallback(prompt)
        skills_text = response.text.strip()
        
        # Preferred path: parse a JSON array directly
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
        
        # Fallback: line-based parsing
        skills = []
        for line in skills_text.split('\n'):
            line = line.strip()
            if line and not line.startswith('#') and not line.lower().startswith('skill'):
                # Remove bullet points or numbering
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
    except Exception as e:
        print(f"Error extracting skills: {e}")
        return ["Error extracting skills"]

def generate_mcq_questions(skills, resume_text, question_count=15):
    num_case = 5 if question_count >= 8 else max(1, question_count // 3)
    num_mcq = max(1, question_count - num_case)

    prompt = f"""
    Create {num_mcq} multiple-choice and {num_case} case based questions based on this resume text to test the candidate's knowledge about their listed skills and experience. Give the case based questions also in the mcq format.
    
    Resume text: {resume_text}
    
    Key skills extracted: {', '.join(skills)}
    
    For each question:
    1. Create a technically relevant question about one of the skills
    2. Provide 4 possible answers with one correct answer
    3. Make sure the questions test real technical knowledge, not just resume facts
    4. Make the case based questions real time scenario dependant
    
    Format your response as a valid JSON array with this structure:
    [
        {{
            "question": "Question text here?",
            "options": ["Option A", "Option B", "Option C", "Option D"],
            "correct_answer": "The correct option"
        }},
        ...more questions...
    ]
    
    Return ONLY the JSON with no other text.
    """
    
    try:
        response = generate_with_fallback(prompt)
        
        # Extracting JSON from the response
        questions_text = response.text.strip()
        # Remove any markdown code block indicators if present
        if questions_text.startswith("```json"):
            questions_text = questions_text[7:]
        if questions_text.endswith("```"):
            questions_text = questions_text[:-3]
        
        questions_text = questions_text.strip()
        questions = json.loads(questions_text)
        
        # Format the response for the frontend and determine correct answer indices
        formatted_questions = []
        correct_answer_indices = []
        
        for q in questions:
            formatted_questions.append({
                "question": q["question"],
                "options": q["options"]
            })
            
            # Find the index of the correct answer
            correct_opt = q["correct_answer"]
            try:
                # First try to find exact string match
                correct_index = q["options"].index(correct_opt)
            except ValueError:
                # If that fails, find the closest matching option
                for i, option in enumerate(q["options"]):
                    if correct_opt in option or option in correct_opt:
                        correct_index = i
                        break
                else:
                    # Default to first option if no match found
                    correct_index = 0
                    
            correct_answer_indices.append(correct_index)
        
        return formatted_questions, questions, correct_answer_indices
    except Exception as e:
        print(f"Error generating questions: {e}")
        print(f"Raw response: {response.text if 'response' in locals() else 'No response'}")
        return [], [], []

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
# Correct answers never ship to the browser. After generating an assessment we
# keep the answer key in memory keyed by a short-lived test ID; the client only
# submits its selections and receives the score back.
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
        
        # Validate file type by extension AND content (magic bytes)
        file_kind = detect_file_type(filename, file_bytes)
        if not file_kind:
            return jsonify({"error": "Unsupported or invalid file. Please upload a valid PDF or DOCX."}), 400
        
        # Extract text based on validated file type
        if file_kind == 'pdf':
            resume_text = extract_text_from_pdf(file_bytes)
        else:
            resume_text = extract_text_from_docx(file_bytes)
        
        if not resume_text.strip():
            return jsonify({"error": "Could not extract text from the uploaded file."}), 400
        
        question_count = _question_count_from_request()
        payload = build_assessment(resume_text, question_count)
        entry['count'] += 1  # consume one run on success
        payload['usage'] = _usage_payload(entry)
        return jsonify(payload)
    
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
        entry['count'] += 1  # consume one run on success
        payload['usage'] = _usage_payload(entry)
        return jsonify(payload)
    except ValueError as e:
        print(f"Error generating sample demo: {e}")
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        print(f"Error generating sample demo: {e}")
        return jsonify({"error": "Could not run the sample demo right now. Please try again shortly."}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)