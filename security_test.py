import io
import os
import sys
import time

# Suppress noisy model-listing output during import
_real_stdout = sys.stdout
sys.stdout = open(os.devnull, 'w')
import app as app_mod
sys.stdout.close()
sys.stdout = _real_stdout

app_mod._last_request.clear()
app_mod._usage.clear()
app_mod.app.testing = True
client = app_mod.app.test_client()

TEST_KEY = 'test-anon-key-001'
HEADERS = {'X-Anon-Key': TEST_KEY}

def reset_all():
    app_mod._last_request.clear()
    app_mod._usage.clear()

def post_file(name, data):
    reset_all()
    data = data if isinstance(data, bytes) else data.encode('utf-8')
    rv = client.post('/upload_resume',
                     data={'file': (io.BytesIO(data), name)},
                     headers=HEADERS,
                     content_type='multipart/form-data')
    return rv.status_code, rv.get_json()

def post_no_file():
    reset_all()
    rv = client.post('/upload_resume', headers=HEADERS)
    return rv.status_code, rv.get_json()

results = []

def record(label, ok, detail=''):
    results.append((label, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {label} -> {detail}")

# 1. No file part
code, body = post_no_file()
record('missing file -> 400', code == 400 and 'No file part' in (body or {}).get('error', ''),
       f'{code} {body}')

# 2. Wrong extension (.txt)
code, body = post_file('resume.txt', 'hello world')
record('plain .txt -> 400 invalid', code == 400, f'{code} {body}')

# 3. Text file renamed to .pdf (magic-byte check)
code, body = post_file('resume.pdf', 'this is not really a pdf')
record('fake .pdf (no %PDF magic) -> 400', code == 400, f'{code} {body}')

# 4. Bare %PDF header but no body (empty/invalid pdf)
code, body = post_file('resume.pdf', '%PDF-')
record('empty %PDF- header -> 400', code == 400, f'{code} {body}')

# 5. Fake .docx (PK magic, not a real zip)
code, body = post_file('resume.docx', 'PK not-a-real-zip' * 20)
record('fake .docx (PK but not zip) -> 400', code == 400, f'{code} {body}')

# 6. HTML disguised as PDF (XSS attempt through upload)
code, body = post_file('resume.pdf', '%PDF-<script>alert(1)</script>')
record('html-in-pdf -> 400', code == 400, f'{code} {body}')

# 7. Oversized file -> 413
try:
    reset_all()
    big = b'%PDF-' + b'x' * (6 * 1024 * 1024)
    rv = client.post('/upload_resume',
                     data={'file': (io.BytesIO(big), 'big.pdf')},
                     headers=HEADERS,
                     content_type='multipart/form-data')
    record('6MB pdf -> 413', rv.status_code == 413, f'{rv.status_code}')
except Exception as e:
    record('6MB pdf -> 413 (raised)', '413' in str(e) or 'RequestEntityTooLarge' in str(e) or 'too large' in str(e).lower(),
           f'raised: {type(e).__name__}: {e}')

# 8. Security headers present
rv = client.get('/')
h = rv.headers
record('nosniff header', h.get('X-Content-Type-Options') == 'nosniff', h.get('X-Content-Type-Options'))
record('frame-options header', h.get('X-Frame-Options') == 'SAMEORIGIN', h.get('X-Frame-Options'))
record('referrer-policy header', h.get('Referrer-Policy') == 'strict-origin-when-cross-origin', h.get('Referrer-Policy'))
record('permissions-policy header', h.get('Permissions-Policy') is not None, h.get('Permissions-Policy'))

# 9. No internal exception leakage in 500 responses (generic error text only)
reset_all()
def boom(resume_text, question_count=15):
    raise RuntimeError("secret-internal-path\\app.py at line 999")
app_mod.build_assessment = boom
rv = client.post('/sample_demo', headers=HEADERS)
code = rv.status_code
body = rv.get_json() or {}
leaks = any(x in (body.get('error') or '') for x in ['Traceback', 'File "', 'app.py', 'secret-internal', 'RuntimeError'])
record(f'sample_demo ({code}) no internal leakage', code == 500 and not leaks, f'{code} {body}')

# 10. Rate limiter returns 429 on rapid repeats
reset_all()
client.post('/upload_resume', headers=HEADERS)
rv = client.post('/upload_resume', headers=HEADERS)
record('rate limiter -> 429', rv.status_code == 429, f'{rv.status_code} {rv.get_json()}')

# 11. Server-side scoring: answers never ship to the client
reset_all()
def fake_build(resume_text, question_count=15):
    return {
        "skills": ["Python", "Flask"],
        "test": [{"question": "Q1?", "options": ["A", "B", "C", "D"]},
                 {"question": "Q2?", "options": ["A", "B", "C", "D"]}],
        "test_id": "seed123"
    }
app_mod.build_assessment = fake_build
rv = client.post('/sample_demo', headers=HEADERS)
body = rv.get_json() or {}
no_answers_leaked = 'answers' not in body and 'test_id' in body and 'usage' in body
record('generate response has no answer key + has usage', rv.status_code == 200 and no_answers_leaked,
       f'{rv.status_code} keys={sorted(body.keys())}')

# 12. /submit_test scores correctly against the stored key
tid = 'seed123'
app_mod._tests[tid] = {'answers': [0, 1, 2], 'created': time.time()}
rv = client.post('/submit_test', json={'test_id': tid, 'selected': {'0': 0, '1': 1, '2': 3}})
body = rv.get_json() or {}
record('submit_test scores correctly', rv.status_code == 200 and body.get('score') == 2
       and body.get('total') == 3 and body.get('percentage') == 67,
       f'{rv.status_code} {body}')

# 13. /submit_test rejects unknown/expired test IDs
rv = client.post('/submit_test', json={'test_id': 'doesnotexist', 'selected': {}})
record('submit_test rejects unknown test id', rv.status_code == 400, f'{rv.status_code} {rv.get_json()}')

# 14. /healthz is cheap and reachable
rv = client.get('/healthz')
record('healthz returns ok', rv.status_code == 200 and (rv.get_json() or {}).get('status') == 'ok',
       f'{rv.status_code} {rv.get_json()}')

# 15. /api/usage returns usage object
reset_all()
rv = client.get('/api/usage', headers=HEADERS)
u = (rv.get_json() or {}).get('usage', {})
record('api/usage returns valid object', rv.status_code == 200 and u.get('limit') == 5
       and u.get('remaining') == 5 and u.get('used') == 0,
       f'{rv.status_code} {u}')

# 16. Daily quota: 5 successful runs then 6th is rejected (quota exhaustion)
reset_all()
app_mod.build_assessment = fake_build
for i in range(5):
    app_mod._last_request.clear()  # clear only rate limiter, keep usage count
    rv = client.post('/sample_demo', headers=HEADERS)
    u = (rv.get_json() or {}).get('usage', {})
    ok = rv.status_code == 200 and u.get('used') == i + 1
    record(f'run {i+1}/5 succeeds', ok, f'{rv.status_code} used={u.get("used")}')
# 6th run
app_mod._last_request.clear()
rv = client.post('/sample_demo', headers=HEADERS)
body = rv.get_json() or {}
record('6th run -> 429 quota exhausted', rv.status_code == 429
       and 'free daily' in (body.get('error') or ''),
       f'{rv.status_code} {body}')

failed = [r for r in results if not r[1]]
print(f'\n=== {len(results) - len(failed)}/{len(results)} checks passed ===')
sys.exit(1 if failed else 0)
