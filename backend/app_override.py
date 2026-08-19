import re

from . import app as app_module


def _parse_questions(response):
    text = (response.text or "").strip()
    text = text.replace("```text", "").replace("```", "").strip()

    pattern = re.compile(
        r"QUESTION\s*:\s*(.*?)\n"
        r"A\s*:\s*(.*?)\n"
        r"B\s*:\s*(.*?)\n"
        r"C\s*:\s*(.*?)\n"
        r"D\s*:\s*(.*?)\n"
        r"ANSWER\s*:\s*([ABCD])",
        re.IGNORECASE | re.DOTALL,
    )

    matches = pattern.findall(text)
    if not matches:
        raise ValueError("Gemini returned no complete questions.")

    formatted = []
    full = []
    correct_indices = []

    for question, a, b, c, d, answer in matches:
        question = question.strip()
        options = [a.strip(), b.strip(), c.strip(), d.strip()]
        answer = answer.strip().upper()

        if not question or any(not option for option in options):
            continue

        correct_index = ord(answer) - ord("A")
        if not 0 <= correct_index < 4:
            raise ValueError("Invalid correct-answer label.")

        formatted.append({"question": question, "options": options})
        full.append({
            "question": question,
            "options": options,
            "correct_answer": options[correct_index],
        })
        correct_indices.append(correct_index)

    if not formatted:
        raise ValueError("Gemini returned no usable questions.")

    return formatted, full, correct_indices


def generate_mcq_questions(skills, resume_text, question_count=15):
    """Generate the entire assessment in one compact plain-text request."""
    case_count = 5 if question_count >= 8 else max(1, question_count // 3)
    mcq_count = question_count - case_count

    prompt = f"""
Generate exactly {question_count} technical multiple-choice questions.
Generate {mcq_count} normal technical questions and {case_count} realistic case-based questions.
Use the candidate's skills and resume only as context; test real technical knowledge.

Keep every question under 18 words.
Keep every answer option under 8 words.
Use exactly four options per question and exactly one correct answer.

Return ONLY the following six-line format for each question. No JSON, no markdown, no bullets, no commentary:
QUESTION: question text
A: option text
B: option text
C: option text
D: option text
ANSWER: A

Repeat the six-line block exactly {question_count} times.

Skills: {', '.join(skills[:20])}
Resume context: {resume_text[:6000]}
"""

    response = app_module.generate_with_fallback(
        prompt,
        max_output_tokens=max(1800, question_count * 140),
        timeout_seconds=75,
    )

    formatted, full, correct = _parse_questions(response)
    if len(formatted) != question_count:
        raise ValueError(
            f"Expected {question_count} questions but received {len(formatted)}."
        )

    print(f"Generated {len(formatted)} questions in one request")
    return formatted, full, correct


# build_assessment() resolves generate_mcq_questions from app_module's globals
# at request time, so replacing that function here keeps the rest of the app intact.
app_module.generate_mcq_questions = generate_mcq_questions
app = app_module.app
