import json

from . import app as app_module


def _parse_questions(response):
    text = (response.text or "").strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError("Gemini did not return a JSON array.")
    if not data:
        raise ValueError("Gemini returned no questions.")

    formatted = []
    full = []
    correct_indices = []

    for q in data:
        question = str(q["question"]).strip()
        options = q["options"]
        correct = str(q["correct_answer"]).strip()

        if not question or not isinstance(options, list) or len(options) != 4:
            raise ValueError("Invalid question structure.")

        option_texts = [str(x).strip() for x in options]
        match = None
        for i, option in enumerate(option_texts):
            if correct == option or correct in option or option in correct:
                match = i
                break
        if match is None:
            raise ValueError("Correct answer does not match an option.")

        formatted.append({"question": question, "options": option_texts})
        full.append({"question": question, "options": option_texts, "correct_answer": option_texts[match]})
        correct_indices.append(match)

    return formatted, full, correct_indices


def _generate_batch(skills, resume_text, count, case_count, timeout=30):
    mcq_count = count - case_count
    prompt = f"""
Generate exactly {mcq_count} technical MCQ(s) and {case_count} realistic case-based MCQ(s).
Use the candidate's skills and resume only as context; test real technical knowledge.
Every question must have exactly four concise options and exactly one correct option.
Keep every question under 25 words and every option under 12 words.
Return ONLY valid JSON. Do not use markdown.

Skills: {', '.join(skills[:20])}
Resume context: {resume_text[:7000]}

JSON format:
[{{"question":"...","options":["...","...","...","..."],"correct_answer":"exact option text"}}]
"""

    response = app_module.generate_with_fallback(
        prompt,
        max_output_tokens=700,
        timeout_seconds=timeout,
    )
    return _parse_questions(response)


def resilient_generate_mcq_questions(skills, resume_text, question_count=15):
    """Generate compact question batches and retry malformed JSON with a smaller batch."""
    formatted_all = []
    full_all = []
    correct_all = []
    remaining = question_count
    batch_number = 1

    while remaining:
        count = min(2, remaining)
        case_count = 1 if batch_number % 3 == 0 and count == 2 else 0

        try:
            formatted, full, correct = _generate_batch(
                skills, resume_text, count, case_count
            )
        except (app_module.GeminiQuotaError, app_module.GeminiGenerationError):
            raise
        except Exception as first_error:
            print(f"Question batch {batch_number} failed: {first_error}; retrying with one question.")
            try:
                formatted, full, correct = _generate_batch(
                    skills, resume_text, 1, 1 if case_count else 0, timeout=25
                )
                count = 1
            except Exception as retry_error:
                print(f"Question batch {batch_number} retry failed: {retry_error}")
                raise ValueError("Could not generate a valid question batch.") from retry_error

        if len(formatted) != count:
            raise ValueError(
                f"Expected {count} questions but received {len(formatted)}."
            )

        formatted_all.extend(formatted)
        full_all.extend(full)
        correct_all.extend(correct)
        remaining -= count
        batch_number += 1

    return formatted_all, full_all, correct_all


# build_assessment() resolves generate_mcq_questions from app_module's globals
# at request time, so replacing that function here keeps the rest of the app intact.
app_module.generate_mcq_questions = resilient_generate_mcq_questions
app = app_module.app
