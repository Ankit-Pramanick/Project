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


def _generate_batch(skills, resume_text, count, case_count, timeout=30):
    mcq_count = count - case_count
    prompt = f"""
Generate exactly {count} technical multiple-choice questions.
Use the candidate's skills and resume only as context and test real technical knowledge.
{mcq_count} should be normal technical questions and {case_count} should be realistic case-based questions.
Keep each question under 20 words and every option under 10 words.

Return the questions ONLY in this exact plain-text format. Do not use JSON, markdown, bullets, or extra commentary.

QUESTION: question text
A: option text
B: option text
C: option text
D: option text
ANSWER: A

Repeat that exact six-line block for each question.

Skills: {', '.join(skills[:20])}
Resume context: {resume_text[:5000]}
"""

    response = app_module.generate_with_fallback(
        prompt,
        max_output_tokens=max(450, count * 350),
        timeout_seconds=timeout,
    )
    return _parse_questions(response)


def resilient_generate_mcq_questions(skills, resume_text, question_count=15):
    """Generate compact plain-text batches and retry incomplete batches safely."""
    formatted_all = []
    full_all = []
    correct_all = []
    remaining = question_count
    batch_number = 1

    while remaining > 0:
        count = min(2, remaining)
        case_count = 1 if batch_number % 3 == 0 and count == 2 else 0

        try:
            formatted, full, correct = _generate_batch(
                skills, resume_text, count, case_count
            )
        except (app_module.GeminiQuotaError, app_module.GeminiGenerationError):
            raise
        except Exception as first_error:
            print(
                f"Question batch {batch_number} failed: {first_error}; "
                "retrying with one question."
            )
            try:
                formatted, full, correct = _generate_batch(
                    skills,
                    resume_text,
                    1,
                    1 if case_count else 0,
                    timeout=25,
                )
                count = 1
            except Exception as retry_error:
                print(f"Question batch {batch_number} retry failed: {retry_error}")
                raise ValueError("Could not generate a valid question batch.") from retry_error

        if len(formatted) < count:
            print(
                f"Question batch {batch_number} returned {len(formatted)}/{count}; "
                "using a one-question retry for the missing questions."
            )
            missing = count - len(formatted)
            for _ in range(missing):
                try:
                    extra_formatted, extra_full, extra_correct = _generate_batch(
                        skills, resume_text, 1, 0, timeout=25
                    )
                    formatted.extend(extra_formatted)
                    full.extend(extra_full)
                    correct.extend(extra_correct)
                except Exception as retry_error:
                    raise ValueError("Could not complete the question batch.") from retry_error

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
