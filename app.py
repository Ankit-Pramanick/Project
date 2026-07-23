# app.py
from flask import Flask, request, jsonify, render_template
import os
import io
import json
import time
import google.generativeai as genai
import PyPDF2
import docx
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Set up Gemini API via Environment Variable with default fallback
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# --- Model fallback list (ordered by preference) ---
# We build a ranked list of available models at startup.
# When generating content, we try each model in order,
# automatically skipping models whose quota is exhausted.

PREFERRED_MODEL_KEYWORDS = [
    'gemini-3.6-flash',
    'gemini-2.0-flash-lite',
    'gemini-2.0-flash',
    'gemini-3.5-flash',
    'gemini-3.1-flash-lite',
    'gemini-2.5-pro',
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
    # First pass: add models that match our preference keywords, in order
    for keyword in PREFERRED_MODEL_KEYWORDS:
        for full_name in supported:
            if keyword in full_name and full_name not in seen:
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
    Extract key skills, technologies, and qualifications from this resume text. Format as a simple list:
    
    {text}
    """
    
    try:
        response = generate_with_fallback(prompt)
        skills_text = response.text
        
        # Process the response to get a clean list of skills
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

def generate_mcq_questions(skills, resume_text):
    prompt = f"""
    Create 10 multiple-choice and 5 case based questions based on this resume text to test the candidate's knowledge about their listed skills and experience. Give the cased based questions also in the mcq format.
    
    Resume text: {resume_text}
    
    Key skills extracted: {', '.join(skills)}
    
    For each question:
    1. Create a technically relevant question about one of the skills
    2. Provide 4 possible answers with one correct answer
    3. Make sure the questions test real technical knowledge, not just resume facts
    4. Make the test based questions real time scenario dependant
    
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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_resume', methods=['POST'])
def upload_resume():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    try:
        filename = secure_filename(file.filename)
        file_bytes = file.read()
        
        # Extract text based on file type
        if filename.lower().endswith('.pdf'):
            resume_text = extract_text_from_pdf(file_bytes)
        elif filename.lower().endswith('.docx'):
            resume_text = extract_text_from_docx(file_bytes)
        else:
            return jsonify({"error": "Unsupported file format. Please upload PDF or DOCX."}), 400
        
        if not resume_text.strip():
            return jsonify({"error": "Could not extract text from the uploaded file."}), 400
        
        # Extract skills
        skills = extract_skills_from_text(resume_text)
        if not skills:
            return jsonify({"error": "Could not extract skills from the resume."}), 400
        
        # Generate MCQ test
        formatted_questions, full_questions, correct_indices = generate_mcq_questions(skills, resume_text)
        if not formatted_questions:
            return jsonify({"error": "Could not generate questions from the resume."}), 400
        
        return jsonify({
            "skills": skills,
            "test": formatted_questions,
            "answers": correct_indices
        })
    
    except Exception as e:
        print(f"Error processing file: {e}")
        return jsonify({"error": f"Error processing file: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)