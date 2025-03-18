# app.py
from flask import Flask, request, jsonify, render_template
import os
import google.generativeai as genai
import PyPDF2
import docx
from werkzeug.utils import secure_filename
import json

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Set up Gemini API - replace with your actual API key
GEMINI_API_KEY = "AIzaSyAHT3w96tMdb_I-IzDjWg2giYQq5i7EbUA"
genai.configure(api_key=GEMINI_API_KEY)

# Try to get available models
def get_available_models():
    try:
        models = genai.list_models()
        model_names = [model.name for model in models]
        print("Available models:", model_names)
        return model_names
    except Exception as e:
        print(f"Error listing models: {e}")
        return []

# Choose the best available model
available_models = get_available_models()
if 'gemini-1.5-pro' in available_models:
    MODEL_NAME = 'gemini-1.5-pro'
elif 'gemini-pro' in available_models:
    MODEL_NAME = 'gemini-pro'
else:
    # Default fallback if we can't determine available models
    MODEL_NAME = 'gemini-1.5-pro'  

print(f"Using model: {MODEL_NAME}")
model = genai.GenerativeModel(MODEL_NAME)

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
    except Exception as e:
        print(f"Error extracting PDF text: {e}")
    return text

def extract_text_from_docx(docx_path):
    text = ""
    try:
        doc = docx.Document(docx_path)
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
        response = model.generate_content(prompt)
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
        response = model.generate_content(prompt)
        
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
                # Note: We don't send the correct answer directly
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
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Extract text based on file type
        if filename.lower().endswith('.pdf'):
            resume_text = extract_text_from_pdf(file_path)
        elif filename.lower().endswith('.docx'):
            resume_text = extract_text_from_docx(file_path)
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
        
        # Save the full questions with correct answers for later reference
        results_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{os.path.splitext(filename)[0]}_results.json")
        with open(results_path, 'w') as f:
            json.dump(full_questions, f)
        
        return jsonify({
            "skills": skills,
            "test": formatted_questions,
            "answers": correct_indices  # Send the correct answer indices to frontend
        })
    
    except Exception as e:
        print(f"Error processing file: {e}")
        return jsonify({"error": f"Error processing file: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)