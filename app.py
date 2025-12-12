import os
import time
import uuid
import json
import logging
import re
import csv
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from werkzeug.security import generate_password_hash, check_password_hash
import google.generativeai as genai
from dotenv import load_dotenv
from functools import wraps

load_dotenv()
app = Flask(__name__)
# Secret key is required for session management (Doctor Login) and flash messages
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev_secret_key_123")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    logging.warning("WARNING: GEMINI_API_KEY not found in .env")

genai.configure(api_key=api_key)


USERS_CSV = 'users.csv'
CASES_CSV = 'cases.csv'

#

def init_csv_db():
    """Initialize CSV files with headers and demo data if they don't exist."""
    if not os.path.exists(USERS_CSV):
        with open(USERS_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'username', 'password_hash', 'role', 'full_name', 'specialty', 'doctor_unique_id'])
            # Demo Data - Patients
            writer.writerow(['1', 'patient1', generate_password_hash('p123'), 'patient', 'John Doe', '', ''])
            
            # Demo Data - Doctors
            writer.writerow(['2', 'dr_smith', generate_password_hash('smith123'), 'doctor', 'Dr. James Smith', 'General Medicine', 'DOC-001'])
            writer.writerow(['3', 'dr_patel', generate_password_hash('patel123'), 'doctor', 'Dr. Rajesh Patel', 'Internal Medicine', 'DOC-002'])
            writer.writerow(['4', 'dr_lee', generate_password_hash('lee123'), 'doctor', 'Dr. Sarah Lee', 'Infectious Diseases', 'DOC-003'])
        print("Initialized users.csv with demo data.")

    if not os.path.exists(CASES_CSV):
        with open(CASES_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # We store complex data (JSON) as string in CSV fields
            writer.writerow(['case_id', 'patient_id', 'doctor_id', 'timestamp', 'raw_data_json', 'ai_analysis_json', 'status'])
        print("Initialized cases.csv.")

def get_all_users():
    users = []
    if os.path.exists(USERS_CSV):
        with open(USERS_CSV, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            users = list(reader)
    return users

def get_user_by_username(username):
    users = get_all_users()
    for user in users:
        if user['username'] == username:
            return user
    return None

def get_user_by_id(user_id):
    users = get_all_users()
    for user in users:
        if user['id'] == str(user_id):
            return user
    return None

def get_all_doctors():
    users = get_all_users()
    return [u for u in users if u['role'] == 'doctor']

def add_case(case_data):
    # Check if file is empty to write header
    file_exists = os.path.exists(CASES_CSV)
    with open(CASES_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
             writer.writerow(['case_id', 'patient_id', 'doctor_id', 'timestamp', 'raw_data_json', 'ai_analysis_json', 'status'])
        
        writer.writerow([
            case_data['id'],
            case_data['patient_id'],
            case_data['doctor_id'],
            case_data['timestamp'],
            json.dumps(case_data['raw_data']),
            json.dumps(case_data['ai_analysis']),
            case_data['status']
        ])

def get_cases_for_doctor(doctor_id):
    cases = []
    if os.path.exists(CASES_CSV):
        with open(CASES_CSV, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['doctor_id'] == str(doctor_id):
                    try:
                        row['raw_data'] = json.loads(row['raw_data_json'])
                        row['ai_analysis'] = json.loads(row['ai_analysis_json'])
                        cases.append(row)
                    except json.JSONDecodeError:
                        continue # Skip malformed rows
    return cases

def get_case_by_id(case_id):
    if os.path.exists(CASES_CSV):
        with open(CASES_CSV, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['case_id'] == str(case_id):
                    try:
                        row['raw_data'] = json.loads(row['raw_data_json'])
                        row['ai_analysis'] = json.loads(row['ai_analysis_json'])
                        return row
                    except json.JSONDecodeError:
                        return None
    return None

# Initialize CSVs on startup
init_csv_db()

# PROMPT
SYSTEM_PROMPT = """
ACT AS: Senior Clinical Consultant & Medical Scribe.
TASK: Analyze patient intake data and generate a structured clinical case file.

LANGUAGE INSTRUCTION: 
- "patient_view" MUST be in {language}.
- "doctor_view" MUST be in ENGLISH.

OUTPUT FORMAT: Return ONLY valid JSON. Do not include markdown formatting like ```json.
{{
  "patient_view": {{
    "primary_diagnosis": "Name of condition",
    "summary": "Warm explanation in {language}.",
    "pathophysiology": "Simple analogy in {language}.",
    "care_plan": ["Step 1", "Step 2"],
    "red_flags": ["Sign 1", "Sign 2"]
  }},
  "doctor_view": {{
    "subjective": "Medical terminology summary of HPI.",
    "objective": "Concise summary of reported vitals.",
    "assessment": "Differential diagnosis ranked by probability.",
    "plan": "Suggested pharmacotherapy and follow-up.",
    "subjective_list": ["Point 1", "Point 2"],
    "objective_list": ["Point 1", "Point 2"],
    "assessment_list": ["Point 1", "Point 2"],
    "plan_list": ["Point 1", "Point 2"]
  }},
  "safety": {{
    "is_safe": true,
    "warnings": []
  }}
}}
"""

# DECORATORS

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('landing'))
        return f(*args, **kwargs)
    return decorated_function

def patient_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if session.get('role') != 'patient':
            return redirect(url_for('landing'))
        return f(*args, **kwargs)
    return decorated_function

def doctor_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if session.get('role') != 'doctor':
            return redirect(url_for('landing'))
        return f(*args, **kwargs)
    return decorated_function

def clean_medical_text(text):
    if not text: return ""
    text = re.sub(r'\[\*\*', '', text)
    text = re.sub(r'\*\*\]', '', text)
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    return text.strip()

def log_interaction(case_id, inputs, latency):
    try:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "case_id": case_id,
            "model": "gemini-2.5-flash",
            "latency_ms": round(latency * 1000, 2),
            "symptoms_snippet": inputs.get('symptoms', '')[:50]
        }
        logging.info(f"MLOPS LOG: {json.dumps(log_entry)}")
        
        csv_file = 'clinical_logs.csv'
        file_exists = os.path.isfile(csv_file)
        with open(csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=log_entry.keys())
            if not file_exists: writer.writeheader()
            writer.writerow(log_entry)
    except Exception as e:
        logging.error(f"Logging Error: {e}")

#ROUTES

@app.route('/')
def landing():
    """Landing page - role selection."""
    if 'user_id' in session:
        if session['role'] == 'patient':
            return redirect(url_for('patient_intake'))
        elif session['role'] == 'doctor':
            return redirect(url_for('doctor_dashboard'))
    return render_template('landing.html')

#PATIENT ROUTES 

@app.route('/patient/login', methods=['GET', 'POST'])
def patient_login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = get_user_by_username(username)
        
        if user and user['role'] == 'patient' and check_password_hash(user['password_hash'], password):
            session['user_id'] = user['id']
            session['role'] = 'patient'
          
            session['account_name'] = user['full_name'] 
            return redirect(url_for('patient_intake'))
        else:
            flash("Invalid username or password", "danger")
    return render_template('patient_login.html')

@app.route('/patient/intake')
@login_required
@patient_required
def patient_intake():
    doctors = get_all_doctors()
   
    doctor_list = [{"id": d['id'], "name": d['full_name'], "specialty": d['specialty']} for d in doctors]
    return render_template('intake.html', doctors=doctor_list)

@app.route('/patient/submit', methods=['POST'])
@login_required
@patient_required
def patient_submit():
    start_time = time.time()
    try:
        case_id = str(uuid.uuid4())[:8].upper()
        selected_language = request.form.get('language', 'English')
        doctor_id_str = request.form.get('doctor_id')
        
        if not doctor_id_str:
            flash("Please select a doctor.", "danger")
            return redirect(url_for('patient_intake'))
            
        doctor_id = str(doctor_id_str)
        doctor = get_user_by_id(doctor_id)
        
  
        patient_name_input = request.form.get('name')
        if not patient_name_input:
             patient_name_input = session.get('account_name', 'Unknown')

        raw_data = {
            "id": case_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "patient_name": patient_name_input, 
            "doctor_name": doctor['full_name'] if doctor else "Unknown",
            "name": patient_name_input, 
            "age": request.form.get('age'),
            "gender": request.form.get('gender'),
            "weight": request.form.get('weight'),
            "height": request.form.get('height'),
            "temp": request.form.get('temperature'),
            "bp": request.form.get('blood_pressure'),
            "duration": request.form.get('duration'),
            "allergies": request.form.get('allergies') or "None",
            "current_meds": request.form.get('current_medications') or "None",
            "history": request.form.get('medical_history') or "None",
            "severity": request.form.get('severity'),
            "symptoms": request.form.get('symptoms'),
            "notes": request.form.get('other_notes'),
            "language": selected_language
        }

 
        model = genai.GenerativeModel("gemini-2.5-flash", generation_config={"response_mime_type": "application/json"})
        formatted_prompt = SYSTEM_PROMPT.format(language=selected_language)
        prompt = f"{formatted_prompt}\nPATIENT DATA: {json.dumps(raw_data, default=str)}"
        
        response = model.generate_content(prompt)
        
 
        try:
            ai_text = response.text.strip()
            if ai_text.startswith("```"):
                ai_text = re.sub(r'^```json\s*|\s*```$', '', ai_text, flags=re.MULTILINE)
            ai_analysis = json.loads(ai_text)
        except Exception as e:
            logging.error(f"JSON Parsing Failed: {response.text}")
            flash("AI Service temporarily unavailable. Please try again.", "danger")
            return redirect(url_for('patient_intake'))

        case_record = {
            'id': case_id,
            'patient_id': session['user_id'],
            'doctor_id': doctor_id,
            'timestamp': datetime.now().isoformat(),
            'raw_data': raw_data,
            'ai_analysis': ai_analysis,
            'status': "Pending Review"
        }
        add_case(case_record)
        
        log_interaction(case_id, raw_data, time.time() - start_time)
        return redirect(url_for('patient_result', case_id=case_id))

    except Exception as e:
        logging.error(f"Critical Error: {e}")
        flash(f"System Error: {str(e)}", "danger")
        return redirect(url_for('patient_intake'))

@app.route('/patient/result/<case_id>')
@login_required
@patient_required
def patient_result(case_id):
    case = get_case_by_id(case_id)
    if not case:
        flash("Case not found.", "danger")
        return redirect(url_for('patient_intake'))
    

    if case['patient_id'] != str(session['user_id']):
         flash("Access Denied", "danger")
         return redirect(url_for('patient_intake'))
        
    return render_template('patient_result.html', case=case)

@app.route('/patient/logout')
def patient_logout():
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for('landing'))



@app.route('/doctor/login', methods=['GET', 'POST'])
def doctor_login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = get_user_by_username(username)
        
        if user and user['role'] == 'doctor' and check_password_hash(user['password_hash'], password):
            session['user_id'] = user['id']
            session['role'] = 'doctor'
            session['name'] = user['full_name']
            return redirect(url_for('doctor_dashboard'))
        else:
            flash("Invalid credentials", "danger")
    return render_template('doctor_login.html')

@app.route('/doctor/dashboard')
@login_required
@doctor_required
def doctor_dashboard():
    doctor_id = session.get('user_id')
    cases_list = get_cases_for_doctor(doctor_id)
    

    cases_list.sort(key=lambda x: x['timestamp'], reverse=True)
    
    doctor_info = get_user_by_id(doctor_id)
    return render_template('doctor_dashboard.html', cases=cases_list, doctor=doctor_info)

@app.route('/doctor/view/<case_id>')
@login_required
@doctor_required
def doctor_view(case_id):
    doctor_id = session.get('user_id')
    case = get_case_by_id(case_id)
    
    if not case or case['doctor_id'] != str(doctor_id):
        flash("Case not found or access denied.", "danger")
        return redirect(url_for('doctor_dashboard'))
    
    return render_template('doctor_view.html', case=case)

@app.route('/doctor/logout')
def doctor_logout():
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for('landing'))

if __name__ == '__main__':
    app.run(debug=True)