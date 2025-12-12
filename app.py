import os
import time
import uuid
import json
import logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import google.generativeai as genai
from dotenv import load_dotenv
from functools import wraps

# --- CONFIGURATION ---
load_dotenv()
app = Flask(__name__)
# Secret key is required for session management (Doctor Login) and flash messages
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev_secret_key_123")

# Database Config (SQLite for local persistence)
# This will create a file named 'medical_platform.db' in your project folder
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///medical_platform.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    logging.warning("WARNING: GEMINI_API_KEY not found in .env")

genai.configure(api_key=api_key)

# --- DATABASE MODELS ---

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    role = db.Column(db.String(20), nullable=False) # 'patient' or 'doctor'
    full_name = db.Column(db.String(100))
    
    # Doctor specific fields
    specialty = db.Column(db.String(100), nullable=True)
    doctor_unique_id = db.Column(db.String(50), unique=True, nullable=True) # The "License ID"

class ClinicalCase(db.Model):
    id = db.Column(db.String(10), primary_key=True) # The short UUID
    # Foreign Keys link to the User table
    patient_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    doctor_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Data stored as JSON strings (flexible for AI output)
    raw_data_json = db.Column(db.Text, nullable=False)
    ai_analysis_json = db.Column(db.Text, nullable=False)
    status = db.Column(db.String(20), default="Pending")

# --- INITIALIZATION & SEEDING ---
with app.app_context():
    db.create_all()
    # Check if DB is empty, if so, seed it
    if not User.query.first():
        print("Seeding database with demo data...")
        
        # Demo Patients
        p1 = User(username='patient1', password_hash=generate_password_hash('p123'), role='patient', full_name='John Doe')
        p2 = User(username='patient2', password_hash=generate_password_hash('p123'), role='patient', full_name='Jane Smith')
        p3 = User(username='patient3', password_hash=generate_password_hash('p123'), role='patient', full_name='Alice Wonderland')
        
        # Demo Doctors
        d1 = User(username='dr_smith', password_hash=generate_password_hash('pass_smith'), role='doctor', full_name='Dr. Sarah Smith', specialty='General Physician', doctor_unique_id='DOC-001')
        d2 = User(username='dr_patel', password_hash=generate_password_hash('pass_patel'), role='doctor', full_name='Dr. Raj Patel', specialty='Cardiologist', doctor_unique_id='DOC-002')
        d3 = User(username='dr_lee', password_hash=generate_password_hash('pass_lee'), role='doctor', full_name='Dr. Emily Lee', specialty='Dermatologist', doctor_unique_id='DOC-003')
        
        db.session.add_all([p1, p2, p3, d1, d2, d3])
        db.session.commit()
        print("Database seeded successfully.")

# --- AI PROMPT ---
SYSTEM_PROMPT = """
ACT AS: Senior Clinical Consultant & Medical Scribe.
TASK: Analyze patient intake data and generate a structured clinical case file.

LANGUAGE INSTRUCTION: 
- "patient_view" MUST be in {language}. (Translate concepts to be culturally relevant).
- "doctor_view" MUST be in ENGLISH (Standard Medical Terminology).

OUTPUT FORMAT: Return ONLY valid JSON matching this schema:
{{
  "patient_view": {{
    "summary": "Warm, reassuring explanation in {language}.",
    "pathophysiology": "Simple analogy explaining the mechanism in {language}.",
    "care_plan": ["Step 1 in {language}", "Step 2 in {language}"],
    "red_flags": ["Urgent sign 1 in {language}", "Urgent sign 2 in {language}"]
  }},
  "doctor_view": {{
    "subjective": "Professional medical terminology summary of HPI in ENGLISH.",
    "objective": "Concise summary of reported vitals in ENGLISH.",
    "assessment": "Differential diagnosis ranked by probability in ENGLISH.",
    "plan": "Suggested pharmacotherapy, diagnostics, and follow-up in ENGLISH."
  }},
  "safety": {{
    "is_safe": true,
    "warnings": []
  }}
}}

SAFETY RULES:
- Check for Drug-Allergy interactions.
- Check for Contraindications based on age/history.
- If unsafe, set "is_safe": false and add warnings (in English).
"""

# --- HELPER FUNCTIONS ---

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
    """Remove markdown brackets and format text for readability."""
    if not text: return ""
    import re
    text = re.sub(r'\[\*\*', '', text)
    text = re.sub(r'\*\*\]', '', text)
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    return text.strip()

# --- ROUTES ---

@app.route('/')
def landing():
    """Landing page - role selection."""
    if 'user_id' in session:
        if session['role'] == 'patient':
            return redirect(url_for('patient_intake'))
        elif session['role'] == 'doctor':
            return redirect(url_for('doctor_dashboard'))
    return render_template('landing.html')

# --- AUTHENTICATION ROUTES (New) ---

@app.route('/register/<role>', methods=['GET', 'POST'])
def register(role):
    if role not in ['patient', 'doctor']:
        return redirect(url_for('landing'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        name = request.form.get('name')
        
        # Check if user exists
        if User.query.filter_by(username=username).first():
            flash("Username already exists! Please choose another.", "danger")
            return render_template('register.html', role=role)
            
        # Create User Object
        new_user = User(
            username=username, 
            password_hash=generate_password_hash(password), 
            role=role, 
            full_name=name
        )
        
        # Doctor Specific Logic
        if role == 'doctor':
            doctor_id = request.form.get('doctor_id')
            specialty = request.form.get('specialty')
            
            # Simple Validation: Must be somewhat unique or follow a pattern
            if not doctor_id or len(doctor_id) < 3:
                flash("Invalid Medical License ID.", "danger")
                return render_template('register.html', role=role)
                
            if User.query.filter_by(doctor_unique_id=doctor_id).first():
                flash("Doctor ID already registered.", "danger")
                return render_template('register.html', role=role)
                
            new_user.doctor_unique_id = doctor_id
            new_user.specialty = specialty
            
        try:
            db.session.add(new_user)
            db.session.commit()
            flash(f"Account created successfully! Please login.", "success")
            
            # Redirect to the correct login page
            if role == 'patient':
                return redirect(url_for('patient_login'))
            else:
                return redirect(url_for('doctor_login'))
                
        except Exception as e:
            db.session.rollback()
            logging.error(f"Registration Error: {e}")
            flash("An error occurred during registration.", "danger")
            
    return render_template('register.html', role=role)

# --- PATIENT ROUTES ---

@app.route('/patient/login', methods=['GET', 'POST'])
def patient_login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        # Query DB instead of dict
        user = User.query.filter_by(username=username, role='patient').first()
        
        if user and check_password_hash(user.password_hash, password):
            session['user_id'] = user.id
            session['role'] = 'patient'
            session['name'] = user.full_name
            return redirect(url_for('patient_intake'))
        else:
            flash("Invalid username or password", "danger")
    
    return render_template('patient_login.html')

@app.route('/patient/intake')
@login_required
@patient_required
def patient_intake():
    # Fetch all doctors from DB for the dropdown
    doctors = User.query.filter_by(role='doctor').all()
    # Convert to list of dicts for template
    doctor_list = [{"id": d.id, "name": d.full_name, "specialty": d.specialty} for d in doctors]
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
            doctors = User.query.filter_by(role='doctor').all()
            doctor_list = [{"id": d.id, "name": d.full_name, "specialty": d.specialty} for d in doctors]
            return render_template('intake.html', doctors=doctor_list, error="Please select a doctor")
            
        doctor_id = int(doctor_id_str)
        doctor = db.session.get(User, doctor_id)
        
        # Collect Data
        raw_data = {
            "id": case_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "patient_name": session.get('name'),
            "doctor_name": doctor.full_name if doctor else "Unknown",
            "name": request.form.get('name'),
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

        # AI Processing
        model = genai.GenerativeModel("gemini-2.5-flash", generation_config={"response_mime_type": "application/json"})
        formatted_prompt = SYSTEM_PROMPT.format(language=selected_language)
        
        prompt = f"""
        {formatted_prompt}
        PATIENT DATA: {json.dumps(raw_data, default=str)}
        """
        
        response = model.generate_content(prompt)
        ai_analysis = json.loads(response.text)

        # Cleanup AI text
        if 'patient_view' in ai_analysis:
            ai_analysis['patient_view']['summary'] = clean_medical_text(ai_analysis['patient_view']['summary'])
            # ... (other cleaning logic if needed)

        # Save to SQLite Database
        new_case = ClinicalCase(
            id=case_id,
            patient_id=session['user_id'],
            doctor_id=doctor_id,
            raw_data_json=json.dumps(raw_data),
            ai_analysis_json=json.dumps(ai_analysis),
            status="Pending Review"
        )
        db.session.add(new_case)
        db.session.commit()
        
        return redirect(url_for('patient_result', case_id=case_id))

    except Exception as e:
        logging.error(f"Error processing case: {e}")
        doctors = User.query.filter_by(role='doctor').all()
        doctor_list = [{"id": d.id, "name": d.full_name, "specialty": d.specialty} for d in doctors]
        return render_template('intake.html', doctors=doctor_list, error=f"System Error: {str(e)}")

@app.route('/patient/result/<case_id>')
@login_required
@patient_required
def patient_result(case_id):
    # Fetch from DB
    case = db.session.get(ClinicalCase, case_id)
    
    if not case:
        return "Case not found.", 404
        
    # Security: Only show if patient owns this case
    if case.patient_id != session['user_id']:
        return "Access Denied", 403
        
    # Convert stored JSON strings back to dicts for template
    case_data = {
        "raw_data": json.loads(case.raw_data_json),
        "ai_analysis": json.loads(case.ai_analysis_json)
    }
    
    return render_template('patient_result.html', case=case_data)

@app.route('/patient/logout')
def patient_logout():
    session.clear()
    flash("You have been logged out successfully", "success")
    return redirect(url_for('landing'))

# --- DOCTOR ROUTES ---

@app.route('/doctor/login', methods=['GET', 'POST'])
def doctor_login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username, role='doctor').first()
        
        if user and check_password_hash(user.password_hash, password):
            session['user_id'] = user.id
            session['role'] = 'doctor'
            session['name'] = user.full_name
            return redirect(url_for('doctor_dashboard'))
        else:
            flash("Invalid username or password", "danger")
    
    return render_template('doctor_login.html')

@app.route('/doctor/dashboard')
@login_required
@doctor_required
def doctor_dashboard():
    doctor_id = session.get('user_id')
    
    # Query DB for cases assigned to this doctor
    cases_db = ClinicalCase.query.filter_by(doctor_id=doctor_id).order_by(ClinicalCase.timestamp.desc()).all()
    
    # Prepare data for template
    cases_list = []
    for c in cases_db:
        raw = json.loads(c.raw_data_json)
        ai = json.loads(c.ai_analysis_json)
        cases_list.append({
            "case_id": c.id, # Use 'case_id' to match template expectation
            "raw_data": raw,
            "ai_analysis": ai
        })
    
    # Get doctor info for header
    doctor = db.session.get(User, doctor_id)
    doctor_info = {"name": doctor.full_name, "specialty": doctor.specialty}
    
    return render_template('doctor_dashboard.html', cases=cases_list, doctor=doctor_info)

@app.route('/doctor/view/<case_id>')
@login_required
@doctor_required
def doctor_view(case_id):
    doctor_id = session.get('user_id')
    case = db.session.get(ClinicalCase, case_id)
    
    if not case:
        return "Case not found", 404
    
    # Security Check
    if case.doctor_id != doctor_id:
        flash("You don't have access to this case", "danger")
        return redirect(url_for('doctor_dashboard'))
    
    case_data = {
        "raw_data": json.loads(case.raw_data_json),
        "ai_analysis": json.loads(case.ai_analysis_json)
    }
    
    return render_template('doctor_view.html', case=case_data)

@app.route('/doctor/logout')
def doctor_logout():
    session.clear()
    flash("You have been logged out successfully", "success")
    return redirect(url_for('landing'))

if __name__ == '__main__':
    app.run(debug=True)