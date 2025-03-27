from flask import Flask, render_template, request, jsonify, send_file
from py2neo import Graph  # type: ignore
from fuzzywuzzy import process  # type: ignore
import re
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO
import sqlite3
from datetime import datetime
import torch
from transformers import BertTokenizer, BertForSequenceClassification

app = Flask(__name__)

# SQLite Database Connection
def get_db_connection():
    conn = sqlite3.connect('chat_history.db')
    conn.row_factory = sqlite3.Row
    return conn

# Initialize Database
def initialize_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    # Create tables if they don't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_sessions (
            session_id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_messages (
            message_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER,
            message_text TEXT,
            is_user BOOLEAN,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES chat_sessions (session_id)
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_session_data (
            session_id INTEGER PRIMARY KEY,
            symptoms TEXT,
            FOREIGN KEY (session_id) REFERENCES chat_sessions (session_id)
        )
    ''')

    conn.commit()
    conn.close()

# Initialize the database
initialize_db()

class EnhancedRecommendationEngine:
    def __init__(self):
        try:
            self.graph = Graph("bolt://localhost:7687", auth=("neo4j", "12345678"))
            print("Connected to Neo4j")
        except Exception as e:
            print(f"Neo4j connection error: {e}")
            self.graph = None

        # Load BioBERT model and tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('monologg/biobert_v1.1_pubmed')
        self.model = BertForSequenceClassification.from_pretrained('monologg/biobert_v1.1_pubmed', num_labels=2)
        self.model.eval()  # Set the model to evaluation mode
    
    def recommend(self, query, previous_symptoms=None):
        if not self.graph:
            return {
                "possible_diseases": [],
                "extracted_symptoms": [],
                "all_symptoms": [],
                "next_questions": [],
                "description": "Database connection failure",
                "diagnostic_statement": "Insufficient information due to database connection failure."
            }
        
        # Extract symptoms from current query
        current_symptoms = self.extract_symptoms(query)
        print("Extracted Symptoms from current query:", current_symptoms)
        
        # Combine with previous symptoms if any
        all_symptoms = []
        if previous_symptoms:
            all_symptoms = list(set(previous_symptoms + current_symptoms))
        else:
            all_symptoms = current_symptoms
        
        print("All Symptoms Combined:", all_symptoms)
        
        # Get multiple possible diseases based on symptoms
        possible_diseases = self.recommend_diseases(all_symptoms)
        print("Possible Diseases:", possible_diseases)
        
        # Generate follow-up questions for differential diagnosis
        next_questions = self.generate_follow_up_questions(possible_diseases, all_symptoms)
        
        # Get details for top disease
        top_disease = possible_diseases[0]['disease'] if possible_diseases else "Unknown Disease"
        details = self.get_disease_details(top_disease) if possible_diseases else [{'description': 'No description available.', 'precautions': ['No precautions found.']}]
        
        # Generate diagnosis
        diagnosis = self.generate_diagnosis(possible_diseases, all_symptoms)
        
        return {
            "possible_diseases": possible_diseases,
            "extracted_symptoms": current_symptoms,
            "all_symptoms": all_symptoms,
            "next_questions": next_questions,
            "description": details[0]['description'] if details else "No description available",
            "precautions": details[0]['precautions'] if details else ["No precautions found"],
            "diagnostic_statement": diagnosis
        }

    def extract_symptoms(self, query):
        # Tokenize the input query
        inputs = self.tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=512)
        
        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Process the outputs to extract symptoms
        # For a more realistic implementation, we'll use keyword matching against the database
        symptom_list = self.get_symptom_list()
        
        # Convert query to lowercase for better matching
        query_lower = query.lower()
        extracted_symptoms = []
        
        # Use fuzzy matching to find symptoms in the query
        for symptom in symptom_list:
            if symptom.lower() in query_lower:
                extracted_symptoms.append(symptom)
                continue
                
            # Use fuzzy matching for more complex scenarios
            matches = process.extract(symptom.lower(), [query_lower], limit=1)
            for match, score in matches:
                if score > 80:  # Threshold for fuzzy matching
                    extracted_symptoms.append(symptom)
        
        return list(dict.fromkeys(extracted_symptoms))

    def get_symptom_list(self):
        if not self.graph:
            return []
        
        query = """
        MATCH (s:Symptom)
        RETURN s.name AS symptom
        """
        result = self.graph.run(query).data()
        return [row['symptom'] for row in result]

    def recommend_diseases(self, symptoms):
        if not symptoms or not self.graph:
            return []
        
        query = """
        MATCH (d:Disease)-[:HAS_SYMPTOM]->(s:Symptom)
        WHERE s.name IN $symptoms
        WITH d, COLLECT(DISTINCT s.name) AS matched_symptoms, COUNT(DISTINCT s) AS matched_count
        MATCH (d)-[:HAS_SYMPTOM]->(total:Symptom)
        WITH d, matched_symptoms, matched_count, COUNT(DISTINCT total) AS total_count
        WHERE total_count > 0
        RETURN d.name AS disease, 
               matched_symptoms,
               matched_count, 
               total_count, 
               (matched_count * 1.0 / total_count) AS match_percentage
        ORDER BY match_percentage DESC
        LIMIT 5
        """
        result = self.graph.run(query, symptoms=symptoms).data()
        return result

    def get_disease_details(self, disease_name):
        if not self.graph:
            return [{'description': 'No description available.', 'precautions': ['No precautions found.']}]
        
        query = """
        MATCH (d:Disease {name: $disease_name})
        OPTIONAL MATCH (d)-[:HAS_DESCRIPTION]->(desc:Description)
        OPTIONAL MATCH (d)-[:HAS_PRECAUTION]->(prec:Precaution)
        RETURN 
            COALESCE(desc.text, 'No description available') AS description,
            COLLECT(DISTINCT prec.text) AS precautions
        """
        result = self.graph.run(query, disease_name=disease_name).data()
        return result

    def generate_diagnosis(self, possible_diseases, symptoms):
        if not possible_diseases:
            return "Insufficient symptoms to determine a diagnosis. Please provide more information."
        
        # Create a diagnosis statement with confidence levels
        diagnosis = "Based on the symptoms provided, the following conditions are possible:\n"
        
        for i, disease in enumerate(possible_diseases, 1):
            confidence = disease['match_percentage'] * 100
            confidence_level = "High" if confidence > 80 else "Moderate" if confidence > 50 else "Low"
            
            diagnosis += f"{i}. {disease['disease']} ({confidence_level} confidence: {confidence:.1f}%)\n"
            diagnosis += f"   Matched symptoms: {', '.join(disease['matched_symptoms'])}\n"
        
        diagnosis += "\nMore information is needed for a conclusive diagnosis. Please answer the follow-up questions."
        return diagnosis

    def generate_follow_up_questions(self, possible_diseases, current_symptoms):
        if not self.graph or not possible_diseases:
            return ["Are you experiencing any other symptoms?"]
        
        # Get distinctive symptoms for the top diseases
        distinctive_symptoms = self.get_distinctive_symptoms(possible_diseases, current_symptoms)
        
        # Convert to questions
        questions = []
        for symptom in distinctive_symptoms[:3]:  # Limit to top 3 questions
            questions.append(f"Are you experiencing {symptom}?")
        
        # Add a general question if we don't have enough specific ones
        if len(questions) < 3:
            questions.append("How long have you been experiencing these symptoms?")
            questions.append("Do you have any pre-existing medical conditions?")
        
        return questions

    def get_distinctive_symptoms(self, possible_diseases, current_symptoms):
        distinctive_symptoms = []
        
        # Get all symptoms for the top diseases
        top_diseases = [disease['disease'] for disease in possible_diseases[:3]]
        
        query = """
        MATCH (d:Disease)-[:HAS_SYMPTOM]->(s:Symptom)
        WHERE d.name IN $diseases AND NOT s.name IN $current_symptoms
        RETURN DISTINCT s.name AS symptom
        LIMIT 10
        """
        
        result = self.graph.run(query, diseases=top_diseases, current_symptoms=current_symptoms).data()
        return [row['symptom'] for row in result]

recommendation_engine = EnhancedRecommendationEngine()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_recommendation', methods=['POST'])
def get_recommendation():
    query = request.json.get('message', '')
    session_id = request.json.get('session_id', None)
    
    try:
        # Get previous symptoms for this session if it exists
        previous_symptoms = []
        if session_id:
            previous_symptoms = get_previous_symptoms(session_id)
        
        # Get recommendations with the enhanced engine
        result = recommendation_engine.recommend(query, previous_symptoms)
        
        # Extract all the data from the result
        possible_diseases = result.get('possible_diseases', [])
        all_symptoms = result.get('all_symptoms', [])
        next_questions = result.get('next_questions', [])
        description = result.get('description', 'No description available.')
        precautions = result.get('precautions', ['No precautions found.'])
        diagnostic_statement = result.get('diagnostic_statement', 'Insufficient information.')
        
        # Top disease information
        top_disease = possible_diseases[0]['disease'] if possible_diseases else 'Unable to determine'
        
        # Create a structured response with all information
        response_data = {
            'disease': top_disease,
            'possible_diseases': possible_diseases,
            'description': description,
            'diagnostic_statement': diagnostic_statement,
            'precautions': precautions,
            'extracted_symptoms': all_symptoms,
            'next_questions': next_questions
        }

        # Format the response for display in the chat
        formatted_response = f"""<div class="response-container">
            <p><strong>Extracted Symptoms:</strong> {', '.join(all_symptoms)}</p>
            <p><strong>Possible Diseases:</strong></p>
            <ul>
                {"".join([f"<li>{disease['disease']} (Confidence: {disease['match_percentage']*100:.1f}%)</li>" for disease in possible_diseases[:3]])}
            </ul>
            <p><strong>Top Disease:</strong> {top_disease}</p>
            <p><strong>Description:</strong> {description}</p>
            <p><strong>Diagnostic Statement:</strong> {diagnostic_statement}</p>
            <p><strong>Precautions:</strong></p>
            <ul>
                {"".join([f"<li>{precaution}</li>" for precaution in precautions])}
            </ul>
            <p><strong>Follow-up Questions:</strong></p>
            <ul>
                {"".join([f"<li>{question}</li>" for question in next_questions])}
            </ul>
        </div>"""

        # Save chat to database
        session_id = save_chat_to_db(query, formatted_response, session_id, all_symptoms)
        
        # Add session_id to response
        response_data['session_id'] = session_id
        response_data['formatted_response'] = formatted_response
        
        return jsonify(response_data)
    
    except Exception as e:
        print(f"Error: {e}")
        error_response = f"<div class='response-container'><p>Error: {str(e)}</p></div>"
        session_id = save_chat_to_db(query, error_response, session_id)
        return jsonify({
            'disease': 'Error',
            'description': str(e),
            'diagnostic_statement': 'An error occurred.',
            'precautions': [],
            'session_id': session_id,
            'formatted_response': error_response
        })

@app.route('/download_report', methods=['POST'])
def download_report():
    data = request.json
    disease = data.get('disease', 'Unknown Disease')
    description = data.get('description', 'No description available.')
    diagnostic_statement = data.get('diagnostic_statement', 'No diagnostic statement available.')
    precautions = data.get('precautions', ['No precautions available.'])
    extracted_symptoms = data.get('extracted_symptoms', [])
    possible_diseases = data.get('possible_diseases', [])

    buffer = BytesIO()
    pdf = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Title
    title = Paragraph("Medical Recommendation Report", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 12))
    
    # Date
    date_paragraph = Paragraph(f"<b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['BodyText'])
    story.append(date_paragraph)
    story.append(Spacer(1, 12))
    
    # Symptoms
    if extracted_symptoms:
        symptoms_paragraph = Paragraph(f"<b>Reported Symptoms:</b> {', '.join(extracted_symptoms)}", styles['BodyText'])
        story.append(symptoms_paragraph)
        story.append(Spacer(1, 12))

    # Primary Disease
    disease_paragraph = Paragraph(f"<b>Primary Recommendation:</b> {disease}", styles['BodyText'])
    story.append(disease_paragraph)
    story.append(Spacer(1, 12))
    
    # Other Possible Diseases
    if possible_diseases and len(possible_diseases) > 1:
        other_diseases_paragraph = Paragraph("<b>Other Possible Conditions:</b>", styles['BodyText'])
        story.append(other_diseases_paragraph)
        story.append(Spacer(1, 6))
        
        for i, disease_data in enumerate(possible_diseases[1:4], 2):  # Start from the second disease
            disease_name = disease_data.get('disease', 'Unknown')
            confidence = disease_data.get('match_percentage', 0) * 100
            disease_item = Paragraph(f"{i-1}. {disease_name} (Confidence: {confidence:.1f}%)", styles['BodyText'])
            story.append(disease_item)
            
        story.append(Spacer(1, 12))

    # Description
    description_paragraph = Paragraph(f"<b>Description:</b> {description}", styles['BodyText'])
    story.append(description_paragraph)
    story.append(Spacer(1, 12))

    # Diagnostic Statement
    diagnostic_paragraph = Paragraph(f"<b>Diagnostic Statement:</b> {diagnostic_statement}", styles['BodyText'])
    story.append(diagnostic_paragraph)
    story.append(Spacer(1, 12))

    # Precautions
    precautions_paragraph = Paragraph("<b>Precautions:</b>", styles['BodyText'])
    story.append(precautions_paragraph)
    for precaution in precautions:
        precaution_paragraph = Paragraph(f"- {precaution}", styles['BodyText'])
        story.append(precaution_paragraph)
        story.append(Spacer(1, 6))
    
    # Disclaimer
    story.append(Spacer(1, 24))
    disclaimer = Paragraph("<i>Disclaimer: This report is generated by an AI system and is for informational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.</i>", styles['BodyText'])
    story.append(disclaimer)

    pdf.build(story)
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name='medical_report.pdf', mimetype='application/pdf')

@app.route('/get_chat_history', methods=['GET'])
def get_chat_history():
    conn = get_db_connection()
    cursor = conn.cursor()

    # Fetch all chat sessions
    cursor.execute('SELECT * FROM chat_sessions ORDER BY created_at DESC')
    sessions = cursor.fetchall()

    # Fetch messages for each session
    chat_history = []
    for session in sessions:
        cursor.execute('SELECT * FROM chat_messages WHERE session_id = ? ORDER BY created_at', (session['session_id'],))
        messages = cursor.fetchall()

        # Convert sqlite3.Row objects to dictionaries
        session_dict = dict(session)
        messages_dict = [dict(message) for message in messages]

        # Get the first message as a preview
        preview = "Empty chat" if not messages else messages_dict[0]['message_text'][:30] + "..."

        chat_history.append({
            'session_id': session_dict['session_id'],
            'created_at': session_dict['created_at'],
            'preview': preview,
            'messages': messages_dict
        })

    conn.close()
    return jsonify(chat_history)

def get_previous_symptoms(session_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get the symptoms from the session metadata
    cursor.execute('SELECT symptoms FROM chat_session_data WHERE session_id = ?', (session_id,))
    result = cursor.fetchone()
    
    conn.close()
    
    if result and result['symptoms']:
        return result['symptoms'].split(',')
    return []

def save_chat_to_db(query, response, session_id=None, symptoms=None):
    conn = get_db_connection()
    cursor = conn.cursor()

    # Convert datetime to string
    created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Create a new chat session if session_id is not provided
    if not session_id:
        cursor.execute('INSERT INTO chat_sessions (created_at) VALUES (?)', (created_at,))
        session_id = cursor.lastrowid
        
        # Create metadata table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_session_data (
                session_id INTEGER PRIMARY KEY,
                symptoms TEXT,
                FOREIGN KEY (session_id) REFERENCES chat_sessions (session_id)
            )
        ''')

    # Save user message
    cursor.execute('INSERT INTO chat_messages (session_id, message_text, is_user) VALUES (?, ?, ?)',
                   (session_id, query, True))

    # Save bot response
    cursor.execute('INSERT INTO chat_messages (session_id, message_text, is_user) VALUES (?, ?, ?)',
                   (session_id, response, False))
    
    # Update session data with symptoms
    if symptoms:
        symptoms_str = ','.join(symptoms)
        
        # Check if session data already exists
        cursor.execute('SELECT * FROM chat_session_data WHERE session_id = ?', (session_id,))
        if cursor.fetchone():
            cursor.execute('UPDATE chat_session_data SET symptoms = ? WHERE session_id = ?', 
                          (symptoms_str, session_id))
        else:
            cursor.execute('INSERT INTO chat_session_data (session_id, symptoms) VALUES (?, ?)',
                          (session_id, symptoms_str))

    conn.commit()
    conn.close()

    return session_id

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001, debug=True)