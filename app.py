import google.generativeai as genai
import tensorflow as tf
import numpy as np
import random
import logging
import json
from collections import deque
from typing import List, Dict, Optional
from datetime import datetime
import re
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

# Configure logging
logging.basicConfig(filename='doj_chatbot.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Set up the API key
GEMINI_API_KEY = 'AIzaSyAH-EUUy-ZfWR3P6rfI-cxKVauy4o3TzQU'

genai.configure(api_key=GEMINI_API_KEY)

# Define the model
try:
    model = genai.GenerativeModel('gemini-pro')
except Exception as e:
    logging.error(f"Failed to initialize Gemini model: {str(e)}")
    raise

# Initialize the chat
chat = model.start_chat(history=[])

# Define the initial system message
SYSTEM_PROMPT = """You are an AI assistant for the Department of Justice (DoJ), India. Your role is to provide accurate and helpful information about the following topics:
1. Divisions of DoJ
2. Judge appointments and vacancies in Supreme Court, High Courts, District & Subordinate Courts
3. Case pendency through National Judicial Data Grid (NJDG)
4. Procedure to pay fines for traffic violations
5. Live streaming of court cases
6. Steps for eFiling and ePay
7. Information about Fast Track Courts
8. How to download and use the eCourts Services Mobile app
9. Availing Tele Law Services
10. Checking current status of cases

If a user asks about topics unrelated to the DoJ or the Indian legal system, politely inform them that you can only assist with DoJ-related inquiries and redirect them to relevant DoJ topics. Do not provide information on subjects outside your designated scope.

Provide concise and accurate responses based on the latest available information. If you're unsure about any information, please state that and suggest where the user might find the most up-to-date details.

When providing numerical data, always include a disclaimer that the information may not be current and direct users to official sources for the most up-to-date figures."""

# Mock database for demonstration purposes
MOCK_DB = {
    "judge_vacancies": {
        "supreme_court": 2,
        "high_courts": 123,
        "district_courts": 456
    },
    "case_pendency": {
        "supreme_court": 70254,
        "high_courts": 5986290,
        "district_courts": 41401487
    }
}

def is_doj_related(query: str) -> bool:
    doj_keywords = [
        'doj', 'department of justice', 'ministry of law and justice',
        'judge', 'court', 'case', 'legal', 'judiciary', 'judicial',
        'fine', 'penalty', 'challan',
        'efiling', 'e-filing', 'epay', 'e-pay',
        'tele law', 'telelaw',
        'njdg', 'national judicial data grid',
        'ecourts', 'e-courts',
        'sc', 'supreme court',
        'hc', 'high court',
        'district court', 'subordinate court',
        'tribunal', 'bench',
        'lawyer', 'advocate', 'attorney',
        'litigation', 'lawsuit',
        'plea', 'petition', 'affidavit',
        'verdict', 'judgment', 'ruling',
        'cji', 'chief justice',
        'bar council', 'bar association',
        'legal aid', 'pro bono',
        'bail', 'parole', 'probation',
        'fir', 'first information report',
        'ipc', 'indian penal code',
        'cpc', 'civil procedure code',
        'crpc', 'criminal procedure code',
        'pil', 'public interest litigation',
        'lok adalat', 'gram nyayalaya',
        'alternative dispute resolution', 'adr', 'arbitration', 'mediation',
        'writ', 'habeas corpus', 'mandamus', 'certiorari', 'quo warranto',
        'constitution', 'constitutional',
        'fundamental rights', 'directive principles',
        'jurisdiction', 'appellate',
        'prosecution', 'defence', 'defense',
        'witness', 'testimony', 'evidence',
        'oath', 'sworn statement',
        'court fee', 'stamp duty',
        'judicial review', 'contempt of court' , 'disputes', 'e-committees'
    ]
    return any(keyword in query.lower() for keyword in doj_keywords)

def load_chat_history() -> List[Dict]:
    try:
        with open('chat_history.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []
    except json.JSONDecodeError:
        logging.error("Error decoding chat history JSON. Starting with empty history.")
        return []

def save_chat_history(history: List[Dict]):
    with open('chat_history.json', 'w') as f:
        json.dump(history, f)

chat_history = load_chat_history()

def get_chatbot_response(user_input: str) -> str:
    try:
        response = chat.send_message(user_input)
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": response.text})
        save_chat_history(chat_history)
        return response.text
    except Exception as e:
        logging.error(f"Error getting chatbot response: {str(e)}")
        return "I apologize, but I'm having trouble processing your request right now. Please try again later."

def handle_specific_query(query: str) -> Optional[str]:
    query = query.lower()
    
    if "judge vacancies" in query:
        vacancies = MOCK_DB["judge_vacancies"]
        return f"Based on our last update, there were approximately {vacancies['supreme_court']} vacancies in the Supreme Court, {vacancies['high_courts']} in High Courts, and {vacancies['district_courts']} in District & Subordinate Courts. Please note that these figures may not be current. For the most up-to-date information, please check the official DoJ website."
    
    elif "case pendency" in query:
        pendency = MOCK_DB["case_pendency"]
        return f"As per our last update from the National Judicial Data Grid (NJDG), there were approximately {pendency['supreme_court']} pending cases in the Supreme Court, {pendency['high_courts']} in High Courts, and {pendency['district_courts']} in District & Subordinate Courts. Please note that these figures change daily. For real-time data, please visit the NJDG website at https://njdg.ecourts.gov.in/"
    
    elif "traffic fine" in query:
        return "To pay a traffic violation fine, you can follow these steps:\n1. Visit the official e-Challan website: https://echallan.parivahan.gov.in/\n2. Enter your vehicle number or challan number\n3. View the challan details\n4. Pay the fine using various online payment methods\nAlternatively, you can also pay fines at designated traffic police stations or courts."
    
    elif "efilin" in query or "epay" in query:
        return "For eFiling and ePay services, please visit https://efiling.ecourts.gov.in/. The platform allows you to file cases electronically and make online payments for court fees and other charges. For a step-by-step guide, you can refer to the user manual available on the website."
    
    elif "ecourts app" in query:
        return "To download the eCourts Services Mobile App:\n1. For Android: Visit Google Play Store and search for 'eCourts Services'\n2. For iOS: Visit Apple App Store and search for 'eCourts Services'\n3. Download and install the app\n4. Open the app and start using various services like checking case status, cause lists, court orders, etc."
    
    elif "tele law" in query:
        return "Tele-Law service provides legal advice to people in rural areas through video conferencing. To avail this service:\n1. Visit your nearest Common Service Centre (CSC)\n2. Register for a Tele-Law consultation\n3. Get connected with a panel lawyer for free legal advice\nFor more information, visit https://www.tele-law.in/"
    
    return None

def sanitize_input(input_string: str) -> str:
    return re.sub(r'[^\w\s\-\.,?!]', '', input_string)

# Initialize Flask app
app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/chat', methods=['POST'])
@cross_origin()
def chat_endpoint():
    data = request.json
    user_input = data.get('message', '')
    user_input = sanitize_input(user_input)

    # Log the user input
    logging.info(f"User Input: {user_input}")

    if not is_doj_related(user_input):
        response = "I apologize, but I can only provide information related to the Department of Justice, India. Could you please ask a question about DoJ services, court processes, or the Indian legal system?"
        logging.info("Bot Response: Off-topic query redirected")
    else:
        # Check for specific queries first
        specific_response = handle_specific_query(user_input)
        if specific_response:
            response = specific_response
            logging.info(f"Bot Response (Specific): {response}")
        else:
            response = get_chatbot_response(user_input)
            logging.info(f"Bot Response (General): {response}")

    return jsonify({"response": response})

if __name__ == "__main__":
    try:
        # Initialize the chat with the system prompt
        chat.send_message(SYSTEM_PROMPT)
        app.run(debug=True)
    except Exception as e:
        logging.critical(f"Critical error in main execution: {str(e)}")
        print("A critical error occurred. Please check the logs and try again later.")