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
from transformers import RobertaTokenizer, TFRobertaModel
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.models import Model
from flask import Flask, request, jsonify

# Configure logging
logging.basicConfig(filename='doj_chatbot.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Set up the API key
GEMINI_API_KEY = 'AIzaSyBUN9ZHUnyW9rI2Y12CpywGf0ZNeOXL3-8'
genai.configure(api_key=GEMINI_API_KEY)

# Define the model
try:
    model = genai.GenerativeModel('gemini-pro')
except Exception as e:
    logging.error(f"Failed to initialize Gemini model: {str(e)}")
    raise

# Initialize Flask app
app = Flask(__name__)

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

def sanitize_input(input_string: str) -> str:
    return re.sub(r'[^\w\s\-\.,?!]', '', input_string)

# RoBERTa, Q-Learning, and Improvement Model Parameters
MAX_LENGTH = 128
IMPROVEMENT_MODEL_OUTPUT_SIZE = 50
LEARNING_RATE = 0.001
BATCH_SIZE = 32
MEMORY_SIZE = 2000
GAMMA = 0.95
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
ACTION_SIZE = 5

memory = deque(maxlen=MEMORY_SIZE)

# Initialize RoBERTa tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
roberta_model = TFRobertaModel.from_pretrained('roberta-base')

def get_roberta_embedding(text):
    inputs = tokenizer(text, return_tensors='tf', max_length=MAX_LENGTH, padding='max_length', truncation=True)
    outputs = roberta_model(inputs)
    return outputs.last_hidden_state[:, 0, :]  # Use the [CLS] token embedding

def build_q_model():
    input_layer = Input(shape=(768,))  # RoBERTa embedding size
    hidden1 = Dense(256, activation='relu')(input_layer)
    hidden2 = Dense(128, activation='relu')(hidden1)
    output_layer = Dense(ACTION_SIZE, activation='linear')(hidden2)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mse')
    return model

q_model = build_q_model()

def build_improvement_model():
    query_input = Input(shape=(768,))  # RoBERTa embedding size
    action_input = Input(shape=(ACTION_SIZE,))
    feedback_input = Input(shape=(1,))
    
    combined = Concatenate()([query_input, action_input, feedback_input])
    hidden1 = Dense(256, activation='relu')(combined)
    hidden2 = Dense(128, activation='relu')(hidden1)
    output = Dense(IMPROVEMENT_MODEL_OUTPUT_SIZE, activation='tanh')(hidden2)
    
    model = Model(inputs=[query_input, action_input, feedback_input], outputs=output)
    model.compile(optimizer='adam', loss='mse')
    return model

improvement_model = build_improvement_model()

def choose_action(state):
    if np.random.rand() <= EPSILON:
        return random.randrange(ACTION_SIZE)
    q_values = q_model.predict(state)
    return np.argmax(q_values[0])

def generate_improved_prompt(query: str, action: int, feedback: float) -> str:
    query_embedding = get_roberta_embedding(query)
    action_vector = tf.one_hot(action, ACTION_SIZE)
    
    improvement_vector = improvement_model.predict([
        query_embedding,
        np.expand_dims(action_vector, axis=0),
        np.array([[feedback]])
    ])[0]
    
    improvement_prompt = " ".join([f"Adjust_{i}:{value:.2f}" for i, value in enumerate(improvement_vector)])
    return f"Based on feedback, please adjust your response as follows: {improvement_prompt}"

def get_chatbot_response(user_input: str, action: int) -> str:
    try:
        improvement_prompt = generate_improved_prompt(user_input, action, 0)  # Initial feedback of 0
        
        enhanced_input = f"{user_input}\n\n{improvement_prompt}"
        
        response = chat.send_message(enhanced_input).text
        
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": response})
        save_chat_history(chat_history)
        
        return response
    except Exception as e:
        logging.error(f"Error getting chatbot response: {str(e)}")
        return "I apologize, but I'm having trouble processing your request right now. Please try again later."

def update_models(state, action, reward, next_state):
    global EPSILON
    
    target = reward + GAMMA * np.amax(q_model.predict(next_state)[0])
    target_f = q_model.predict(state)
    target_f[0][action] = target
    q_model.fit(state, target_f, epochs=1, verbose=0)
    
    action_vector = tf.one_hot(action, ACTION_SIZE)
    
    # For simplicity, we're using random targets. In a real scenario, you'd want to derive meaningful targets.
    target = np.random.randn(IMPROVEMENT_MODEL_OUTPUT_SIZE)
    
    improvement_model.fit(
        [state, np.expand_dims(action_vector, axis=0), np.array([[reward]])],
        np.expand_dims(target, axis=0),
        verbose=0
    )
    
    if EPSILON > EPSILON_MIN:
        EPSILON *= EPSILON_DECAY

# Initialize the chat
chat = model.start_chat(history=[])
chat.send_message(SYSTEM_PROMPT)

@app.route('/chat', methods=['POST'])
def process_request():
    try:
        data = request.json
        user_input = data.get('message', '')
        user_input = sanitize_input(user_input)

        if not is_doj_related(user_input):
            return jsonify({
                "response": "I apologize, but I can only provide information related to the Department of Justice, India. Could you please ask a question about DoJ services, court processes, or the Indian legal system?"
            })

        state = get_roberta_embedding(user_input)
        action = choose_action(state)
        
        response = get_chatbot_response(user_input, action)
        
        # Note: We can't get real-time feedback in an API setting, so we'll skip the feedback and model update steps

        return jsonify({"response": response})

    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        return jsonify({"error": "An error occurred while processing your request"}), 500

if __name__ == "__main__":
    try:
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        logging.critical(f"Critical error in main execution: {str(e)}")
        print("A critical error occurred. Please check the logs and try again later.")