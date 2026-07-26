import streamlit as st
import time
import os
import random
import json
import pickle
import numpy as np
import nltk
import requests
from datetime import date
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

# --- 1. SETTINGS & AI LOADING ---
st.set_page_config(page_title="Everest AI Terminal", page_icon="🏔️", layout="wide")

# Ensure NLTK data is ready
for pkg in ['punkt', 'punkt_tab', 'wordnet', 'omw-1.4']:
    nltk.download(pkg, quiet=True)

lemmatizer = WordNetLemmatizer()

@st.cache_resource
def load_brain():
    # Paths are simple because everything is in the root
    intents = json.loads(open("chatbot/intents.json", "r", encoding="utf-8").read())
    words = pickle.load(open("model/words.pkl", "rb"))
    classes = pickle.load(open("model/classes.pkl", "rb"))
    model = load_model("model/chatbot_model.keras")
    return intents, words, classes, model

intents, words, classes, model = load_brain()

# --- 2. THE UI THEME (Institutional Purple) ---
st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #fdfbff 0%, #f3e8ff 50%, #e9d5ff 100%); }
    [data-testid="stSidebar"] { background-color: rgba(255,255,255,0.4) !important; border-right: 1px solid rgba(0,0,0,0.05); }
    .chat-bubble { padding: 15px 20px; border-radius: 20px; margin-bottom: 10px; max-width: 80%; line-height: 1.6; }
    .bot-bubble { background: white; color: #1f2937; border: 1px solid rgba(0,0,0,0.05); align-self: flex-start; }
    .user-bubble { background: #7c3aed; color: white; align-self: flex-end; float: right; clear: both; }
    </style>
""", unsafe_allow_html=True)

# --- 3. SIDEBAR UTILITIES ---
with st.sidebar:
    st.markdown("## 🏔️ Summit Status")
    # Weather Logic
    WEATHER_KEY = "1a3620ab60f7bee0f6023aec20abb86f"
    try:
        w_url = f"https://api.openweathermap.org/data/2.5/weather?lat=27.98&lon=86.92&appid={WEATHER_KEY}&units=metric"
        w_data = requests.get(w_url).json()
        st.metric("Base Camp Temp", f"{round(w_data['main']['temp'])}°C", f"{w_data['weather'][0]['description']}")
    except:
        st.write("Weather: Offline")
    
    st.markdown("---")
    days = (date(date.today().year if date.today() < date(date.today().year, 5, 10) else date.today().year + 1, 5, 10) - date.today()).days
    st.write(f"⏳ **{days} Days** to Summit Window")

# --- 4. CHAT LOGIC ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Greetings! I am the Everest Intelligence Agent. How can I assist your expedition today?"}]

def clean_up(s):
    return [lemmatizer.lemmatize(w.lower()) for w in nltk.word_tokenize(s)]

def predict(s):
    bow = [1 if w in clean_up(s) else 0 for w in words]
    res = model.predict(np.array([bow]), verbose=0)[0]
    idx = np.argmax(res)
    return classes[idx] if res[idx] > 0.70 else "fallback"

# Display Messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User Input
if prompt := st.chat_input("Ask me about Everest..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    tag = predict(prompt)
    for i in intents['intents']:
        if i['tag'] == tag:
            response = random.choice(i['responses'])
            break
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.write(response)