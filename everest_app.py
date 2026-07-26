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
    # Using the stable .h5 format we created
    intents = json.loads(open("chatbot/intents.json", "r", encoding="utf-8").read())
    words = pickle.load(open("model/words.pkl", "rb"))
    classes = pickle.load(open("model/classes.pkl", "rb"))
    model = load_model("model/chatbot_model.h5") # Ensure this is .h5
    return intents, words, classes, model

try:
    intents, words, classes, model = load_brain()
except:
    st.error("AI Brain synchronization in progress. Please refresh in 10 seconds.")
    st.stop()

# --- 2. THE PREMIUM UI OVERHAUL (CSS Injection) ---
st.markdown("""
    <style>
    /* 1. Global Background */
    .stApp {
        background: linear-gradient(135deg, #fdfbff 0%, #f3e8ff 50%, #e9d5ff 100%);
    }

    /* 2. Glassmorphism Chat Container */
    .chat-container {
        padding: 20px;
        display: flex;
        flex-direction: column;
        gap: 20px;
    }

    /* 3. Message Bubbles - EXACT MATCH TO BEFORE */
    .msg-box {
        max-width: 80%;
        padding: 16px 20px;
        border-radius: 24px;
        font-size: 15px;
        line-height: 1.6;
        margin-bottom: 20px;
        font-family: 'Inter', sans-serif;
        position: relative;
    }
    
    .label {
        display: block;
        font-size: 9px;
        font-weight: 800;
        margin-bottom: 6px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .user-msg {
        align-self: flex-end;
        background: #7c3aed;
        color: white;
        border-bottom-right-radius: 4px;
        box-shadow: 0 10px 20px rgba(124, 58, 237, 0.2);
        margin-left: auto; /* Force to Right */
    }
    .user-msg .label { color: rgba(255,255,255,0.7); }

    .bot-msg {
        align-self: flex-start;
        background: white;
        color: #1f2937;
        border: 1px solid rgba(0,0,0,0.05);
        border-bottom-left-radius: 4px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.03);
    }
    .bot-msg .label { color: #7c3aed; }

    /* 4. Hide default Streamlit chat icons */
    [data-testid="stChatMessage"] { background-color: transparent !important; }
    
    /* 5. Sidebar Styling */
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.4) !important;
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. SIDEBAR (The "Status Center") ---
with st.sidebar:
    st.markdown("### 🏔️ Summit Status")
    WEATHER_KEY = "1a3620ab60f7bee0f6023aec20abb86f"
    try:
        w_url = f"https://api.openweathermap.org/data/2.5/weather?lat=27.98&lon=86.92&appid={WEATHER_KEY}&units=metric"
        w_data = requests.get(w_url).json()
        st.metric("Base Camp Temp", f"{round(w_data['main']['temp'])}°C", f"{w_data['weather'][0]['description']}")
    except:
        st.write("Weather: Syncing...")
    
    st.markdown("---")
    days = (date(2027, 5, 10) - date.today()).days # Standard Season opener
    st.markdown(f"**Expedition Clock:** {days} Days to Window")
    
    st.markdown("---")
    st.markdown("### 🛠️ Rapid Actions")
    if st.button("📋 Gear Checklist", use_container_width=True):
        st.session_state.temp_prompt = "What equipment do I need to climb Everest?"

# --- 4. CHAT HISTORY DISPLAY ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Greetings! I am the Everest Intelligence Agent. How can I assist your expedition today?"}]

# Custom rendering function to get the "Before" look
def render_message(role, content):
    if role == "user":
        st.markdown(f"""
            <div class="msg-box user-msg">
                <span class="label">ME</span>
                {content}
            </div>
        """, unsafe_allow_html=True)
    else:
        # Format the bot's response for bold and newlines
        formatted = content.replace("**", "<b>").replace("**", "</b>").replace("\n", "<br>")
        st.markdown(f"""
            <div class="msg-box bot-msg">
                <span class="label">Everest AI</span>
                {formatted}
            </div>
        """, unsafe_allow_html=True)

# Draw the chat history
for msg in st.session_state.messages:
    render_message(msg["role"], msg["content"])

# --- 5. AI INFERENCE LOGIC ---
def predict(s):
    sentence_words = nltk.word_tokenize(s)
    clean_words = [lemmatizer.lemmatize(w.lower()) for w in sentence_words]
    bow = [1 if w in clean_words else 0 for w in words]
    res = model.predict(np.array([bow]), verbose=0)[0]
    idx = np.argmax(res)
    return classes[idx] if res[idx] > 0.70 else "fallback"

# --- 6. USER INTERACTION ---
# Check if button was clicked
prompt = st.chat_input("Ask about Everest...")
if "temp_prompt" in st.session_state:
    prompt = st.session_state.pop("temp_prompt")

if prompt:
    # Add User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun() # Refresh to show user bubble immediately

# Logic for response (runs after rerun)
if st.session_state.messages[-1]["role"] == "user":
    user_input = st.session_state.messages[-1]["content"]
    tag = predict(user_input)
    for i in intents['intents']:
        if i['tag'] == tag:
            response = random.choice(i['responses'])
            break
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()