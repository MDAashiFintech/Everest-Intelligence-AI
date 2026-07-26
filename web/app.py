import sys
import os
import json
import requests
import nltk # Added for auto-download
from flask import Flask, render_template, request, jsonify
from datetime import date
from dotenv import load_dotenv

# --- 1. INFRASTRUCTURE SETUP ---
load_dotenv()

# NEW: Automated NLTK Data Check for Cloud Production
def setup_nltk():
    required_data = ['punkt', 'punkt_tab', 'wordnet', 'omw-1.4']
    for data in required_data:
        try:
            nltk.data.find(f'tokenizers/{data}' if 'punkt' in data else f'corpora/{data}')
        except LookupError:
            nltk.download(data)

setup_nltk()

# Setup paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from chatbot.chatbot_utils import predict_class, get_response

app = Flask(__name__)

# --- 2. CONFIGURATION ---
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
EVEREST_LAT = "27.98"
EVEREST_LON = "86.92"

# --- 3. LOGIC ENGINES ---

def get_everest_weather():
    if not WEATHER_API_KEY:
        return {"temp": "--", "wind": "--", "desc": "Key Missing"}
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={EVEREST_LAT}&lon={EVEREST_LON}&appid={WEATHER_API_KEY}&units=metric"
        res = requests.get(url, timeout=5)
        data = res.json()
        if res.status_code != 200:
            return {"temp": "--", "wind": "--", "desc": "Activating..."}
        return {
            "temp": round(data["main"]["temp"]),
            "wind": round(data["wind"]["speed"] * 3.6),
            "desc": data["weather"][0]["description"].title()
        }
    except Exception as e:
        print(f"Weather API Error: {e}")
        return {"temp": "--", "wind": "--", "desc": "Offline"}

def get_summit_countdown():
    today = date.today()
    summit_start = date(today.year, 5, 10)
    if today > summit_start:
        summit_start = date(today.year + 1, 5, 10)
    return (summit_start - today).days

# --- 4. DATA LOADING ---
INTENTS_PATH = os.path.join(BASE_DIR, "chatbot", "intents.json")
try:
    with open(INTENTS_PATH, "r", encoding="utf-8") as f:
        intents_data = json.load(f)
    sidebar_data = [{"tag": i["tag"], "patterns": i["patterns"][:3]} for i in intents_data["intents"]]
except Exception as e:
    sidebar_data = []

# --- 5. ROUTES ---

@app.route("/")
def index():
    weather = get_everest_weather()
    countdown = get_summit_countdown()
    return render_template("index.html", sidebar=sidebar_data, weather=weather, countdown=countdown)

@app.route("/chat", methods=["POST"])
def chat():
    user_data = request.json
    if not user_data or "message" not in user_data:
        return jsonify({"response": "No message received."}), 400
    user_message = user_data.get("message")
    try:
        # AI Pipeline execution
        intent_tag = predict_class(user_message)
        bot_response = get_response(intent_tag)
        return jsonify({"response": bot_response})
    except Exception as e:
        # This captures the real error in the Render Logs
        print(f"Chat Logic Error: {str(e)}") 
        return jsonify({"response": "My internal systems are recalibrating. Try again in a moment."})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)