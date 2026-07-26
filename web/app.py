import os
import sys
import requests
import json
import nltk
from flask import Flask, render_template, request, jsonify
from datetime import date
from dotenv import load_dotenv

# --- 1. PREPARE CLOUD ENVIRONMENT ---
load_dotenv()

# Automatically download required NLTK data on startup
def prepare_nltk():
    pkgs = ['punkt', 'punkt_tab', 'wordnet', 'omw-1.4']
    for p in pkgs:
        nltk.download(p, quiet=True)

prepare_nltk()

# Setup paths so Flask can find the 'chatbot' folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from chatbot.chatbot_utils import predict_class, get_response

app = Flask(__name__)

# --- 2. UTILITY LOGIC ---
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

def get_everest_weather():
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat=27.98&lon=86.92&appid={WEATHER_API_KEY}&units=metric"
        res = requests.get(url, timeout=5).json()
        return {
            "temp": round(res["main"]["temp"]),
            "wind": round(res["wind"]["speed"] * 3.6),
            "desc": res["weather"][0]["description"].title()
        }
    except:
        return {"temp": "--", "wind": "--", "desc": "Offline"}

def get_summit_countdown():
    today = date.today()
    target = date(today.year, 5, 10)
    if today > target:
        target = date(today.year + 1, 5, 10)
    return (target - today).days

# Load sidebar data from intents.json
try:
    with open(os.path.join(BASE_DIR, "chatbot", "intents.json"), "r", encoding="utf-8") as f:
        intents_data = json.load(f)
    sidebar_data = [{"tag": i["tag"], "patterns": i["patterns"][:3]} for i in intents_data["intents"]]
except:
    sidebar_data = []

# --- 3. ROUTES ---

@app.route("/")
def index():
    return render_template("index.html", 
                           sidebar=sidebar_data, 
                           weather=get_everest_weather(), 
                           countdown=get_summit_countdown())

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")
    try:
        # Run user input through the TensorFlow brain
        tag = predict_class(user_message)
        response = get_response(tag)
        return jsonify({"response": response})
    except Exception as e:
        print(f"RUNTIME CHAT ERROR: {str(e)}")
        return jsonify({"response": "My systems are still waking up. Please ask me again in 5 seconds!"})

if __name__ == "__main__":
    # Render requires port handling
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)