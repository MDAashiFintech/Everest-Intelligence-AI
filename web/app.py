import os
import sys
import requests
import json
import nltk
from flask import Flask, render_template, request, jsonify
from datetime import date
from dotenv import load_dotenv

# --- 1. CLOUD INITIALIZATION ---
load_dotenv()

# Download NLTK data before anything else
def init_infrastructure():
    for pkg in ['punkt', 'punkt_tab', 'wordnet', 'omw-1.4']:
        nltk.download(pkg, quiet=True)

init_infrastructure()

# Ensure the app can find the chatbot folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from chatbot.chatbot_utils import predict_class, get_response

app = Flask(__name__)

# --- 2. LOGIC ---
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

def get_everest_weather():
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat=27.98&lon=86.92&appid={WEATHER_API_KEY}&units=metric"
        res = requests.get(url, timeout=5).json()
        return {"temp": round(res["main"]["temp"]), "wind": round(res["wind"]["speed"] * 3.6), "desc": res["weather"][0]["description"].title()}
    except: return {"temp": "--", "wind": "--", "desc": "Offline"}

def get_summit_countdown():
    target = date(date.today().year, 5, 10)
    if date.today() > target: target = date(date.today().year + 1, 5, 10)
    return (target - date.today()).days

# Load sidebar from intents.json
INTENTS_PATH = os.path.join(BASE_DIR, "chatbot", "intents.json")
try:
    with open(INTENTS_PATH, "r", encoding="utf-8") as f:
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
    user_msg = request.json.get("message")
    try:
        tag = predict_class(user_msg)
        resp = get_response(tag)
        return jsonify({"response": resp})
    except Exception as e:
        print(f"Chat Logic Error: {str(e)}")
        return jsonify({"response": "Internal systems are recalibrating. Please wait 10 seconds."})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)