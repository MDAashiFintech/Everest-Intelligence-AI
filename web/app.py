import os
import sys
import requests
import json
import nltk
from flask import Flask, render_template, request, jsonify
from datetime import date
from dotenv import load_dotenv

# --- CLOUD INFRASTRUCTURE SETUP ---
load_dotenv()

# Force NLTK packages for the cloud
for pkg in ['punkt', 'punkt_tab', 'wordnet', 'omw-1.4']:
    nltk.download(pkg, quiet=True)

# Add project root to path so we can import 'chatbot'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from chatbot.chatbot_utils import predict_class, get_response

app = Flask(__name__)

def get_everest_weather():
    key = os.getenv("WEATHER_API_KEY")
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat=27.98&lon=86.92&appid={key}&units=metric"
        res = requests.get(url, timeout=5).json()
        return {
            "temp": round(res["main"]["temp"]),
            "wind": round(res["wind"]["speed"] * 3.6),
            "desc": res["weather"][0]["description"].title()
        }
    except:
        return {"temp": "--", "wind": "--", "desc": "Offline"}

def get_summit_countdown():
    target = date(date.today().year, 5, 10)
    if date.today() > target:
        target = date(date.today().year + 1, 5, 10)
    return (target - date.today()).days

# Load sidebar data from intents.json
try:
    with open(os.path.join(BASE_DIR, "chatbot", "intents.json"), "r", encoding="utf-8") as f:
        intents_data = json.load(f)
    sidebar_data = [{"tag": i["tag"], "patterns": i["patterns"][:3]} for i in intents_data["intents"]]
except:
    sidebar_data = []

# --- ROUTES ---

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
    except:
        return jsonify({"response": "Internal systems are recalibrating. Please wait 5 seconds."})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)