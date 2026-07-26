import sys
import os
import json
import requests
from flask import Flask, render_template, request, jsonify
from datetime import date
from dotenv import load_dotenv

# --- 1. INITIALIZATION & SECURITY ---
# Load variables from the hidden .env file
load_dotenv()

# Setup paths so the web app can see the 'chatbot' logic folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# Import the AI brain from your chatbot folder
from chatbot.chatbot_utils import predict_class, get_response

app = Flask(__name__)

# --- 2. CONFIGURATION ---
# Grabs the key from your .env file (Local) or Render Environment (Cloud)
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
EVEREST_LAT = "27.98"
EVEREST_LON = "86.92"

# --- 3. ANALYTICAL ENGINES ---

def get_everest_weather():
    """Fetches real-time meteorological data for Everest Base Camp"""
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
            "wind": round(data["wind"]["speed"] * 3.6), # m/s to km/h
            "desc": data["weather"][0]["description"].title()
        }
    except Exception as e:
        print(f"Weather API Error: {e}")
        return {"temp": "--", "wind": "--", "desc": "Offline"}

def get_summit_countdown():
    """Calculates days remaining until the next May summit window"""
    today = date.today()
    summit_start = date(today.year, 5, 10)
    if today > summit_start:
        summit_start = date(today.year + 1, 5, 10)
    return (summit_start - today).days

# --- 4. DATA PREPARATION ---
# Load intents to populate the sidebar "Knowledge Base"
INTENTS_PATH = os.path.join(BASE_DIR, "chatbot", "intents.json")
try:
    with open(INTENTS_PATH, "r", encoding="utf-8") as f:
        intents_data = json.load(f)
    # Get first 3 patterns of each tag for the sidebar buttons
    sidebar_data = [{"tag": i["tag"], "patterns": i["patterns"][:3]} for i in intents_data["intents"]]
except Exception as e:
    print(f"Error loading intents for sidebar: {e}")
    sidebar_data = []

# --- 5. ROUTES ---

@app.route("/")
def index():
    """Renders the main animated dashboard"""
    weather = get_everest_weather()
    countdown = get_summit_countdown()
    return render_template("index.html", 
                           sidebar=sidebar_data, 
                           weather=weather, 
                           countdown=countdown)

@app.route("/chat", methods=["POST"])
def chat():
    """Handles the conversational AI interaction"""
    user_data = request.json
    if not user_data or "message" not in user_data:
        return jsonify({"response": "I didn't receive a message."}), 400
        
    user_message = user_data.get("message")
    
    try:
        # Pass user text through the TensorFlow model
        intent_tag = predict_class(user_message)
        bot_response = get_response(intent_tag)
        return jsonify({"response": bot_response})
    except Exception as e:
        print(f"Chat Logic Error: {e}")
        return jsonify({"response": "My internal systems are recalibrating. Try again in a moment."})

# --- 6. PRODUCTION SERVER ---
if __name__ == "__main__":
    # Render uses the 'PORT' environment variable to assign a web address
    port = int(os.environ.get("PORT", 5000))
    # host='0.0.0.0' is required for cloud access
    app.run(host='0.0.0.0', port=port, debug=True)