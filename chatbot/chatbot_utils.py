import random
import json
import pickle
import numpy as np
import os
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

# --- ABSOLUTE DISCOVERY PATHS ---
# This looks for the files relative to this script, no matter where the server starts
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) # root/chatbot
BASE_DIR = os.path.dirname(CURRENT_DIR)                  # root
MODEL_DIR = os.path.join(BASE_DIR, "model")
CHATBOT_DIR = os.path.join(BASE_DIR, "chatbot")

lemmatizer = WordNetLemmatizer()

def load_ai_assets():
    try:
        # Construct paths
        intents_path = os.path.join(CHATBOT_DIR, "intents.json")
        words_path = os.path.join(MODEL_DIR, "words.pkl")
        classes_path = os.path.join(MODEL_DIR, "classes.pkl")
        model_path = os.path.join(MODEL_DIR, "chatbot_model.h5") # Fixed to .h5

        # Open and load
        with open(intents_path, "r", encoding="utf-8") as f:
            ints = json.load(f)
        
        w = pickle.load(open(words_path, "rb"))
        c = pickle.load(open(classes_path, "rb"))
        m = load_model(model_path)
        
        print(f"✅ AI CORE: System online and verified.")
        return ints, w, c, m
    except Exception as e:
        print(f"❌ AI CORE FAILURE: {str(e)}")
        return None, None, None, None

# Initialize assets immediately
intents, words, classes, model = load_ai_assets()

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s: bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    if model is None: return "error"
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]), verbose=0)[0]
    top_index = np.argmax(res)
    # Threshold at 0.70 for professional accuracy
    return classes[top_index] if res[top_index] > 0.70 else "fallback"

def get_response(tag):
    if intents is None: return "Internal system is offline. Please check data paths."
    if tag == "error": return "Memory retrieval error. Please try again."
    
    for i in intents["intents"]:
        if i["tag"] == tag:
            return random.choice(i["responses"])
            
    return "I'm not sure how to assist with that. Could you provide more detail?"