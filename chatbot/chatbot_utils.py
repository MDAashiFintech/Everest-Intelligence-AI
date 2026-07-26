import random
import json
import pickle
import numpy as np
import os
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

# --- 1. DYNAMIC PATH SYSTEM ---
# This finds the folders relative to this script, no matter where it is run
CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__)) # chatbot/
BASE_DIR = os.path.dirname(CURRENT_FILE_DIR)                  # Everest-Intelligence-AI/
MODEL_DIR = os.path.join(BASE_DIR, "model")
CHATBOT_DIR = os.path.join(BASE_DIR, "chatbot")

lemmatizer = WordNetLemmatizer()

def load_ai_assets():
    """Loads the neural network and vocabulary from the cloud filesystem"""
    try:
        # Define exact absolute paths for the server
        intents_path = os.path.join(CHATBOT_DIR, "intents.json")
        words_path = os.path.join(MODEL_DIR, "words.pkl")
        classes_path = os.path.join(MODEL_DIR, "classes.pkl")
        model_path = os.path.join(MODEL_DIR, "chatbot_model.keras")

        # Load files
        with open(intents_path, "r", encoding="utf-8") as f:
            ints = json.load(f)
        
        w = pickle.load(open(words_path, "rb"))
        c = pickle.load(open(classes_path, "rb"))
        m = load_model(model_path)
        
        print("✅ SUCCESS: AI assets loaded from the filesystem.")
        return ints, w, c, m
    except Exception as e:
        print(f"❌ CRITICAL LOAD FAILURE: {str(e)}")
        return None, None, None, None

# Initialize assets once when server starts
intents, words, classes, model = load_ai_assets()

# --- 2. LOGIC FUNCTIONS ---

def clean_up_sentence(sentence):
    # Proper tokenization
    sentence_words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    if model is None:
        return "asset_error"
        
    bow = bag_of_words(sentence)
    # verbose=0 keeps the Render logs clean
    res = model.predict(np.array([bow]), verbose=0)[0]
    
    top_index = np.argmax(res)
    # Only return the tag if confidence is > 70%
    return classes[top_index] if res[top_index] > 0.70 else "fallback"

def get_response(tag):
    if tag == "asset_error":
        return "I'm having trouble accessing my memory modules right now."
    if intents is None:
        return "I'm sorry, my internal systems are currently offline."
        
    for i in intents["intents"]:
        if i["tag"] == tag:
            return random.choice(i["responses"])
            
    return "I'm not sure how to answer that specific query. Could you try rephrasing?"