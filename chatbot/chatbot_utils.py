import random
import json
import pickle
import numpy as np
import os
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

# --- 1. PATH CONFIGURATION (CRITICAL FOR CLOUD) ---
# This ensures we find the files regardless of where the script is called from
current_dir = os.path.dirname(os.path.abspath(__file__)) # chatbot/
BASE_DIR = os.path.dirname(current_dir)                  # Root/
MODEL_DIR = os.path.join(BASE_DIR, "model")
CHATBOT_DIR = os.path.join(BASE_DIR, "chatbot")

# --- 2. ASSET LOADING ---
lemmatizer = WordNetLemmatizer()

def load_chatbot_assets():
    try:
        # Load Intents with UTF-8 encoding for safety
        with open(os.path.join(CHATBOT_DIR, "intents.json"), "r", encoding="utf-8") as f:
            intents = json.load(f)
        
        # Load Pickles
        words = pickle.load(open(os.path.join(MODEL_DIR, "words.pkl"), "rb"))
        classes = pickle.load(open(os.path.join(MODEL_DIR, "classes.pkl"), "rb"))
        
        # Load Keras Model
        model = load_model(os.path.join(MODEL_DIR, "chatbot_model.keras"))
        
        return intents, words, classes, model
    except Exception as e:
        print(f"Error loading chatbot assets: {e}")
        # Return None to handle gracefully if needed
        return None, None, None, None

intents, words, classes, model = load_chatbot_assets()

# --- 3. PROCESSING FUNCTIONS ---

def clean_up_sentence(sentence):
    # Tokenize the sentence
    sentence_words = nltk.word_tokenize(sentence)
    # Lemmatize and lowercase
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
    # If model failed to load, return fallback
    if model is None:
        return "fallback"
        
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]), verbose=0)[0] # verbose=0 keeps logs clean
    
    top_index = np.argmax(res)
    top_prob = res[top_index]
    
    # Confidence Threshold (Institutional Standard)
    threshold = 0.75
    if top_prob < threshold:
        return "fallback"
    
    return classes[top_index]

def get_response(intent_tag):
    # If no assets loaded
    if intents is None:
        return "I'm sorry, my internal systems are offline."
        
    for item in intents["intents"]:
        if item["tag"] == intent_tag:
            return random.choice(item["responses"])
            
    # Default fallback response if tag isn't found in JSON
    return "I'm not exactly sure how to answer that. Could you try rephrasing?"