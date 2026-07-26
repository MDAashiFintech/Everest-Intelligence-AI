import random
import json
import pickle
import numpy as np
import os
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

# --- PATH-PROOF SYSTEM (MATCHES YOUR TREE) ---
# This file is in /chatbot/chatbot_utils.py
CHATBOT_FOLDER = os.path.dirname(os.path.abspath(__file__)) 
# Root is one level up
ROOT_FOLDER = os.path.dirname(CHATBOT_FOLDER)
# Model is in /model/
MODEL_FOLDER = os.path.join(ROOT_FOLDER, "model")

lemmatizer = WordNetLemmatizer()

def load_ai_assets():
    try:
        # 1. Path to JSON is inside the chatbot folder
        intents_path = os.path.join(CHATBOT_FOLDER, "intents.json")
        # 2. Paths to models are in the model folder
        words_path = os.path.join(MODEL_FOLDER, "words.pkl")
        classes_path = os.path.join(MODEL_FOLDER, "classes.pkl")
        model_path = os.path.join(MODEL_FOLDER, "chatbot_model.h5")

        # Load assets
        with open(intents_path, "r", encoding="utf-8") as f:
            ints = json.load(f)
        
        w = pickle.load(open(words_path, "rb"))
        c = pickle.load(open(classes_path, "rb"))
        m = load_model(model_path)
        
        print("✅ AI BRAIN: Assets successfully synced in the cloud.")
        return ints, w, c, m
    except Exception as e:
        print(f"❌ AI BRAIN ERROR: {str(e)}")
        return None, None, None, None

# Run loader
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
    # 0.70 threshold for better balance
    return classes[top_index] if res[top_index] > 0.70 else "fallback"

def get_response(tag):
    if intents is None: return "Internal system is offline. Please check data paths."
    if tag == "error": return "Memory retrieval error. Please try again."
    
    for i in intents["intents"]:
        if i["tag"] == tag:
            return random.choice(i["responses"])
    return "I'm not sure how to answer that specifically."