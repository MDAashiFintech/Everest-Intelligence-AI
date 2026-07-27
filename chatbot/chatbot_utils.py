import random
import json
import pickle
import numpy as np
import os
import nltk
from nltk.stem import WordNetLemmatizer
# We move the load_model import here to save memory at boot
from tensorflow.keras.models import load_model

# Force TensorFlow to use only 1 CPU thread (Saves massive RAM)
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
MODEL_DIR = os.path.join(BASE_DIR, "model")

lemmatizer = WordNetLemmatizer()

# Global variables for the "Brain"
intents = None
words = None
classes = None
model = None

def load_ai_assets():
    global intents, words, classes, model
    try:
        if model is not None: # Already loaded
            return True
            
        with open(os.path.join(CURRENT_DIR, "intents.json"), "r", encoding="utf-8") as f:
            intents = json.load(f)
        
        words = pickle.load(open(os.path.join(MODEL_DIR, "words.pkl"), "rb"))
        classes = pickle.load(open(os.path.join(MODEL_DIR, "classes.pkl"), "rb"))
        model = load_model(os.path.join(MODEL_DIR, "chatbot_model.h5"))
        
        print("✅ AI assets loaded into memory.")
        return True
    except Exception as e:
        print(f"❌ Load Failure: {e}")
        return False

# We do NOT call load_ai_assets() here anymore. We wait for the first chat.

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
    # If the brain isn't awake yet, wake it up now
    if model is None:
        if not load_ai_assets(): return "error"
        
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]), verbose=0)[0]
    idx = np.argmax(res)
    return classes[idx] if res[idx] > 0.70 else "fallback"

def get_response(tag):
    if tag == "error": return "I'm having trouble accessing my memory modules."
    if intents is None: load_ai_assets()
    
    for i in intents["intents"]:
        if i["tag"] == tag: return random.choice(i["responses"])
    return "I'm not sure about that."