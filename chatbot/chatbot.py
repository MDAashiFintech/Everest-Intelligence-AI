import random
import json
import pickle
import numpy as np
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import os

# Resolve paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
DATA_FILE = os.path.join(BASE_DIR, "chatbot", "intents.json")

lemmatizer = WordNetLemmatizer()
intents = json.loads(open(DATA_FILE).read())
words = pickle.load(open(os.path.join(MODEL_DIR, "words.pkl"), "rb"))
classes = pickle.load(open(os.path.join(MODEL_DIR, "classes.pkl"), "rb"))
model = load_model(os.path.join(MODEL_DIR, "chatbot_model.keras"))


def clean_up_sentence(sentence):
    sentence_words = sentence.split()
    return [lemmatizer.lemmatize(word.lower()) for word in sentence_words]


def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    top_index = np.argmax(res)
    top_probability = res[top_index]
    ERROR_THRESHOLD = 0.75
    if top_probability < ERROR_THRESHOLD:
        return [{"intent": "fallback", "probability": str(top_probability)}]
    else:
        return [{"intent": classes[top_index], "probability": str(top_probability)}]


def get_response(intents_list, intents_json):
    tag = intents_list[0]["intent"]
    for intent in intents_json["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return "I'm not sure how to help with that."


print("🏔️ Everest AI Chatbot is ready! Ask me anything about Mount Everest.")
print("Type 'exit', 'quit', or 'bye' to end the conversation.")

while True:
    message = input("You: ")
    if message.lower() in ["exit", "quit", "bye"]:
        print("Everest AI: Goodbye! Stay safe on your Everest journey 🏔️")
        break
    ints = predict_class(message)
    res = get_response(ints, intents)
    print(f"Everest AI: {res}")
