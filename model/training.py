import os
import random
import json
import pickle
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers.legacy import SGD

# ───────────── Setup ─────────────
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
INTENTS_PATH = os.path.join(BASE_DIR, "chatbot", "intents.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "model")

os.makedirs(OUTPUT_DIR, exist_ok=True)

lemmatizer = WordNetLemmatizer()
intents = json.loads(open(INTENTS_PATH).read())

# ───────────── Preprocessing ─────────────
words = []
classes = []
documents = []
ignore_letters = ["?", "!", ".", "/", "@"]

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        word_list = pattern.split()
        words.extend(word_list)
        documents.append((word_list, intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

stop_words = set(stopwords.words("english"))
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))
classes = sorted(set(classes))

# Save preprocessed data
pickle.dump(words, open(os.path.join(OUTPUT_DIR, "words.pkl"), "wb"))
pickle.dump(classes, open(os.path.join(OUTPUT_DIR, "classes.pkl"), "wb"))

# ───────────── Training Data ─────────────
training = []
output_empty = [0] * len(classes)
for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1 if word in word_patterns else 0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)

train_x = list(training[:, 0])
train_y = list(training[:, 1])

# ───────────── Model Definition ─────────────
model = Sequential([
    Input(shape=(len(train_x[0]),)),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(64, activation="relu"),
    Dropout(0.5),
    Dense(len(train_y[0]), activation="softmax")
])

sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

# ───────────── Training ─────────────
hist = model.fit(
    np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1
)

# ───────────── Save Outputs ─────────────
# Save in the new Keras format
model.save(os.path.join(OUTPUT_DIR, "chatbot_model.keras"))

# Save in HDF5 format for compatibility
model.save(os.path.join(OUTPUT_DIR, "chatbot_model.h5"))

with open(os.path.join(OUTPUT_DIR, "training_data.pkl"), "wb") as f:
    pickle.dump((train_x, train_y), f)

with open(os.path.join(OUTPUT_DIR, "training_history.pkl"), "wb") as f:
    pickle.dump(hist.history, f)

# ───────────── Summary ─────────────
final_accuracy = hist.history["accuracy"][-1]
final_loss = hist.history["loss"][-1]
print(f"\n✅ Final Training Accuracy: {final_accuracy:.4f}")
print(f"📉 Final Training Loss: {final_loss:.4f}")
