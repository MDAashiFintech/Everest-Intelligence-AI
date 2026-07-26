import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import dataframe_image as dfi
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
CHATBOT_DIR = os.path.join(BASE_DIR, "chatbot")
OUTPUT_DIR = os.path.join(BASE_DIR, "evaluation")

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = load_model(os.path.join(MODEL_DIR, "chatbot_model.keras"))
intents = json.loads(open(os.path.join(CHATBOT_DIR, "intents.json")).read())
words = pickle.load(open(os.path.join(MODEL_DIR, "words.pkl"), "rb"))
classes = pickle.load(open(os.path.join(MODEL_DIR, "classes.pkl"), "rb"))
lemmatizer = WordNetLemmatizer()

# Prepare test data
X_test = []
y_true = []

for intent in intents["intents"]:
    tag = intent["tag"]
    for pattern in intent["patterns"]:
        sentence_words = [lemmatizer.lemmatize(w.lower()) for w in pattern.split()]
        bag = [1 if w in sentence_words else 0 for w in words]
        X_test.append(bag)
        y_true.append(tag)

X_test = np.array(X_test)
y_true = np.array(y_true)
pred_probs = model.predict(X_test)
predicted_classes = [classes[np.argmax(p)] for p in pred_probs]

# Evaluation report
report = classification_report(y_true, predicted_classes, output_dict=True)
df = pd.DataFrame(report).transpose()

df.to_csv(os.path.join(OUTPUT_DIR, "evaluation_report.csv"))

# Plot
labels = [l for l in df.index if l not in ["accuracy", "macro avg", "weighted avg"]]
x = np.arange(len(labels))
width = 0.25

fig, ax = plt.subplots(figsize=(16, 6))
ax.bar(x - width, df.loc[labels, "precision"], width, label="precision")
ax.bar(x, df.loc[labels, "recall"], width, label="recall")
ax.bar(x + width, df.loc[labels, "f1-score"], width, label="f1-score")

ax.set_ylabel("Score")
ax.set_title("Evaluation Metrics per Intent")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=90)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "evaluation_metrics_chart.png"))
plt.show()

# Save table image
dfi.export(df.round(2), os.path.join(OUTPUT_DIR, "evaluation_table.png"))

# Print text table
accuracy = report.get("accuracy", 0)
macro_avg = report.get("macro avg", {})
f1_score = macro_avg.get("f1-score", 0)

print("\n📊 Evaluation Results for Everest AI Chatbot:\n")
print("Accuracy: {:.2f}".format(accuracy))
print("F1-score: {:.2f}".format(f1_score))
print("Detailed report:\n")
print(
    "{:<30} {:>10} {:>10} {:>10} {:>10}".format(
        "Intent", "Precision", "Recall", "F1-score", "Support"
    )
)
print("-" * 75)

for intent in classes:
    if intent in report:
        row = report[intent]
        print(
            "{:<30} {:>10.2f} {:>10.2f} {:>10.2f} {:>10}".format(
                intent,
                row["precision"],
                row["recall"],
                row["f1-score"],
                int(row["support"]),
            )
        )

print("-" * 75)
print(
    "{:<30} {:>10.2f} {:>10.2f} {:>10.2f} {:>10}".format(
        "Accuracy", accuracy, accuracy, accuracy, int(sum(df["support"]))
    )
)
