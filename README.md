
# 📦 Project Setup Instructions

A beginner-friendly step-by-step instruction to set up and run the chatbot project locally.

---

## ⚙️ System Requirements
Python version: 3.12.2
OS: macOS (also works on Windows/Linux with slight path changes)
Terminal or VS Code recommended

---

## 🗂 Project Structure
Ensure your folder structure looks like this:
```
COM727_EverestChatbot/
│
├── chatbot/                     # Core chatbot logic
|   ├── chatbot.py
│   ├── chatbot_utils.py
│   └── intents.json
│
├── model/                      # Model training & evaluation
│   ├── training.py
│   ├── evaluate_model.py
│   ├── chatbot_model.keras
│   ├── classes.pkl
│   ├── words.pkl
│   ├── training_data.pkl
│   └── training_history.pkl
│
├── web/                        # Flask web interface
│   ├── app.py
│   ├── templates/
│   │   └── index.html
│   └── static/
│       └── style.css
│
├── requirements.txt
└── venv/
```

---

## 📁 1. Download / Clone the Project

If using Git:

```bash
git clone https://github.com/0ansam43/COM727_AE2_EverestChatbot-
cd COM727_EverestChatbot
```

Or manually copy the full folder into your system.

---

## 🐍 2. Ensure Python 3.10 or Higher is Installed

Check Python version:

```bash
python3 --version
```

If not installed, [download Python here](https://www.python.org/downloads/).

---

## 🧪 3. Create a Virtual Environment

```bash
python3 -m venv venv
```

### 🔹 Activate the environment:

#### On Mac/Linux:

```bash
source venv/bin/activate
```

#### 🪟 On Windows:

```bash
venv\Scripts\activate
```

---

## 📦 4. Install Required Libraries

Ensure you're inside the project root and venv is active:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🧠 5. Download NLTK Resources

Only run this once per system:

```bash
python -m nltk.downloader punkt wordnet stopwords
```

---

## 🛠️ 6. Train the Model (Optional If Already Trained)

```bash
python model/training.py
```

Creates `chatbot_model.keras`, `words.pkl`, `classes.pkl`, etc.

---

## 📊 7. Evaluate Model Performance (Optional)

```bash
python model/evaluate_model.py
```

Outputs CSV, evaluation table image, and bar chart.

---

## 🌐 8. Run the Web Interface

```bash
cd web
python app.py
```

Then open:

```
http://127.0.0.1:5000
```

---

## 💬 Chat Features

- Sidebar with expandable tags
- Suggested question buttons
- Typing animation
- Responsive layout

---

## 🧹 Troubleshooting / Reset

```bash
deactivate

rm -rf venv

python3 -m venv venv

source venv/bin/activate

pip install -r requirements.txt
```

Check if Python is correctly linked:

```bash
which python
```
