# 🏔️ Everest Intelligence AI

<p align="center">
  <strong>An NLP-powered conversational AI assistant for Mount Everest and mountaineering intelligence.</strong>
</p>

<p align="center">
  <a href="https://everest-ai-aashif.up.railway.app">
    <img src="https://img.shields.io/badge/Live%20Demo-Railway-7B2CBF?style=for-the-badge&logo=railway&logoColor=white" alt="Live Demo">
  </a>
  <a href="https://github.com/MDAashiFintech/Everest-Intelligence-AI">
    <img src="https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub Repository">
  </a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.11">
  <img src="https://img.shields.io/badge/TensorFlow-2.15-FF6F00?style=flat-square&logo=tensorflow&logoColor=white" alt="TensorFlow 2.15">
  <img src="https://img.shields.io/badge/Keras-2.15-D00000?style=flat-square&logo=keras&logoColor=white" alt="Keras 2.15">
  <img src="https://img.shields.io/badge/Flask-3.0.3-000000?style=flat-square&logo=flask&logoColor=white" alt="Flask 3.0.3">
  <img src="https://img.shields.io/badge/NLTK-3.9.1-154F3C?style=flat-square" alt="NLTK 3.9.1">
  <img src="https://img.shields.io/badge/Railway-Deployed-7B2CBF?style=flat-square&logo=railway&logoColor=white" alt="Railway Deployment">
  <img src="https://img.shields.io/github/license/MDAashiFintech/Everest-Intelligence-AI?style=flat-square" alt="License">
</p>

---

## 🌐 Live Application

The production version is deployed on Railway:

### [Launch Everest Intelligence AI](https://everest-ai-aashif.up.railway.app)

GitHub repository:

### [MDAashiFintech/Everest-Intelligence-AI](https://github.com/MDAashiFintech/Everest-Intelligence-AI)

---

## 📌 Project Overview

**Everest Intelligence AI** is a Flask-based conversational AI application that answers questions about Mount Everest and mountaineering.

The chatbot uses a TensorFlow neural network to classify user messages into predefined intents. It then selects a suitable response from a structured Mount Everest knowledge base.

The application also includes:

- live weather information for the Everest region;
- an expedition countdown;
- suggested questions organised by knowledge category;
- a responsive browser-based chat interface;
- lazy loading of AI assets to reduce initial server memory usage;
- production deployment using Railway and Gunicorn.

This project demonstrates an end-to-end AI application workflow, including NLP preprocessing, neural network training, inference, web development, API integration, cloud deployment, and dependency management.

---

## ✨ Main Features

- 🤖 Neural-network-based intent classification
- 🧠 Natural language preprocessing with NLTK
- 🏔️ Mount Everest and expedition knowledge base
- 🌦️ Live Everest weather through OpenWeather API
- ⏳ Dynamic summit-window countdown
- 💬 Suggested question buttons
- 🧭 Expandable knowledge categories
- ⌨️ Chat typing animation
- 📱 Responsive web interface
- ⚡ Lazy loading of TensorFlow model assets
- ☁️ Automatic Railway deployment from GitHub
- 🔐 Environment-variable-based API configuration

---

## 🧠 How the Chatbot Works

The chatbot follows this pipeline:

```text
User message
     │
     ▼
Flask /chat endpoint
     │
     ▼
NLTK tokenisation and lemmatisation
     │
     ▼
Bag-of-Words vector
     │
     ▼
TensorFlow neural network
     │
     ▼
Predicted intent
     │
     ▼
Response selected from intents.json
     │
     ▼
JSON response returned to the browser
```

The AI model is loaded only when the first chat message is submitted. This avoids loading TensorFlow and the trained model before they are needed.

---

## 🛠️ Technology Stack

### Artificial Intelligence and NLP

- Python 3.11
- TensorFlow 2.15
- Keras 2.15
- NLTK
- NumPy
- Bag-of-Words text representation
- Feed-forward neural network
- Intent classification

### Backend

- Flask
- Gunicorn
- Requests
- Python Dotenv

### Frontend

- HTML5
- CSS3
- JavaScript
- Responsive chat interface

### External Services

- OpenWeather API
- Railway
- GitHub

---

## 📁 Project Structure

```text
Everest-Intelligence-AI/
│
├── chatbot/
│   ├── chatbot_utils.py
│   └── intents.json
│
├── evaluation/
│   ├── evaluation_metrics_chart.png
│   ├── evaluation_report.csv
│   └── evaluation_table.png
│
├── model/
│   ├── chatbot_model.h5
│   ├── chatbot_model.keras
│   ├── classes.pkl
│   ├── words.pkl
│   ├── training.py
│   ├── evaluate_model.py
│   ├── training_data.pkl
│   └── training_history.pkl
│
├── web/
│   ├── app.py
│   ├── static/
│   │   └── style.css
│   └── templates/
│       └── index.html
│
├── .gitattributes
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
└── runtime.txt
```

### Important files

| File | Purpose |
|---|---|
| `chatbot/intents.json` | Contains training patterns, intent labels, and chatbot responses |
| `chatbot/chatbot_utils.py` | Loads the model and performs inference |
| `model/training.py` | Preprocesses the dataset and trains the neural network |
| `model/chatbot_model.h5` | Trained model loaded by the Flask application |
| `model/classes.pkl` | Saved intent labels |
| `model/words.pkl` | Saved vocabulary |
| `web/app.py` | Flask application and API routes |
| `web/templates/index.html` | Main web interface |
| `web/static/style.css` | Frontend styling |
| `requirements.txt` | Production Python dependencies |

---

## ⚙️ System Requirements

Recommended local environment:

- Python 3.11
- pip
- Git
- macOS, Windows, or Linux
- Internet connection for weather information and first-time NLTK downloads

The production deployment uses Python 3.11 and TensorFlow CPU 2.15.

---

## 🚀 Local Installation

### 1. Clone the repository

```bash
git clone https://github.com/MDAashiFintech/Everest-Intelligence-AI.git
cd Everest-Intelligence-AI
```

---

### 2. Create a Python 3.11 virtual environment

#### macOS or Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

#### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

#### Windows Command Prompt

```cmd
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
```

Verify the active Python version:

```bash
python --version
```

Expected:

```text
Python 3.11.x
```

---

## 📦 Install Dependencies

### Linux or Railway-compatible environment

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The production `requirements.txt` contains:

```text
Flask==3.0.3
gunicorn==23.0.0

tensorflow-cpu==2.15.0
numpy<2

nltk==3.9.1
pandas==2.2.2

python-dotenv==1.0.1
requests==2.32.3
```

### Apple Silicon Mac

The `tensorflow-cpu` package is not distributed for Apple Silicon macOS. Install the macOS build instead:

```bash
python -m pip install --upgrade pip
pip install tensorflow-macos==2.15.0
pip install "numpy<2" nltk==3.9.1 pandas==2.2.2 Flask==3.0.3 gunicorn==23.0.0 python-dotenv==1.0.1 requests==2.32.3
```

Verify TensorFlow and Keras:

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
python -c "import keras; print(keras.__version__)"
```

Expected:

```text
2.15.0
2.15.0
```

---

## 🔐 Environment Variables

Create a `.env` file in the project root:

```env
WEATHER_API_KEY=YOUR_OPENWEATHER_API_KEY
```

Do not commit the `.env` file to GitHub.

The repository's `.gitignore` excludes environment files and virtual environments.

When `WEATHER_API_KEY` is unavailable or the weather service cannot be reached, the application falls back to an offline weather state.

---

## 📚 NLTK Resources

The Flask application checks for required NLTK data and downloads missing resources when necessary.

They can also be installed manually:

```bash
python -m nltk.downloader punkt punkt_tab wordnet omw-1.4 stopwords
```

Required resources:

- `punkt`
- `punkt_tab`
- `wordnet`
- `omw-1.4`
- `stopwords`

---

## 🧠 Training the AI Model

The repository already contains a trained model, so retraining is not required to run the application.

Retrain after modifying `chatbot/intents.json`:

```bash
python model/training.py
```

Training generates or updates:

```text
model/chatbot_model.h5
model/chatbot_model.keras
model/classes.pkl
model/words.pkl
model/training_data.pkl
model/training_history.pkl
```

### Model architecture

```text
Input layer
    │
    ▼
Dense layer: 128 neurons, ReLU
    │
    ▼
Dropout: 0.5
    │
    ▼
Dense layer: 64 neurons, ReLU
    │
    ▼
Dropout: 0.5
    │
    ▼
Output layer: Softmax
```

### Training configuration

| Parameter | Value |
|---|---|
| Optimizer | Legacy SGD |
| Learning rate | 0.01 |
| Momentum | 0.9 |
| Nesterov | Enabled |
| Epochs | 200 |
| Batch size | 5 |
| Loss function | Categorical cross-entropy |
| Output activation | Softmax |

### Latest training result

| Metric | Result |
|---|---:|
| Final training accuracy | 90.60% |
| Final training loss | 0.2366 |

> The reported accuracy is training accuracy. It should not be treated as independent test-set accuracy unless it is supported by a separate evaluation dataset.

---

## 📊 Model Evaluation

Run the evaluation script:

```bash
python model/evaluate_model.py
```

Evaluation artefacts are stored in:

```text
evaluation/
├── evaluation_metrics_chart.png
├── evaluation_report.csv
└── evaluation_table.png
```

---

## ▶️ Run the Flask Application Locally

From the project root:

```bash
python web/app.py
```

The development server should start at:

```text
http://127.0.0.1:5000
```

Open that address in a browser.

A successful model load will print:

```text
✅ AI assets loaded into memory.
```

The model is lazy-loaded when the first chat request is sent.

---

## 💬 Example Questions

Try questions such as:

```text
Hello
```

```text
Where is Mount Everest?
```

```text
How high is Mount Everest?
```

```text
How long does it take to climb Everest?
```

```text
What equipment is required to climb Everest?
```

```text
How much does it cost to climb Everest?
```

```text
What safety measures should climbers follow?
```

```text
What is the death zone?
```

```text
Tell me about Sherpas.
```

```text
What is the best season to climb Everest?
```

---

## 🌦️ Weather Integration

The application requests current weather information for the Everest region using the OpenWeather API.

The backend requests:

- temperature in Celsius;
- wind speed;
- weather description.

The weather function includes timeout handling and an offline fallback so a weather API failure does not stop the chatbot interface.

---

## ☁️ Railway Deployment

The live version is hosted on Railway:

[https://everest-ai-aashif.up.railway.app](https://everest-ai-aashif.up.railway.app)

### Railway environment variables

```text
NIXPACKS_PYTHON_VERSION=3.11
WEATHER_API_KEY=YOUR_OPENWEATHER_API_KEY
```

Railway supplies the `PORT` variable automatically. It should not be created manually.

### Railway build settings

| Setting | Value |
|---|---|
| Builder | Nixpacks |
| Build command | Leave blank |
| Root directory | Repository root |
| Start command | See below |

### Start command

```bash
gunicorn --chdir web app:app --bind 0.0.0.0:$PORT --timeout 120
```

### Deployment workflow

```text
Local development
      │
      ▼
Git commit and push
      │
      ▼
GitHub main branch
      │
      ▼
Railway automatic build
      │
      ▼
Gunicorn starts Flask
      │
      ▼
Live application
```

---

## 🛠️ Deployment Issue Resolved

During the original deployment, Railway failed with:

```text
externally-managed-environment
```

The project previously contained a custom `nixpacks.toml` that ran:

```bash
python3 -m ensurepip
```

Railway uses an immutable Nix environment, so manually modifying its Python installation triggered the PEP 668 error.

The issue was fixed by:

- removing the custom `nixpacks.toml`;
- removing manual `ensurepip` and pip build commands;
- allowing Railway to create and manage its Python environment;
- setting Python 3.11 through Railway variables;
- leaving the Railway build command blank;
- using an explicit Gunicorn start command.

---

## 🛠️ Model Compatibility Issue Resolved

After deployment, the chatbot originally returned:

```text
I'm having trouble accessing my memory modules.
```

Railway logs showed:

```text
Unrecognized keyword arguments: ['batch_shape']
```

This was caused by a TensorFlow/Keras serialization mismatch between the old saved model and TensorFlow 2.15.

The issue was fixed by:

- creating a Python 3.11 virtual environment;
- installing TensorFlow/Keras 2.15;
- correcting the training script;
- replacing the deprecated optimizer configuration;
- retraining the model;
- regenerating `chatbot_model.h5`, `classes.pkl`, and `words.pkl`;
- testing model inference locally;
- deploying the regenerated model to Railway.

---

## 🔧 Troubleshooting

### TensorFlow is not installed

```text
ModuleNotFoundError: No module named 'tensorflow'
```

Activate the project environment first:

```bash
source .venv/bin/activate
```

Then verify:

```bash
which python
python --version
```

---

### `tensorflow-cpu==2.15.0` cannot be installed on Apple Silicon

Use:

```bash
pip install tensorflow-macos==2.15.0
```

Railway and Linux continue to use:

```text
tensorflow-cpu==2.15.0
```

---

### zsh reports `no such file or directory: 2`

This occurs when entering an unquoted version constraint:

```bash
pip install numpy<2
```

Quote it:

```bash
pip install "numpy<2"
```

---

### Keras reports that `decay` is deprecated

Use the legacy optimizer import in `model/training.py`:

```python
from tensorflow.keras.optimizers.legacy import SGD
```

---

### The chatbot cannot access its memory modules

Check Railway or local logs for:

```text
❌ Load Failure:
```

Verify that these files exist and are tracked by Git:

```text
model/chatbot_model.h5
model/classes.pkl
model/words.pkl
chatbot/intents.json
```

Check them with:

```bash
git ls-files model chatbot
```

---

### Weather displays Offline

Verify the environment variable:

```env
WEATHER_API_KEY=YOUR_OPENWEATHER_API_KEY
```

Also confirm that the API key is active and has been added to Railway Variables.

---

### Reset the local virtual environment

#### macOS or Linux

```bash
deactivate
rm -rf .venv
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

For Apple Silicon:

```bash
pip install tensorflow-macos==2.15.0
pip install "numpy<2" nltk==3.9.1 pandas==2.2.2 Flask==3.0.3 gunicorn==23.0.0 python-dotenv==1.0.1 requests==2.32.3
```

---

## 🔒 Security Notes

- Never commit `.env`.
- Never place the weather API key directly inside Python code.
- Store production secrets in Railway Variables.
- Rotate an API key immediately if it is exposed publicly.
- Avoid committing virtual environments or local caches.
- Keep `__pycache__`, `.DS_Store`, and `.venv` ignored.

---

## 🚧 Future Improvements

Potential future development includes:

- retrieval-augmented generation for broader Everest knowledge;
- multilingual question answering;
- speech input and text-to-speech;
- conversation history;
- confidence scores in the interface;
- improved fallback responses;
- automated model evaluation;
- train, validation, and test splits;
- Docker deployment;
- continuous integration testing;
- live expedition and route information;
- admin interface for updating intents;
- migration from Bag-of-Words to transformer embeddings.

---

## 👨‍💻 Author

### MD Aashif Ansari

**AI Product Engineer and Data Scientist**

- GitHub: [MDAashiFintech](https://github.com/MDAashiFintech)
- LinkedIn: [MD Aashif Ansari](https://www.linkedin.com/in/md-aashif-ansari786/)
- Live project: [Everest Intelligence AI](https://everest-ai-aashif.up.railway.app)

---

## 📄 License

This project is licensed under the MIT License.

See the [LICENSE](LICENSE) file for details.

---

## ⭐ Support

If this project is useful, consider starring the repository:

[⭐ Star Everest Intelligence AI](https://github.com/MDAashiFintech/Everest-Intelligence-AI)

---

<p align="center">
  Built with Python, TensorFlow, Flask and Railway.
</p>
