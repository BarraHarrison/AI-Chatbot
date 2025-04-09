# 🤖 AI Chatbot Assistant

Welcome to the **AI Chatbot Assistant**, a powerful and interactive Python-based chatbot built using **PyTorch**, **PyQt5**, and modern AI libraries. This project combines machine learning, voice recognition, text-to-speech, and real-time APIs — all within a sleek GUI interface.

Whether you prefer typing or speaking, this assistant is ready to help with news updates, weather forecasts, stock prices, jokes, and more!

---

## 🚀 Features

✅ **Terminal Chatbot** – Classic command-line interface powered by a custom-trained PyTorch model.

✅ **Voice Recognition** – Speak naturally to the bot using your microphone via Google Speech Recognition.

✅ **Text-to-Speech (TTS)** – The chatbot talks back using `pyttsx3`, supporting multiple voices and languages.

✅ **Real-Time API Integrations**  
- **News Headlines** (NewsAPI.org)  
- **Weather Forecasts** (OpenWeatherMap)  
- **Stock Market Prices** (TwelveData API)

✅ **Graphical User Interface (GUI)** – Built with `PyQt5`, the bot features a clean, interactive design inspired by modern assistant apps.

✅ **Standalone Terminal App** – Runs independently from your terminal using the pre-trained model, no retraining required.

✅ **Custom App Icon** – Personalized `.icns` icon for use when packaged into a native desktop application (macOS).

---

## 💪 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/YourUsername/AI-Chatbot-Assistant.git
cd AI-Chatbot-Assistant
```

### 2. Create and Activate a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

> If `requirements.txt` is missing, manually install:
```bash
pip install torch numpy nltk pyttsx3 PyQt5 speechrecognition python-dotenv requests
```

### 4. Add Your `.env` File
Create a `.env` file in the root directory and include the following:
```env
NEWS_API_KEY=your_newsapi_key
WEATHER_API_KEY=your_openweathermap_key
STOCK_API_KEY=your_twelvedata_key
```

---

## 🧠 Running the Chatbot

### ➔ Option 1: Terminal Mode
```bash
python3 main.py
```

### ➔ Option 2: GUI Mode (with voice + TTS)
```bash
python3 chat_gui.py
```

Once launched, the chatbot will be ready to interact with you using text or voice.

---

## 🎤 Voice Commands Examples

- “Tell me a joke”
- “What's the weather like?”
- “How is Nvidia doing today?”
- “What are the latest news headlines?”
- “What’s today’s date?”

---

## 📦 Packaging (Optional)
You can package the app into a macOS `.app` bundle using PyInstaller:
```bash
pyinstaller chat_gui.spec
```
> Remember to configure `datas` and `icon` correctly in the `.spec` file.

---

## 📚 Conclusion

This AI Chatbot Assistant is more than just a project — it's a complete personal assistant powered by speech, APIs, and machine learning. Whether you’re a developer learning NLP or just want a smart terminal buddy, this assistant is built to adapt and grow.

---

## 🧑‍💻 Author

Built with 💻 by **Barra Harrison**  
[GitHub](https://github.com/BarraHarrison)

---

## 📜 License

This project is licensed under the MIT License.

---