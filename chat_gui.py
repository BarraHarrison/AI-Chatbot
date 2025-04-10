from main import ChatbotAssistant, get_stocks, get_date, get_time, get_joke, get_news, get_weather, get_specific_stock, company_to_symbol
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QLineEdit, QPushButton, QMessageBox
)
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
import speech_recognition as sr
import sys
import pyttsx3

class ChatBotGUI(QWidget):
    def __init__(self, assistant):
        super().__init__()
        self.assistant = assistant

        self.engine = pyttsx3.init(driverName='nsss')
        self.engine.setProperty('rate', 170)
        self.engine.setProperty('volume', 1.0)
        self.engine.setProperty('voice', 'com.apple.speech.synthesis.voice.moira')

        self.setWindowTitle("AI Chatbot Assistant")
        self.setGeometry(100, 100, 500, 600)

        self.layout = QVBoxLayout()
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.layout.addWidget(self.chat_display)

        self.input_layout = QHBoxLayout()

        self.input_box = QLineEdit()
        self.input_box.setPlaceholderText("Type your message...")
        self.input_box.returnPressed.connect(self.handle_input)
        self.input_layout.addWidget(self.input_box)

        self.send_button = QPushButton("📤")
        self.send_button.setToolTip("Send")
        self.send_button.clicked.connect(self.handle_input)
        self.input_layout.addWidget(self.send_button)

        self.mic_button = QPushButton("🎤")
        self.mic_button.setToolTip("Speak")
        self.mic_button.clicked.connect(self.handle_voice_input)
        self.input_layout.addWidget(self.mic_button)

        self.layout.addLayout(self.input_layout)
        self.setLayout(self.layout)

    def handle_input(self):
        user_input = self.input_box.text().strip()
        if not user_input:
            return

        self.chat_display.append(f"🧑 You: {user_input}")

        # 🔴 Detect exit intent
        if user_input.lower() in ["leave", "quit", "stop"]:
            reply = QMessageBox.question(
                self,
                "Exit Confirmation",
                "Do you want to finish our conversation?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )

            if reply == QMessageBox.Yes:
                self.chat_display.append("🤖 Alright, goodbye! 👋")
                self.engine.say("Alright, goodbye!")
                self.engine.runAndWait()
                QApplication.quit()
            else:
                self.chat_display.append("🤖 No problem, let’s keep chatting!")
            self.input_box.clear()
            return

        # Normal processing
        response = self.assistant.process_message(user_input)
        self.chat_display.append(f"🤖 Bot: {response}\n")
        self.input_box.clear()

    def handle_voice_input(self):
        recognizer = sr.Recognizer()
        mic = sr.Microphone()

        self.chat_display.append("🎤 Listening...")

        try:
            with mic as source:
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.listen(source, timeout=5)

            user_input = recognizer.recognize_google(audio)
            self.chat_display.append(f"🧑 You (voice): {user_input}")

            # 🔴 Check for voice-exit intent
            if user_input.lower() in ["leave", "quit", "stop"]:
                reply = QMessageBox.question(
                    self,
                    "Exit Confirmation",
                    "Do you want to finish our conversation?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )

                if reply == QMessageBox.Yes:
                    self.chat_display.append("🤖 Alright, goodbye! 👋")
                    self.engine.say("Alright, goodbye!")
                    self.engine.runAndWait()
                    QApplication.quit()
                else:
                    self.chat_display.append("🤖 No problem, let’s keep chatting!")
                    self.engine.say("No problem, let’s keep chatting!")
                    self.engine.runAndWait()
                return

            # ✅ Process the spoken input normally
            response = self.assistant.process_message(user_input)
            self.chat_display.append(f"🤖 Bot: {response}\n")
            self.engine.say(response)
            self.engine.runAndWait()

        except sr.UnknownValueError:
            self.chat_display.append("🤖 Sorry, I didn't catch that. Please try again.")
        except sr.RequestError:
            self.chat_display.append("⚠️ Could not request results from Google Speech API.")
        except Exception as e:
            self.chat_display.append(f"⚠️ Mic error: {str(e)}")




if __name__ == "__main__":
    assistant = ChatbotAssistant("intents.json", function_mappings={
        "stocks": get_stocks,
        "date": get_date,
        "time": get_time,
        "joke": get_joke,
        "news": get_news,
        "weather": get_weather
    })

    assistant.parse_intents()
    assistant.prepare_data()

    import os
    if os.path.exists("chatbot_model.pth") and os.path.exists("dimensions.json"):
        assistant.load_model("chatbot_model.pth", "dimensions.json")
    else:
        assistant.train_model(batch_size=8, lr=0.001, epochs=100)
        assistant.save_model("chatbot_model.pth", "dimensions.json")

    app = QApplication(sys.argv)
    window = ChatBotGUI(assistant)
    window.show()
    sys.exit(app.exec_())
