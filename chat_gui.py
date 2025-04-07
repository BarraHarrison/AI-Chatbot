import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QTextEdit, QLineEdit, QPushButton

class ChatBotGUI(QWidget):
    def __init__(self, assistant):
        super().__init__()
        self.assistant = assistant
        self.setWindowTitle("AI Chatbot Assistant")
        self.setGeometry(100, 100, 500, 600)

        self.layout = QVBoxLayout()

        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.layout.addWidget(self.chat_display)

        self.input_box = QLineEdit()
        self.input_box.returnPressed.connect(self.handle_input)
        self.layout.addWidget(self.input_box)

        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.handle_input)
        self.layout.addWidget(self.send_button)

        self.setLayout(self.layout)

    def handle_input(self):
        user_input = self.input_box.text().strip()
        if not user_input:
            return

        self.chat_display.append(f"🧑 You: {user_input}")
        response = self.assistant.process_message(user_input)
        self.chat_display.append(f"🤖 Bot: {response}\n")

        self.input_box.clear()

if __name__ == "__main__":
    from main import ChatbotAssistant, get_stocks, get_date, get_time, get_joke, get_news, get_weather  # your functions
    from main import get_specific_stock, company_to_symbol

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
