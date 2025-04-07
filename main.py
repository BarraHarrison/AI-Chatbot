import os 
import json
import random
import datetime
import requests

import nltk
import numpy as np
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
from torch.utils.data import DataLoader, TensorDataset
from dotenv import load_dotenv

load_dotenv()
nltk.download("punkt_tab")

NEWS_API_KEY = os.getenv("NEWS_API_KEY")

class ChatbotModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(ChatbotModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 120)
        self.fc2 = nn.Linear(120, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class ChatbotAssistant:
    def __init__(self, intents_path, function_mappings=None):
        self.model = None
        self.intents_path = intents_path
        self.documents = []
        self.vocabulary = []
        self.intents = []
        self.intents_responses = {}
        self.function_mappings = function_mappings
        self.X = None
        self.y = None

    @staticmethod
    def tokenize_and_lemmatize(text):
        lemmatizer = nltk.WordNetLemmatizer()
        words = nltk.word_tokenize(text)
        return [lemmatizer.lemmatize(word.lower()) for word in words]

    def bag_of_words(self, words, vocabulary):
        return [1 if word in words else 0 for word in vocabulary]

    def parse_intents(self):
        if os.path.exists(self.intents_path):
            with open(self.intents_path, "r") as f:
                intents_data = json.load(f)

            for intent in intents_data["intents"]:
                if intent["tag"] not in self.intents:
                    self.intents.append(intent["tag"])
                    self.intents_responses[intent["tag"]] = intent["responses"]

                for pattern in intent["patterns"]:
                    pattern_words = self.tokenize_and_lemmatize(pattern)
                    self.vocabulary.extend(pattern_words)
                    self.documents.append((pattern_words, intent["tag"]))

            self.vocabulary = sorted(set(self.vocabulary))

    def prepare_data(self):
        bags = []
        indices = []

        for document in self.documents:
            words = document[0]
            bag = self.bag_of_words(words, self.vocabulary)
            intent_index = self.intents.index(document[1])
            bags.append(bag)
            indices.append(intent_index)

        self.X = np.array(bags)
        self.y = np.array(indices)

    def train_model(self, batch_size, lr, epochs):
        X_tensor = torch.tensor(self.X, dtype=torch.float32)
        y_tensor = torch.tensor(self.y, dtype=torch.long)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model = ChatbotModel(self.X.shape[1], len(self.intents))

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            running_loss = 0.0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch+1}: Loss: {running_loss / len(loader):.4f}")

    def save_model(self, model_path, dimensions_path):
        torch.save(self.model.state_dict(), model_path)
        with open(dimensions_path, "w") as f:
            json.dump({"input_size": self.X.shape[1], "output_size": len(self.intents)}, f)

    def load_model(self, model_path, dimensions_path):
        with open(dimensions_path, "r") as f:
            dimensions = json.load(f)
        self.model = ChatbotModel(dimensions["input_size"], dimensions["output_size"])
        self.model.load_state_dict(torch.load(model_path))

    def process_message(self, input_message):
        words = self.tokenize_and_lemmatize(input_message)
        bag = self.bag_of_words(words, self.vocabulary)
        bag_tensor = torch.tensor([bag], dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(bag_tensor)
            probs = torch.softmax(predictions, dim=1)
            confidence, predicted_class_index = torch.max(probs, dim=1)

        confidence = confidence.item()
        predicted_class_index = predicted_class_index.item()

        if confidence < 0.75:
            return random.choice(self.intents_responses.get("fallback", ["I'm not sure I understand."]))


        predicted_intent = self.intents[predicted_class_index]

        if self.function_mappings and predicted_intent in self.function_mappings:
            return self.function_mappings[predicted_intent]()

        if self.intents_responses[predicted_intent]:
            return random.choice(self.intents_responses[predicted_intent])
        else:
            return None



def get_stocks():
    api_key = os.getenv("STOCK_API_KEY")
    symbols = ["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "NFLX", "INTC", "IBM"]
    selected = random.sample(symbols, 2)
    results = []

    for symbol in selected:
        print(f"Fetching data for: {symbol}")
        url = f"https://api.twelvedata.com/quote?symbol={symbol}&apikey={api_key}"
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            if "price" in data:
                price = data["price"]
                change = data["percent_change"]
                results.append(f"{symbol}: ${price} ({change}%)")
            else:
                results.append(f"{symbol}: No data available.")
        else:
            results.append(f"{symbol}: Failed to fetch data.")

    return "📈 Real-time stock update:\n" + "\n".join(results)


def get_date():
    today = datetime.date.today().strftime("%A, %B %d, %Y")
    return f"Today's date is {today} 📅"

def get_time():
    now = datetime.datetime.now().strftime("%H:%M:%S")
    return f"The current time is {now} 🕒"


def get_joke():
    jokes = [
        "Why did the programmer quit his job? Because he didn't get arrays.",
        "Why do Java developers wear glasses? Because they can't C#.",
        "How many programmers does it take to change a light bulb? None. It's a hardware problem!"
    ]
    return random.choice(jokes)

def get_news():
    url = f"https://newsapi.org/v2/top-headlines?country=us&pageSize=3&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    
    if response.status_code == 200:
        articles = response.json().get("articles", [])
        headlines = [f"- {article['title']}" for article in articles]
        return "📰 Here are the top headlines:\n" + "\n".join(headlines)
    else:
        return "Sorry, I couldn't fetch the news at the moment."
    

def get_weather():
    api_key = os.getenv("WEATHER_API_KEY")
    city = input("🤖 Which city would you like the weather for? ").strip()

    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        weather = data["weather"][0]["description"].capitalize()
        temp = data["main"]["temp"]
        feels_like = data["main"]["feels_like"]
        return f"🌦️ Weather in {city}:\n{weather}, {temp}°C (feels like {feels_like}°C)"
    else:
        return f"Sorry, I couldn't find the weather for '{city}'. Please try another city."



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

    model_exists = os.path.exists("chatbot_model.pth") and os.path.exists("dimensions.json")

    if model_exists:
        print("📦 Loading existing model...")
        assistant.load_model("chatbot_model.pth", "dimensions.json")
    else:
        print("🛠️ Training model...")
        assistant.train_model(batch_size=8, lr=0.001, epochs=100)
        assistant.save_model("chatbot_model.pth", "dimensions.json")

    print("🤖 Chatbot is ready! Type your message below (type 'quit', 'stop', or 'leave' to exit).")

    while True:
        message = input("You: ").strip().lower()

        if message in ["quit", "stop", "leave"]:
            confirm = input("🤖 Do you want to finish our conversation? (yes/no): ").strip().lower()
            if confirm in ["yes", "y"]:
                print("🤖 Alright, goodbye! 👋")
                break
            else:
                print("🤖 No problem! Let's keep chatting.")
                continue

        response = assistant.process_message(message)
        if response:
            print(f"🤖 {response}")
        else:
            print("🤖 Sorry, I didn't understand that.")