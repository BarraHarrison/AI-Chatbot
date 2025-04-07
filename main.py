import os 
import json
import random

import nltk
import numpy as np
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
from torch.utils.data import DataLoader, TensorDataset

nltk.download("punkt_tab")

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

        predicted_class_index = torch.argmax(predictions, dim=1).item()
        predicted_intent = self.intents[predicted_class_index]

        if self.function_mappings and predicted_intent in self.function_mappings:
            return self.function_mappings[predicted_intent]()

        if self.intents_responses[predicted_intent]:
            return random.choice(self.intents_responses[predicted_intent])
        else:
            return None


def get_stocks():
    stocks = ["META", "TSLA", "MSFT"]
    return random.sample(stocks, 2)


if __name__ == "__main__":
    assistant = ChatbotAssistant("intents.json", function_mappings={
    "stocks": get_stocks,
    "date": get_date,
    "time": get_time,
    "joke": get_joke
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