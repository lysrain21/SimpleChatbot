import json
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


def load_data(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

data = load_data('chat_data.json')
# Download the necessary data
nltk.download('punkt')

def tokenize(sentence):
    return word_tokenize(sentence.lower())

# Build vocabulary
all_words = []
for pair in data:
    all_words.extend(tokenize(pair['question']))
    all_words.extend(tokenize(pair['answer']))

# Remove duplicates and sort
all_words = sorted(set(all_words))

# Create a word to index mapping
word2idx = {word: idx for idx, word in enumerate(all_words)}

print(word2idx)
class ChatbotModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ChatbotModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

input_size = len(word2idx)
hidden_size = 8
output_size = len(data)  # One output for each possible answer

model = ChatbotModel(input_size, hidden_size, output_size)
# Convert sentences to bag of words vectors
def bag_of_words(tokenized_sentence, words):
    sentence_words = set(tokenized_sentence)
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, word in enumerate(words):
        if word in sentence_words:
            bag[idx] = 1.0
    return bag

# Prepare training data
X_train = []
y_train = []
for idx, pair in enumerate(data):
    X_train.append(bag_of_words(tokenize(pair['question']), all_words))
    y_train.append(idx)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Training parameters
learning_rate = 0.001
num_epochs = 1000
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    X_train_tensor = torch.tensor(X_train)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    
    optimizer.zero_grad()
    output = model(X_train_tensor)
    loss = criterion(output, y_train_tensor)
    
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
torch.save(model.state_dict(), 'chatbot_model.pth')

def predict(model, sentence):
    tokenized_sentence = tokenize(sentence)
    bow = bag_of_words(tokenized_sentence, all_words)
    bow_tensor = torch.tensor(bow)
    
    with torch.no_grad():
        output = model(bow_tensor)
    
    predicted_idx = torch.argmax(output).item()
    return data[predicted_idx]['answer']

# Test the chatbot
print(predict(model, "Hello"))
print(predict(model, "What is your name?"))
