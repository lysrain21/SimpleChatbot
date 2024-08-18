# evaluate.py
from test_data import test_data
import torch
import numpy as np
from main import load_data, tokenize, bag_of_words, ChatbotModel
from main import input_size, hidden_size, output_size

def load_model():
    # Load the trained model (assuming it's saved)
    model = ChatbotModel(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load('chatbot_model.pth'))
    model.eval()
    return model

def evaluate_accuracy(model, test_data, all_words, data):
    correct = 0
    total = len(test_data)
    
    for pair in test_data:
        question = pair['question']
        expected_answer = pair['answer']
        
        # Predict
        tokenized_sentence = tokenize(question)
        bow = bag_of_words(tokenized_sentence, all_words)
        bow_tensor = torch.tensor(bow, dtype=torch.float32)
        
        with torch.no_grad():
            output = model(bow_tensor)
        
        predicted_idx = torch.argmax(output).item()
        predicted_answer = data[predicted_idx]['answer']
        
        if predicted_answer == expected_answer:
            correct += 1
    
    accuracy = (correct / total) * 100
    print(f'Accuracy: {accuracy:.2f}%')

if __name__ == "__main__":
    # Load data
    data = load_data('chat_data.json')  # Ensure the file path is correct
    all_words = sorted(set(word for pair in data for word in tokenize(pair['question']) + tokenize(pair['answer'])))
    
    model = load_model()
    # You need to define or load your test data here
    # For example:
    # test_data = load_data('test_data.json')  # Ensure this file exists or is properly defined
    test_data = [...]  # Replace with actual test data
    evaluate_accuracy(model, test_data, all_words, data)
