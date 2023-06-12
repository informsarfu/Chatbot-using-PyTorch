import random
import json
from nltk_utils import tokenization,bag_of_words
import torch
from model import NeuralNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

with open("intents.json",'r') as f:
    intents = json.load(f)

trained_file = "data.pth"
data = torch.load(trained_file)

input_size = data['input_size']
output_size = data['output_size']
hidden_size = data['hidden_size']
all_words = data['all_words']
tags = data['tags']
model_state = data['model_state']

    
model = NeuralNet(input_size,hidden_size,output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sazzy"
print("Ask me anything!! (type QUIT to exit)")

while True:
    sentence = input("You: ")
    if sentence == "QUIT" or sentence == "quit":
        break
    
    tokens = tokenization(sentence)
    bag = bag_of_words(tokens,all_words)
    bag = bag.reshape(1,bag.shape[0])
    bag = torch.from_numpy(bag)
    
    output = model(bag)
    val,idx = torch.max(output,dim=1)
    tag = tags[idx.item()]
    
    probablity = torch.softmax(output,dim=1)
    final_prob = probablity[0][idx.item()]
    print(final_prob.item())
    
    if final_prob.item() > 0.75:
        for intent in intents["intents"]:
            if intent['tag'] == tag:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: I am sorry I do not understand your question....")
    
 
    
    


