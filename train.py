from nltk_utils import tokenization,stemming,bag_of_words
import json
import numpy as np
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import NeuralNet


with open('intents.json','r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        tokens = tokenization(pattern)
        all_words.extend(tokens)
        xy.append((tokens,tag))
        

exclude_chars = ['.','?','!',',']
all_words = [stemming(word) for word in all_words if word not in exclude_chars]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(f"all collected words: {all_words} \n")
print(f"all collected tags: {tags} \n")

x_train = []
y_train = []

for (tokens,tag) in xy:
    bag = bag_of_words(tokens,all_words)
    x_train.append(bag)
    
    label = tags.index(tag)
    y_train.append(label)
    
x_train = np.array(x_train)
y_train = np.array(y_train)


#hyper-parameters:
batch_size = 8
hidden_size = 8
input_size = len(all_words)
output_size = len(tags)
learning_rate = 0.01
num_of_epochs = 1000
    
    
class ChatbotDataset(Dataset): 
    def __init__(self):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train
    
    def __getitem__(self,idx):
        return self.x_data[idx],self.y_data[idx]

    def __len__(self):
        return self.n_samples
    

dataset = ChatbotDataset()
train_loader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True,num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
model = NeuralNet(input_size,hidden_size,output_size).to(device)

#loss and optimizer
criteria = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)

print("Preparing to train the model... \n")

for epoch in range(num_of_epochs):
    for (words,labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        #forward
        output = model(words)
        loss = criteria(output,labels)
        
        #backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if (epoch+1)%100==0:
        print(f"epoch = {epoch+1}/{num_of_epochs}; loss = {loss.item():.4f}")

print(f"\n final epoch = {epoch+1}/{num_of_epochs}; final loss = {loss.item():.4f} \n")

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "all_words": all_words,
    "hidden_size": hidden_size,
    "tags": tags
}

trained_file = "data.pth"
torch.save(data,trained_file)

print("*****Model training successful*****")
print(f"File saved to {trained_file}")



