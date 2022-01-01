import torch
import torch.nn as nn

class Net:
    def __init__(self,input_size,hidden_size=128,num_layers=1,drop=0):
        self.lstm = nn.LSTM(input_size=input_size,
                    hidden_size=hidden_size,num_layers=num_layers,dropout=drop)
        self.fc = nn.Linear(hidden_size,input_size)


    def forward(self,x):
        x,_ = self.lstm(x)
        x = self.fc(x)
        return x

    def save_model(self,f):
        torch.save(self.state_dict(),f)

    def load_model(self,f):
        self.load_state_dict(torch.load(f),map_location='cpu')