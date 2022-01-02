import torch
import torch.nn as nn

class CBOW(nn.Module):
    def __init__(self,input_size,hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,input_size)

    def forward(self,x):
        x = sum([*x]).float() #(bs,input_size)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

    def save_model(self):
        torch.save(self.state_dict(),'checkpoints/cbow_model.pt')

    def load_model(self,f):
        self.load_state_dict(torch.load('checkpoints/cbow_model.pt',map_location='cpu'))


class Net(nn.Module):
    def __init__(self,input_size,hidden_size=128,num_layers=1,drop=0):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                    hidden_size=hidden_size,num_layers=num_layers,
                    dropout=drop if num_layers>1 else 0,
                    batch_first=True)
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(hidden_size,input_size)


    def forward(self,x,hidden=None):
        x,hidden = self.lstm(x)
        x = self.drop(x[:,-1,])
        x = self.fc(x)
        return x,hidden

    def save_model(self):
        torch.save(self.state_dict(),'checkpoints/classifier_model.pt')

    def load_model(self,f):
        self.load_state_dict(torch.load('checkpoints/cbow_model.pt',map_location='cpu'))