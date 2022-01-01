import torch
import torch.nn as nn

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

    def save_model(self,f):
        torch.save(self.state_dict(),f)

    def load_model(self,f):
        self.load_state_dict(torch.load(f,map_location='cpu'))