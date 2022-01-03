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

    def load_model(self):
        self.load_state_dict(torch.load('checkpoints/cbow_model.pt',map_location='cpu'))


class Net(nn.Module):
    def __init__(self,input_size,embedding,hidden_size=128,num_layers=1,drop=0):
        super().__init__()
        self.embedding = embedding
        self.embedding.requires_grad_(False)
        self.embed_out_size = self.embedding.shape[0] 

        assert input_size==embedding.shape[1],f'Input size ({input_size}) should match input size of embedding {embedding.shape[1]}'

        self.lstm = nn.LSTM(input_size=self.embed_out_size,
                    hidden_size=hidden_size,num_layers=num_layers,
                    dropout=drop if num_layers>1 else 0,
                    batch_first=True)
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(hidden_size,input_size)


    def forward(self,x,hidden=None):
        #x of shape (bs,timesteps,input_size)
        x = self.embed_encode(x)
        x,hidden = self.lstm(x)
        x = self.drop(x[:,-1,])
        x = self.fc(x)
        return x,hidden

    
    def embed_encode(self,x):
        out = torch.zeros((x.shape[0],x.shape[1],self.embed_out_size))

        for t in range(x.shape[1]):
            out[:,t] = (self.embedding@x[:,t].T).T

        return out

    def save_model(self):
        torch.save(self.state_dict(),'checkpoints/classifier_model.pt')

    def load_model(self):
        self.load_state_dict(torch.load('checkpoints/classifier_model.pt',map_location='cpu'))