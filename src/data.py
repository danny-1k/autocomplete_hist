import random
import pickle
import torch
from torch.utils.data import Dataset


class AutocompleteData(Dataset):
    def __init__(self,train=True):

        self.X = pickle.load(open('../data/X.pkl','rb'))
        self.Y = pickle.load(open('../data/Y.pkl','rb'))

        data = list(zip(self.X,self.Y))
        random.shuffle(data)
        self.X,self.Y = zip(*data)

        pct = int(.8*len(self.Y))

        if train:
            self.X = self.X[:pct]
            self.Y = self.Y[:pct]
        else:
            self.X = self.X[pct:]
            self.Y = self.Y[pct:]

        self.word_to_idx = pickle.load(open('../data/word_idx.pkl','rb'))
        self.vocab = pickle.load(open('../data/vocab.pkl','rb'))

    def __getitem__(self, index):
        x = self.X[index]
        y = self.Y[index]
        x = torch.eye(len(self.vocab))[[self.word_to_idx[i] for i in x]]
        y = self.word_to_idx[y]
        return x,y

    def __len__(self):
        return len(self.Y)
