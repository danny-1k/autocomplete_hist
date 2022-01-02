import json
import pickle
import argparse
import random

class Proc:
    def __init__(self,freq_thresh,past_count):
        self.json = json.load(open('../data/data.json')) 
        self.vocab = ['UNK']
        self.word_count = {}
        self.freq_thresh = freq_thresh
        self.past_count = past_count

    def remove_punctuation(self):
        for idx in self.json:
            if self.json[idx]['title'] == 'Search' and not(self.json[idx]['value'].startswith('http')):
                self.json[idx] = ''.join([ c for c in self.json[idx]['value'] if c.isalnum() or c==' '])

    def to_lower(self):
        for idx in self.json:
            if type(self.json[idx]) == str:
                self.json[idx] = self.json[idx].lower()
            else:
                self.json[idx] = self.json[idx]['value'].lower()
    
    def add_word(self,word):
        if self.word_count.get(word) == None:
            self.word_count[word] = 1
        else:
            self.word_count[word] += 1
            if (self.word_count[word] >= self.freq_thresh) and not(word in self.vocab):
                self.vocab.append(word)

    def save_vocab(self):
        word_idx = {v:i for i,v in enumerate(self.vocab)}
        idx_word = {i:v for i,v in enumerate(self.vocab)}

        pickle.dump(self.vocab,open('../data/vocab.pkl','wb'))
        pickle.dump(word_idx,open('../data/word_idx.pkl','wb'))
        pickle.dump(idx_word,open('../data/idx_word.pkl','wb'))


    def build_vocab_freq(self):
        for idx in self.json:
            words = self.json[idx].split()
            [self.add_word(w) for w in words]

    def save_data(self):

        self.remove_punctuation()
        self.to_lower()
        self.build_vocab_freq()

        X = []
        Y = []

        for idx in self.json:
            words = self.json[idx].split()
            words = [word if word in self.vocab else 'UNK' for word in words]
            if len(words) > self.past_count:
                for n in range(len(words)-self.past_count):
                    x = words[n:self.past_count+n]
                    y = words[self.past_count+n]
                    X.append(x)
                    Y.append(y)

        self.save_vocab()
        data = list(zip(X,Y))
        random.shuffle(data)
        X,Y = zip(*data)

        pickle.dump(X,open('../data/X.pkl','wb'))
        pickle.dump(Y,open('../data/Y.pkl','wb'))


    def save_cbow_data(self,window):
        
        X = []
        Y = []

        for idx in self.json:
            words = self.json[idx].split()
            words = [word if word in self.vocab else 'UNK' for word in words]
            if len(words) > ((2*window)+1):
                for n in range(len(words)-(2*window)+1):

                    full = words[n:(2*window)+1+n]

                    front = full[:window]
                    back = full[-window:]
                    x = front+back
                    y = full[window]

                    X.append(x)
                    Y.append(y)

                
                pickle.dump(X,open('../data/cbowX.pkl','wb'))
                pickle.dump(Y,open('../data/cbowY.pkl','wb'))


parser = argparse.ArgumentParser()

parser.add_argument('--past_count',help='Number of previous words as context')
parser.add_argument('--freq_thresh',help='How many times a token has to appear in the dataset')
parser.add_argument('--window',help='window for CBOW dataset')

args = parser.parse_args()

past_count = int(args.past_count or 3)
freq_thresh = int(args.freq_thresh or 5)
window = int(args.window or 1)

data_proc = Proc(freq_thresh=freq_thresh,past_count=past_count)
data_proc.save_data()
data_proc.save_cbow_data(window)