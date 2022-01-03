import torch
import argparse
from flask import Flask,render_template,request
import json

from model import CBOW,Net
import pickle

vocab = pickle.load(open('../data/vocab.pkl','rb'))
word_idx = pickle.load(open('../data/word_idx.pkl','rb'))
idx_to_word = pickle.load(open('../data/idx_word.pkl','rb'))

cbow_model = CBOW(len(vocab),128)
cbow_model.load_model()

embedding = cbow_model.fc1.weight
net = Net(len(vocab),hidden_size=128,drop=0,num_layers=1,embedding=embedding)
net.load_model()
net.eval()

net.requires_grad_(False)

def preprocess_text(text):
    text = text.lower()
    text = ''.join([i for i in text if i.isalnum() or i==' '])
    text = text.split()
    return text

def oov(text):
    text = [i  if i in vocab else 'UNK' for i in text]
    return text

def encode_tokens(tokens):
    tokens = [word_idx[t] for t in tokens]
    tokens = torch.eye(len(vocab))[tokens]

    return tokens

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/suggest',methods=['POST'])
def suggest():
    original_text = request.json['text']
    text = preprocess_text(original_text)
    text = oov(text)
    oov_text = text
    text = encode_tokens(text)
    suggestions = []

    next,hidden = net(text.unsqueeze(0))

    probs = torch.softmax(next.squeeze(),0)
    top_5 = probs.numpy().argsort()[::-1][:5]

    for s in top_5:
        if probs[s] >=.02:
            suggestions.append(idx_to_word[s])

    return json.dumps({'suggestions':suggestions})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug')

    args = parser.parse_args()
    debug = bool(args.debug)

    app.run(debug=debug)