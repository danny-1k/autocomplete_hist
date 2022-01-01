import torch
import torch.nn as nn
from model import Net
from data import AutocompleteData
from torch.utils.data import DataLoader

import pickle

import matplotlib.pyplot as plt

from tqdm import tqdm

train = DataLoader(AutocompleteData(train=True))
test = DataLoader(AutocompleteData(train=False))

vocab = pickle.load(open('../data/vocab.pkl'))

net = Net(len(vocab))

lossfn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(),lr=1e-4)

train_loss_over_time = []
test_loss_over_time = []

for epoch in tqdm(range(30)):
    net.train()
    batch_train_loss = []
    batch_test_loss = []
    
    for x,y in train:
        pred = net(x.float())
        loss = lossfn(x,y)
        batch_train_loss.append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


    net.eval()
    with torch.no_grad():
        for x,y in test:
            pred = net(x.float())
            loss = lossfn(x,y)
            batch_test_loss.append(loss.item())

    train_loss_over_time.append(sum(batch_train_loss)/len(batch_train_loss))
    test_loss_over_time.append(sum(batch_test_loss)/len(batch_test_loss))

    if len(test_loss_over_time) or (test_loss_over_time[-1]<test_loss_over_time[-2]):
        net.save_model('checkpoints/model.pt')


    plt.plot(train_loss_over_time,label='train loss')
    plt.plot(test_loss_over_time,label='test loss')
    plt.legend()
    plt.savefig('plots/loss_plot.png')
    plt.close('all')