import torch
import torch.nn as nn
from model import Net
from data import AutocompleteData
from torch.utils.data import DataLoader

import pickle

import matplotlib.pyplot as plt

from tqdm import tqdm

train = DataLoader(AutocompleteData(train=True),batch_size=32,shuffle=True)
test = DataLoader(AutocompleteData(train=False),batch_size=32,shuffle=True)

vocab = pickle.load(open('../data/vocab.pkl','rb'))
net = Net(len(vocab),hidden_size=128,drop=0,num_layers=1)

lossfn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(),lr=3e-3)

train_loss_over_time = []
test_loss_over_time = []

for epoch in tqdm(range(30)):
    net.train()
    batch_train_loss = []
    batch_test_loss = []
    
    for x,y in train:
        pred,_ = net(x.float())
        loss = lossfn(pred,y)
        batch_train_loss.append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


    net.eval()
    with torch.no_grad():
        for x,y in test:
            pred,_ = net(x.float())
            loss = lossfn(pred,y)
            batch_test_loss.append(loss.item())

    train_loss_over_time.append(sum(batch_train_loss)/len(batch_train_loss))
    test_loss_over_time.append(sum(batch_test_loss)/len(batch_test_loss))

    if len(test_loss_over_time)==1 or (test_loss_over_time[-1]<test_loss_over_time[-2]):
        net.save_model()


    plt.plot(train_loss_over_time,label='train loss')
    plt.plot(test_loss_over_time,label='test loss')
    plt.legend()
    plt.savefig('plots/loss_plot.png')
    plt.close('all')