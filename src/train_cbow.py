import torch
import torch.nn as nn
from model import CBOW
from data import CBOWData
from torch.utils.data import DataLoader
import pickle
from tqdm import tqdm

import matplotlib.pyplot as plt

train = DataLoader(CBOWData(train=True),batch_size=32,shuffle=True)
test = DataLoader(CBOWData(train=False),batch_size=32,shuffle=True)

vocab = pickle.load(open('../data/vocab.pkl','rb'))
net = CBOW(len(vocab),hidden_size=128)

lossfn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(),lr=3e-3)

train_loss_over_time = []
test_loss_over_time = []

for epoch in tqdm(range(30)):
    net.train()

    batch_train_loss = []
    batch_test_loss = []

    for x,y in train:
        pred = net(x)
        loss = lossfn(pred,y)
        batch_train_loss.append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


    net.eval()

    with torch.no_grad():
        for x,y in test:
            pred = net(x)
            loss = lossfn(pred,y)
            batch_test_loss.append(loss.item())

    train_loss_over_time.append(sum(batch_train_loss)/len(batch_train_loss))
    test_loss_over_time.append(sum(batch_test_loss)/len(batch_test_loss))

    if len(test_loss_over_time)==1 or (test_loss_over_time[-1]<test_loss_over_time[-2]):
        net.save_model()


    plt.plot(train_loss_over_time,label='train loss')
    plt.plot(test_loss_over_time,label='test loss')
    plt.legend()
    plt.savefig('plots/cbow_loss_plot.png')
    plt.close('all')
