import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn
from torch.nn import functional as F

import pickle

import json
import re

import numpy as np
import pymorphy2
import torchtext.vocab as vocab


morph = pymorphy2.MorphAnalyzer()
vectors = vocab.Vectors('../data/ruwiki_20180420_100d.txt') # file created by gensim

PAD_IDX = len(vectors.stoi) - 2
UNK_IDX = len(vectors.stoi) - 1
TOTAL_EMBS = len(vectors)

class SimilarityNet(nn.Module):
    def __init__(self):
        super(SimilarityNet, self).__init__()
        self.embs = nn.Embedding(len(vectors), embedding_dim=100, padding_idx=PAD_IDX)
        self.process = nn.Sequential(
            nn.Conv1d(100, 128, 15),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.05),
            nn.Conv1d(128, 64, 9),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.05),
            nn.Conv1d(64, 32, 5),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.05),
            nn.Conv1d(32, 32, 5),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.05),
        )
        self.fc = nn.Sequential(
            nn.Linear(32, 1),
        )
        
    def forward(self, x):
        x = self.process(self.embs(x).permute(0, 2, 1))
        x = torch.mean(x, dim=-1)
        return self.fc(x)
    
class EmojiNet(nn.Module):
    def __init__(self):
        super(EmojiNet, self).__init__()
        self.embs = nn.Embedding(len(vectors), embedding_dim=100, padding_idx=PAD_IDX)
        self.process = nn.Sequential(
            nn.Conv1d(100, 128, 15),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.05),
            nn.Conv1d(128, 128, 9),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.05),
            nn.Conv1d(128, 32, 5),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.05),
        )
        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 624),            
        )
        
    def forward(self, x):
        x = self.process(self.embs(x).permute(0, 2, 1))
        x = x.view(x.shape[0], -1)
        return self.fc(x)

simnet = SimilarityNet()
simnet.load_state_dict(torch.load('../models/simnet.state_dict'))
simnet.eval()
emojinet = EmojiNet()
emojinet.load_state_dict(torch.load('../models/emojinet.state_dict'))
emojinet.eval()

def normal_form(word):
    return morph.parse(word)[0].normal_form

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def preprocess(text):
    return ' '.join([normal_form(i) for i in re.findall(r'[А-я]+', text) if len(i) > 2])


def vectorize(text):
    try:
        text = preprocess(text['text'])
    except TypeError as e:
        return None
    inds = []
    if len(text.split()) < 3:
        return None
    for word in text.split():
        try:
            inds.append(vectors.stoi[word])
        except KeyError:
            inds.append(UNK_IDX)
    if len(inds) < 2:
        return None
    while len(inds) < 90:
        inds.append(PAD_IDX)
    return inds[:90]

    
def process_similarity(x):
    x = vectorize(x)
    if not x:
        return None
    x = torch.Tensor(x).long().unsqueeze(0)
    ans = sigmoid(simnet(x).item())
    return ans    


with open('../models/index2emoji.pkl', 'rb') as f:
    i2e = pickle.load(f)

def process_emoji(x):
    x = vectorize(x)
    if not x:
        return None
    x = torch.Tensor(x).long().unsqueeze(0)
    x = emojinet(x)
    ans = np.argsort(torch.sigmoid(x).detach().numpy())[:, -3::]
    return [i2e[i] for i in ans[0]]    
    
    
