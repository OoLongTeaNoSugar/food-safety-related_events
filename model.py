# -*- ecoding:utf-8 -*-

import torch
import torch.nn as nn
import torch.utils.data

# model1
class Bi_GRU(nn.Module):
    def __init__(self):
        super(Bi_GRU, self).__init__()
        hidden_size = 60

        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False

        self.embedding_dropout = nn.Dropout2d(0.1)
        self.bi_gru = nn.GRU(embed_size, hidden_size, bidirectional=True, batch_first=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(120, 16)
        self.out = nn.Linear(16, 1)

    def forward(self, x):
        h_embedding = self.embedding(x)

        # bigru
        h_gru, _ = self.bi_gru(h_embedding)

        # attention

        # pool
        max_pool, _ = torch.max(h_gru, 1)

        h_relu = self.relu(self.linear(max_pool))
        h_drop = self.dropout(h_relu)
        out = self.out(h_drop)

        return out

# Attention
