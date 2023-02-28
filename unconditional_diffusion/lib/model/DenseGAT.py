import dgl
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseGAT(nn.Module):
    def __init__(self, in_size, hid_size, out_size, lr=0.01, dropout=0.5, weight_decay=5e-4):
        super(DenseGAT, self).__init__()

        self.conv1 = dgl.nn.GATConv(in_size, hid_size, 1)
        self.conv2 = dgl.nn.GATConv(hid_size, out_size, 1)
        self.lr = lr
        self.dropout = dropout
        self.weight_decay = weight_decay

    def forward(self, graph, feat):
        feat = self.conv1(graph, feat)
        feat = F.relu(feat)
        feat = F.dropout(feat, self.dropout, training=self.training)
        feat = self.conv2(graph, feat)

        return F.log_softmax(feat, dim=1).squeeze()

    def fit(self, graph: dgl.DGLGraph, labels: torch.Tensor, epochs: int, mask: torch.tensor=None, verbose=True):
        self.train()
        print(type(graph))
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        t = tqdm(range(epochs), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', disable=not verbose)
        t.set_description("GAT Training")

        for epoch in t:
            optimizer.zero_grad()
            predictions = self(graph, graph.ndata['feat'])
            if mask != None:
                loss = F.cross_entropy(predictions[mask], labels[mask])
            else:
                loss = F.cross_entropy(predictions, labels)
            loss.backward()
            optimizer.step()
            t.set_postfix({"loss": round(loss.item(), 2)})

        return loss.item()

