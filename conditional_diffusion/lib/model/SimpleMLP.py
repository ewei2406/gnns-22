import dgl
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMLP(nn.Module):
    def __init__(self, in_size, hid_size, out_size, lr, dropout, weight_decay):
        super(SimpleMLP, self).__init__()

        self.conv1 = torch.nn.Linear(in_size, hid_size)
        self.conv2 = torch.nn.Linear(hid_size, out_size)
        self.lr = lr
        self.dropout = dropout
        self.weight_decay = weight_decay

    def forward(self, feat):
        feat = self.conv1(feat)
        feat = F.relu(feat)
        feat = F.dropout(feat, self.dropout, training=self.training)
        feat = self.conv2(feat)

        return F.log_softmax(feat, dim=1).squeeze()

    def fit(self, feat, labels, epochs: int, mask: torch.Tensor | None=None, verbose=True):
        self.train()
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        t = tqdm(range(epochs), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', disable=not verbose)
        t.set_description("GCN Training")

        for epoch in t:
            optimizer.zero_grad()
            predictions = self(feat)
            if mask != None:
                loss = F.cross_entropy(predictions[mask], labels[mask])
            else:
                loss = F.cross_entropy(predictions, labels)
            loss.backward()
            optimizer.step()
            t.set_postfix({"loss": round(loss.item(), 2)})

        return loss.item()