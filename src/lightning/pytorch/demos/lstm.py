"""Demo of a simple LSTM language model.

Code is adapted from the PyTorch examples at
https://github.com/pytorch/examples/blob/main/word_language_model
"""
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler
from lightning.pytorch.demos import WikiText2
import lightning as L


class LSTMModel(nn.Module):
    def __init__(self, vocab_size=33278, ninp=512, nhid=512, nlayers=4, dropout=0.2):
        super().__init__()
        self.vocab_size = vocab_size
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(vocab_size, ninp)
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout, batch_first=True)
        self.decoder = nn.Linear(nhid, vocab_size)
        self.nlayers = nlayers
        self.nhid = nhid
        self.init_weights()

    def init_weights(self):
        nn.init.uniform_(self.encoder.weight, -0.1, 0.1)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -0.1, 0.1)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        decoded = decoded.view(-1, self.vocab_size)
        return F.log_softmax(decoded, dim=1), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (
            weight.new_zeros(self.nlayers, bsz, self.nhid),
            weight.new_zeros(self.nlayers, bsz, self.nhid),
        )


class SequenceSampler(Sampler[List[int]]):
    def __init__(self, dataset, batch_size):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.chunk_size = len(self.dataset) // self.batch_size

    def __iter__(self):
        n = len(self.dataset)
        for i in range(self.chunk_size):
            yield list(range(i, n - (n % self.batch_size), self.chunk_size))

    def __len__(self):
        return self.chunk_size


class LightningLSTM(L.LightningModule):
    def __init__(self, vocab_size: int = 33278):
        super().__init__()
        self.model = LSTMModel(vocab_size=vocab_size)
        self.hidden = None

    def on_train_epoch_end(self):
        self.hidden = None

    def training_step(self, batch, batch_idx):
        input, target = batch
        if self.hidden is None:
            self.hidden = self.model.init_hidden(input.size(0))
        self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())
        output, self.hidden = self.model(input, self.hidden)
        loss = F.nll_loss(output, target.view(-1))
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def prepare_data(self) -> None:
        WikiText2(download=True)

    def train_dataloader(self) -> DataLoader:
        dataset = WikiText2()
        return DataLoader(dataset, batch_sampler=SequenceSampler(dataset, batch_size=20))

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=20.0)
