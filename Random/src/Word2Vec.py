import torch.nn as nn
import torch.nn.functional as F


class Word2Vec(nn.Module):

    def __init__(self, vocab_size, context_size, embedding_dim=128, skipgram=False):
        super(Word2Vec, self).__init__()

        self.context_size = 2 * context_size
        self.skipgram = skipgram
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim *
                      (1 if self.skipgram else self.context_size), 128),
            nn.ReLU(),
            nn.Linear(128, vocab_size *
                      (self.context_size if self.skipgram else 1))
        )

    def forward(self, X):
        if self.skipgram:
            return self.skipgram_forward(X)
        else:
            return self.cbow_forward(X)

    def skipgram_forward(self, X):
        embeds = self.embedding(X).view((X.shape[0], -1))
        out = self.layers(embeds)
        log_probs = out.view(X.shape[0], self.context_size, -1).swapaxes(1, 2)
        return F.log_softmax(log_probs, dim=1)

    def cbow_forward(self, X):
        embeds = self.embedding(X).view(X.shape[0], -1)
        log_probs = self.layers(embeds)
        return F.log_softmax(log_probs, dim=1)
