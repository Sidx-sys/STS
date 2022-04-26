import numpy as np
from .Vocab import Vocab


class TFIDFVectorizer:

    def __init__(self, vocab: Vocab):
        self.vocab = vocab
        self.idf = np.zeros(len(self.vocab))

    def fit(self, sentences):

        for doc in sentences:
            doc = self.vocab.sequencify(doc, addEOSBOS=True)
            occ_vec = list(set(doc))
            np.add.at(self.idf, occ_vec, 1)

        for i in range(len(self.vocab)):
            self.idf[i] = np.log(
                (len(sentences) + 1) / (self.idf[i] + 1)) + 1

    def transform(self, sentences):

        out = np.zeros((len(sentences), len(self.vocab)))
        for i, doc in enumerate(sentences):
            doc = list(self.vocab.sequencify(doc, addEOSBOS=True))
            tf = np.zeros(len(self.vocab))
            for idx in set(doc):
                tf[idx] = doc.count(idx)

            out[i] = tf * self.idf

        return out
