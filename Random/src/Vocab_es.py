from collections import defaultdict
import re
import numpy as np
from typing import List, Union

from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords


class Vocab_es:

    stemmer = SnowballStemmer(language='spanish')
    stop_words = set(stopwords.words('spanish'))

    def __init__(self, sentences: List[str], oov_threshold: int = 2, remove_stopwords=False) -> None:

        self.oov_token = '<OOV>'
        self.bos_token = '<BOS>'
        self.eos_token = '<EOS>'
        self.pad_token = '<PAD>'
        self.remove_stopwords = remove_stopwords

        self.oov_threshold = oov_threshold

        self.vocab = {self.oov_token, self.bos_token,
                      self.eos_token, self.pad_token}
        self.word2idx = {
            self.pad_token: 0,
            self.oov_token: 1,
            self.bos_token: 2,
            self.eos_token: 3
        }
        self.idx2word = {v: k for k, v in self.word2idx.items()}

        tokenized_list = self.tokenize(sentences)

        self.build_vocab(tokenized_list)

    def tokenize(self, sentences: List[str]) -> List[List[str]]:
        tokenized_list = []
        for sen in sentences:
            sen = sen.lower()
            sen = re.sub(r'\s+', ' ', sen)
            sen = re.sub(r'[^a-z ]', '', sen)

            # Optionally remove stop words and stem
            if self.remove_stopwords:
                cleaned_sen = []
                for tok in sen.split():
                    if tok not in Vocab.stop_words:
                        cleaned_sen.append(Vocab.stemmer.stem(tok))

                tokenized_list.append(cleaned_sen)
            else:
                tokenized_list.append(sen.split())
        return tokenized_list

    def build_vocab(self, tokenized_list: List[List[str]]) -> None:
        freq = defaultdict(int)

        for sen in tokenized_list:
            for token in sen:
                freq[token] += 1

        for token, f in freq.items():
            if f > self.oov_threshold:
                idx = len(self.vocab)
                self.vocab.add(token)
                self.word2idx[token] = idx
                self.idx2word[idx] = token

    def get_idx(self, word: str) -> int:
        if word in self.vocab:
            return self.word2idx[word]
        return self.word2idx[self.oov_token]

    def get_word(self, idx: int) -> str:
        assert idx < len(self), "Index out of Range"
        return self.idx2word[idx]

    def sequencify(self, sentence: str, addEOSBOS: bool = False) -> np.ndarray:
        tokens = self.tokenize([sentence])[0]
        seq = []

        for tok in tokens:
            seq.append(self.get_idx(tok))

        if addEOSBOS:
            seq = [self.get_idx(self.bos_token)] + seq + \
                [self.get_idx(self.eos_token)]
        return np.array(seq)

    def sentencify(self, seq: Union[np.ndarray, List[int]]) -> List[str]:
        sentence = []

        for idx in seq:
            sentence.append(self.get_word(idx))

        return sentence

    def __len__(self) -> int:
        return len(self.vocab)

    def __contains__(self, item) -> bool:
        return item in self.vocab

    def __iter__(self) -> iter:
        return iter(self.vocab)
