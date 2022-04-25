import re
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from collections import defaultdict
import math
import numpy as np
import scipy


class Tokenizer:
    def __init__(self):
        self.regex_subs = {
            r'(https?:\/\/)\S+': "0URL0",
            r'(?<!http://)www\.\S+': "0URL0",
            r'(\W)(?=\1)': '',
            r'(?<=[a-zA-Z])(\-)(?=[a-zA-Z])': ''
        }

        self.punctuations = [r'\.', r'\.{2,}',
                             r'\!+', r'\:+', r'\;+', r'\"+', r"\'+", r'\?+', r'\,+', r'\(|\)|\[|\]|\{|\}|\<|\>']

        self.delimiter = '<SPLIT>'
        self.stemmer = SnowballStemmer(language='english')
        self.stop_words = set(stopwords.words('english'))

    def clean_line(self, line):
        for pattern, rep in self.regex_subs.items():
            line = re.sub(pattern, rep, line)
        for pattern in self.punctuations:
            line = re.sub(pattern, '', line)
        return line.lower()

    def tokenize_line(self, line):
        line = re.sub('\s+', self.delimiter, line)

        token_list = [x.strip()
                      for x in line.split(self.delimiter) if x.strip() != '']

        return token_list

    def clean_and_tokenize(self, lines):
        if isinstance(lines, list):
            cleaned_tokens = []
            for line in lines:
                if not len(line.strip()):
                    continue
                line = self.clean_line(line)
                tokens = self.tokenize_line(line)
                cleaned_tokens.append(tokens)
            return cleaned_tokens
        else:
            line = self.clean_line(lines)
            tokens = self.tokenize_line(line)
            return tokens

    def _clean(self, line):
        for punc in self.punctuations:
            line = re.sub(punc, '', line)

        line = self.clean_line(line)

        cleaned = []
        for token in line.split():
            if token not in self.stop_words:
                cleaned.append(self.stemmer.stem(token))

        return " ".join(cleaned)

    def clean(self, lines):
        if isinstance(lines, list):
            cleaned_lines = []
            for line in lines:
                if not len(line.strip()):
                    continue
                line = self._clean(line)
                cleaned_lines.append(line)
            return cleaned_lines
        else:
            line = self._clean(lines)
            return line


def file_to_data(path):
    try:
        with open(path) as f:
            data = [[x.rstrip().split('\t')[1], x.rstrip().split(
                '\t')[2], x.rstrip().split('\t')[0]] for x in f.readlines()]
    except FileNotFoundError:
        print("File does not exist")
        return

    formatted_data = []
    for row in data:
        formatted_data.append([row[0], row[1], row[2]])

    return formatted_data


def train_dev_test_split(data, dev_size=0.15, test_size=0.10):
    X, y = [], []

    for row in data:
        X.append([row[0], row[1]])
        y.append(float(row[2]))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=69)

    return X_train, X_test, y_train, y_test


class TfidfVectorizer():
    def __init__(self):
        self.tf_l = []
        self.tf_r = []

        self.idf = defaultdict(int)

        self.vocab = {}
        self.data_l = None
        self.data_r = None
        self.vocab_len = 0
        self.num_docs = 0

    def create_vocab(self, data):
        print("Creating Vocabulary...")
        self.data_l = []
        self.data_r = []
        for items in data:
            self.data_l.append(items[0].split())
            self.data_r.append(items[1].split())
        self.num_docs = len(self.data_l)

        for text in self.data_l:
            for token in text:
                if not token in self.vocab:
                    self.vocab[token] = self.vocab_len
                    self.vocab_len += 1

        for text in self.data_r:
            for token in text:
                if not token in self.vocab:
                    self.vocab[token] = self.vocab_len
                    self.vocab_len += 1

    def compute_tf(self):
        print("Computing TF Scores...")
        for text in self.data_l:
            d = defaultdict(int)
            for token in text:
                d[self.vocab[token]] += 1
            self.tf_l.append(d)

        for text in self.data_r:
            d = defaultdict(int)
            for token in text:
                d[self.vocab[token]] += 1
            self.tf_r.append(d)

    def compute_idf(self):
        print("Computing IDF Scores...")
        for token in self.vocab:
            df = 0
            for text in self.data_l:
                if token in text:
                    df += 1

            for text in self.data_r:
                if token in text:
                    df += 1

            self.idf[self.vocab[token]] = math.log(
                (1 + self.num_docs)/(1 + df)) + 1

    def fit_transform(self, data):
        self.create_vocab(data)
        self.compute_tf()
        self.compute_idf()
        print("Creating TF-IDF Vectors...")
        X_l = np.zeros((self.num_docs, self.vocab_len), dtype='float32')
        X_r = np.zeros((self.num_docs, self.vocab_len), dtype='float32')

        for i in range(self.num_docs):
            for token in self.data_l[i]:
                X_l[i][self.vocab[token]] = self.tf_l[i][self.vocab[token]
                                                         ] * self.idf[self.vocab[token]]

        for i in range(self.num_docs):
            for token in self.data_r[i]:
                X_r[i][self.vocab[token]] = self.tf_r[i][self.vocab[token]
                                                         ] * self.idf[self.vocab[token]]

        return X_l, X_r

    def transform(self, data):
        data_l = []
        data_r = []
        tf_l = []
        tf_r = []
        num_docs = len(data)

        for items in data:
            data_l.append(items[0].split())
            data_r.append(items[1].split())

        for text in data_l:
            d = defaultdict(int)
            for token in text:
                if token in self.vocab:
                    d[self.vocab[token]] += 1
            tf_l.append(d)

        for text in data_r:
            d = defaultdict(int)
            for token in text:
                if token in self.vocab:
                    d[self.vocab[token]] += 1
            tf_r.append(d)

        X_l = np.zeros((num_docs, self.vocab_len), dtype='float32')
        X_r = np.zeros((num_docs, self.vocab_len), dtype='float32')

        for i in range(num_docs):
            for token in data_l[i]:
                if token in self.vocab:
                    X_l[i][self.vocab[token]] = tf_l[i][self.vocab[token]
                                                        ] * self.idf[self.vocab[token]]

        for i in range(num_docs):
            for token in data_r[i]:
                if token in self.vocab:
                    X_r[i][self.vocab[token]] = tf_r[i][self.vocab[token]
                                                        ] * self.idf[self.vocab[token]]

        return X_l, X_r


def cosine_similarity(vec_1, vec_2):
    return vec_1@vec_2.T/(np.linalg.norm(vec_1) * np.linalg.norm(vec_2))


def output_preds(data_l, data_r, min_output=0.2):
    return [5 * (min_output if cosine_similarity(data_l[i], data_r[i]) == 0.0 else cosine_similarity(data_l[i], data_r[i])) for i in range(len(data_l))]


if __name__ == '__main__':
    year = '2014'
    data_file = 'images.test.tsv'
    path = f'data/sts/semeval-sts/{year}/{data_file}'
    data = file_to_data(path)

    tokenizer = Tokenizer()
    cleaned_data = []
    for row in data:
        cleaned_data.append(
            [tokenizer.clean(row[0]), tokenizer.clean(row[1]), row[2]])

    X_train_data, X_test_data, y_train, y_test = train_dev_test_split(
        cleaned_data)

    vectorizer = TfidfVectorizer()
    X_train_l, X_train_r = vectorizer.fit_transform(X_train_data)
    X_test_l, X_test_r = vectorizer.transform(X_test_data)
    preds = output_preds(X_test_l, X_test_r)
    pearson_score, _ = scipy.stats.pearsonr(preds, y_test)
    print(pearson_score)
