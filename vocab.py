import os
import torch
from collections import Counter, OrderedDict
import numpy as np


class Vocab(object):
    def __init__(self):
        self.counter = Counter()
        self.counter_word = Counter()
        self.max_len = 0
        self.data = []

    def tokenize(self, line):
        line = line.strip()
        line = line.lower()
        symbols = line
        return symbols

    def count_file(self, path):
        assert os.path.exists(path), 'file [{}] does not exists.'.format(path)
        with open(path, 'r') as f:
            for idx, line in enumerate(f):
                line = line.strip().lower()
                word_list = line.split(' ')
                self.counter_word.update(word_list)
                for word in word_list:
                    if self.max_len < len(word):
                        self.max_len = len(word)
                symbols = self.tokenize(line)
                self.counter.update(symbols)

    def build_vocab(self):
        self.idx2sym = []
        self.sym2idx = OrderedDict()
        for sym, cnt in self.counter.most_common():
            self.add_symbol(sym)
        self.id2word = []
        self.word2idx = OrderedDict()
        for word, cnt in self.counter_word.most_common():
            self.add_word(word)
        print("Word Vocabulary Size : %d" % len(self.id2word))
        print("Character Vocabulary Size : %d" % len(self.idx2sym))
        print("Max Length of Word - 2 : %d" % self.max_len)

    def add_symbol(self, sym):
        if sym not in self.sym2idx:
            self.idx2sym.append(sym)
            self.sym2idx[sym] = len(self.idx2sym) - 1

    def add_word(self, word):
        if word not in self.word2idx:
            self.id2word.append(word)
            self.word2idx[word] = len(self.id2word) - 1

    def encode_file(self, data):
        encoded_word = []
        encoded = []
        for word in data:
            if word in self.word2idx:
                encoded_word.append(self.word2idx[word])
        for idx, word in enumerate(data):
            encoded.append(self.convert_to_tensor(word))
        return torch.LongTensor(encoded_word), torch.LongTensor([t.numpy() for t in encoded])

    def convert_to_tensor(self, symbols):
        return torch.LongTensor(self.get_indices(symbols))

    def get_indices(self, symbols):
        char_idx = []
        for sym in symbols:
            char_idx.append(self.get_idx(sym))
        for i in range(0, self.max_len - len(char_idx)):
            char_idx.append(0)
        return char_idx

    def get_idx(self, sym):
        if sym in self.sym2idx:
            return self.sym2idx[sym]

    def __len__(self):
        return len(self.idx2sym)

    def get_len(self):
        return len(self.id2word)
