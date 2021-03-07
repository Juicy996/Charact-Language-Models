import os
import torch
from vocab import Vocab


class Corpus(object):
    def __init__(self, trn_path, val_path, tst_path, batch_size):
        self.trn_path = trn_path
        self.val_path = val_path
        self.tst_path = tst_path
        self.batch_size = batch_size
        self.vocab = Vocab()
        self.vocab.count_file(trn_path)
        self.vocab.count_file(val_path)
        self.vocab.count_file(tst_path)
        self.vocab.build_vocab()
        self.trn_data, self.trn_num_batches = self.batchify(trn_path, self.batch_size)
        self.val_data, self.val_num_batches = self.batchify(val_path, self.batch_size)
        self.tst_data, self.tst_num_batches = self.batchify(tst_path, self.batch_size)
        self.trn_word, self.trn_char = self.vocab.encode_file(self.trn_data)
        self.len_word = len(self.trn_word)
        self.val_word, self.val_char = self.vocab.encode_file(self.val_data)
        self.len_word_val = len(self.val_word)
        self.tst_word, self.tst_char = self.vocab.encode_file(self.tst_data)
        self.trn_word = self.trn_word.view(self.batch_size, -1)
        # print(self.trn_word.size())
        self.trn_char = self.trn_char.view(self.batch_size, -1, self.vocab.max_len)
        # print(self.trn_char.size())
        self.val_word = self.val_word.view(self.batch_size, -1)
        self.val_char = self.val_char.view(self.batch_size, -1, self.vocab.max_len)

    def batchify(self, path, batch_size):
        assert os.path.exists(path), 'file [{}] does not exists.'.format(path)
        data = []
        with open(path, 'r') as f:
            for line in f:
                line = line.lower()
                words = line.split()
                data += words
        total_len = len(data)
        print("Total Length of Corpus : %d" % total_len)
        num_batches = total_len // batch_size
        data = data[: num_batches * batch_size]
        return data, num_batches

    def get_max_len(self):
        return self.vocab.max_len


def get_corpus(trn_path, val_path, tst_path, batch_size):
    fn = os.path.join('data', 'cache.pt')
    if os.path.exists(fn):
        print('Loading catched dataset...')
        corpus = torch.load(fn)
    else:
        print('Producing dataset...')
        corpus = Corpus(trn_path, val_path, tst_path, batch_size)
        # torch.save(corpus, fn)

    return corpus
