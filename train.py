import argparse
from dataset import get_corpus
from model import LM
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

parse = argparse.ArgumentParser()
parse.add_argument('--train', default='data/train.txt')
parse.add_argument('--valid', default='data/valid.txt')
parse.add_argument('--test', default='data/test.txt')
parse.add_argument('--batch_size', default=20)
parse.add_argument('--embed_dim', default=15)
parse.add_argument('--kernels', default=[1, 2, 3, 4, 5, 6])
parse.add_argument('--channels', default=25)
parse.add_argument('--seq_len', default=30)
parse.add_argument('--hidden_size', default=300)
parse.add_argument('--learning_rate', default=1)
parse.add_argument('--epochs', default=25)
parse.add_argument('--clip', default=5)

args = parse.parse_args()


def validate(seq_len, valid, valid_idx, model, h):
    val_loss = 0
    step = 0

    n = corpus.len_word_val // (args.seq_len * args.batch_size)
    for j in range(n):
        # val_input = valid[:, j * args.seq_len: j * args.seq_len + args.seq_len, :].cuda()
        # val_true = valid_idx[:, (j * args.seq_len + 1):j * args.seq_len + args.seq_len + 1].cuda()
        val_input = valid[:, j * args.seq_len: j * args.seq_len + args.seq_len, :]
        val_true = valid_idx[:, (j * args.seq_len + 1):j * args.seq_len + args.seq_len + 1]

        y, _ = model(val_input, h)
        loss = torch.nn.functional.cross_entropy(y, val_true.view(-1).long())
        val_loss += loss.item()
        step += 1

        model.zero_grad()

    print('Validation Loss: %.3f, Perplexity: %5.2f' % (val_loss / step, np.exp(val_loss / step)))

    return val_loss / step


def train(corpus):
    cur_best = 10000

    model = LM(args.wtoken, args.ctoken, args.max_len, args.embed_dim, args.channels, args.kernels, args.hidden_size)

    if torch.cuda.is_available():
        model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.epochs):

        # h0 = torch.zeros(2, args.batch_size, args.hidden_size).cuda()
        # c0 = torch.zeros(2, args.batch_size, args.hidden_size).cuda()
        # hidden = (h0, c0)

        model.train(True)

        # hidden_state = [torch.zeros(2, args.batch_size, args.hidden_size).cuda()] * 2  ########
        hidden_state = [torch.zeros(2, args.batch_size, args.hidden_size)] * 2 

        n = corpus.len_word // (args.seq_len * args.batch_size)
        # print('n:', n)
        count = 0
        for i in range(n):
            # for i in range(0, (n-1)*corpus.len_word, args.seq_len):
            model.zero_grad()
            count += 1
            # print(i, count)
            # inputs = corpus.trn_char[:, i * args.seq_len: i * args.seq_len + args.seq_len, :].cuda()
            inputs = corpus.trn_char[:, i * args.seq_len: i * args.seq_len + args.seq_len, :]
            # print('input:', inputs.size())
            # targets = corpus.trn_word[:, (i * args.seq_len + 1):i * args.seq_len + args.seq_len + 1].cuda()
            targets = corpus.trn_word[:, (i * args.seq_len + 1):i * args.seq_len + args.seq_len + 1]
            # print('target:', targets.size())

            temp = []

            for state in hidden_state:
                temp.append(state.detach())

            hidden_state = temp

            out, hidden_state = model(inputs, hidden_state)
            # hidden[0].detach()
            # hidden[1].detach()

            loss = torch.nn.functional.cross_entropy(out, targets.view(-1).long())

            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            optimizer.step()

            # step = (i + 1) // args.seq_len
            if i % 100 == 0:
                print('Epoch %d/%d, Batch x Seq_Len %d/%d, Loss: %.3f, Perplexity: %5.2f' % (
                    epoch, args.epochs, i, corpus.trn_num_batches // args.seq_len, loss.item(), np.exp(loss.item())))

        model.eval()
        val_loss = validate(args.seq_len, corpus.trn_char, corpus.trn_word, model, hidden_state)
        val_perplex = np.exp(val_loss)

        if val_perplex < cur_best:
            print("The current best val loss: ", val_loss)
            cur_best = val_perplex
            torch.save(model.state_dict(), 'model.pkl')


if __name__ == '__main__':
    corpus = get_corpus(args.train, args.valid, args.test, args.batch_size)
    args.max_len = corpus.get_max_len()
    ctoken = len(corpus.vocab)
    wtoken = corpus.vocab.get_len()
    args.ctoken = ctoken
    args.wtoken = wtoken
    train(corpus)
