from gensim import downloader
import numpy as np
import torch
from torch import nn
from pos_embedding import transform

def load_data(path_to_data):
    sentences = []
    sentence = []
    pos = []

    # read data
    with open(path_to_data, 'r', encoding='utf-8') as f:
        data = f.readlines()

    # split data
    for line in data:
        if line != '\n':
            line_data = line.split('\t')
            # take only indexes 0,1,3,6
            word = [line_data[0], line_data[1], line_data[3], line_data[6]]
            pos.append(word[2])
            sentence.append(word)
        else:
            sentences.append(sentence)
            sentence = []
    return sentences, list(set(pos))


def tokenize(sentences, model, length, pos_list):
    pos_to_vec = transform(pos_list)
    set_data = []

    for sentence in sentences:
        for word in sentence:
            if word[1] not in model.key_to_index:
                word_vec = torch.zeros(length)
            else:
                word_vec = torch.Tensor(model[word[1]].tolist())

            pos_vec = pos_to_vec[word[2]]
            final_vec = torch.cat((word_vec, pos_vec))
            set_data.append(final_vec)

    return set_data


class AutoencoderNN(torch.nn.Module):
    def __init__(self):
        super(AutoencoderNN).__init__()
        self.loss = nn.MSELoss()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(45, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 4)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(4, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 45)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, self.loss(decoded, x)


def main():
    train_sentences, pos_train = load_data('train.labeled')
    test_sentences, pos_test = load_data('test.labeled')
    tokenize_model = downloader.load('glove-twitter-100')
    tokenized_train = tokenize(train_sentences, tokenize_model, 100, pos_train)
    print(tokenized_train[0].shape)
    exit()
    tokenized_test = tokenize(test_sentences, tokenize_model, 100, pos_test)

    # print(train_sentences[0])


if __name__ == '__main__':
    main()
