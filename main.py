from gensim import downloader
import numpy as np
import torch
from torch import nn

# Pre-processing and tokenization


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


def pos_to_oneHot(pos_list):
    tensor_dim = len(pos_list)
    one_hot_tensors = []
    for pos in pos_list:
        one_hot_tensor = torch.zeros(tensor_dim)
        one_hot_tensor[pos_list.index(pos)] = 1
        one_hot_tensors.append(one_hot_tensor)

    return torch.stack(one_hot_tensors)


def tokenize(sentences, model, length, pos_list):
    pos_to_vec = pos_to_oneHot(pos_list)
    print(pos_to_vec)
    set_data = []

    for sentence in sentences:
        for word in sentence:
            if word[1] not in model.key_to_index:
                word_vec = torch.zeros(length)
            else:
                word_vec = torch.Tensor(model[word[1]].tolist())
            print(word[2])
            pos_vec = pos_to_vec[word[2]]
            print(pos_vec)
            exit()
            final_vec = torch.cat((word_vec, pos_vec))
            set_data.append(final_vec)

    return set_data


def main():
    train_sentences, pos_train = load_data('train.labeled')
    test_sentences, pos_test = load_data('test.labeled')
    tokenize_model = downloader.load('glove-twitter-100')
    tokenized_train = tokenize(train_sentences, tokenize_model, 100, pos_train)
    tokenized_test = tokenize(test_sentences, tokenize_model, 100, pos_test)

    # print(train_sentences[0])


if __name__ == '__main__':
    main()
