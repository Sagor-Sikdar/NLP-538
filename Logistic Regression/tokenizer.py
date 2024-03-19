import random, os, sys
import numpy as np
import math
import csv
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import re
import heapq
import collections
import string
import matplotlib.pyplot as plt
import torch.optim as optim

np.set_printoptions(threshold = np.inf)
# torch.set_printoptions(10000)


def remove_non_ascii(text):
    return re.sub(r'[^\x00-\x7F]+', '?', text)


def wordTokenizer(text):
    #input: text, a single string to be word tokenized.
    #output: words, a list of strings of all word tokens, in order, from the string

    # print(text)
    tokens = re.findall(r'[^\n\t\f\v\r ]+', text)

    # defining the regular expressions
    single_quote_pattern = r"(\w+)(n't)|(\w+)(\'\w+)"
    emoticon_pattern = r'(:\)|:-\)|:\(|:-\()'
    numeric_dot_pattern = r'([$#@])*(\d+)(\.{1})(\d+)'
    dash_pattern = r'([$#@])*([^.,/>?<);:\'"!@#$%^&*()-]+)(-)+([^.,/>?<);:\'"!@#$%^&*()-]+)'
    symbol_concat_pattern = r'([$#@])*([^.,/>?<);:\'"!@#$%^&*()-]+)'
    punctuation_pattern = r'(|\.{1,}|,{1,}|/{1,}|>{1,}|<{1,}|\?{1,}|\){1,}|\({1,}|;{1,}|:{1,}|\'{1,}|"{1,}|!{1,}|@{1,}|#{1,}|\${1,}|%{1,}|\^{1,}|&{1,}|\*{1,}|-{1,}|(\x00-\x7F){1,})'
    punctuation_free_pattern = r'[^.,/>?<);:\'"!@#$%^&*()-]+'

    # combining all the expressions for each token
    combined_pattern = '|'.join([single_quote_pattern, emoticon_pattern, numeric_dot_pattern, dash_pattern, symbol_concat_pattern, punctuation_pattern, punctuation_free_pattern])

    words = []
    for token in tokens:
      matches = re.finditer(combined_pattern, token)
      for idx in matches:
          if idx.group(0) == "":
            continue
          if idx.group(1) and idx.group(2):
            # print(f"{idx.group(1) }\n{idx.group(2)}")
            # these comparison is for Xn't to make it "X" and "n't"
            words.extend([idx.group(1), idx.group(2)])
          elif idx.group(3) and idx.group(4):
            # print(f"{idx.group(3) }\n{idx.group(4)}")
            # these comparison is for X's to make it "X" and "'s"
            words.extend([idx.group(3), idx.group(4)])
          else:
            # print(f"{idx.group(0)}")
            # it captures all other cases
            words.append(idx.group(0))

    return words


def get_initial_vocab(text):
    corpus = collections.defaultdict(int)
    for line in text.split():
        corpus[' '.join(line)] += 1
    return corpus

def pairwise_freq_calculation(corpus):
    pairs = collections.defaultdict(int)

    for word in corpus:
        symbols = word.split()
        for itr in range(len(symbols) - 1):
            pairs[symbols[itr], symbols[itr+1]] += corpus[word]
    return pairs

def merge_vocab(pair, corpus):
    new_courpus = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in corpus:
        new_word = p.sub(''.join(pair), word)
        new_courpus[new_word] = corpus[word]
    return new_courpus


def spacelessBPELearn(docs, max_vocabulary=1000):
    #input: docs, a list of strings to be used as the corpus for learning the BPE vocabulary
    #output: final_vocabulary, a set of all members of the learned vocabulary
    #  return final_vocabulary

    updated_corpus = get_initial_vocab(docs)

    vocab = collections.Counter()
    for word in updated_corpus:
        # vocab.update(word.split())
        vocab += collections.Counter(word)
    del vocab[' ']

    iteration = 0
    remaining = max_vocabulary - len(list(vocab.keys()))
    while(remaining > 0):
        pairs = pairwise_freq_calculation(updated_corpus)
        if not pairs:
            break

        best_pair = max(pairs, key=pairs.get)
        vocab[best_pair[0] + best_pair[1]] = pairs[best_pair]
        updated_corpus = merge_vocab(best_pair, updated_corpus)

        if iteration in [0, 1, 10, 100, 500]:
            sorted_pair = sorted(pairs.items(), key=lambda x: x[1], reverse=True)
            print("Iteration {} --> {}".format(iteration, sorted_pair[:5]))
        iteration += 1
        remaining -= 1

    return vocab

def spacelessBPETokenize(text, vocab):

    #input: text, a single string to be word tokenized.

    #       vocab, a set of valid vocabulary words

    #output: words, a list of strings of all word tokens, in order, from the string
    # Initialize an empty list to store the words
    tokens = []

    for txt in text.split():
        start_index = 0
        end_index = len(txt)

        while(start_index < end_index):
            if txt[start_index:end_index] in vocab:
                tokens.append(txt[start_index:end_index])
                start_index = end_index
                end_index = len(txt)

            else:
                end_index -= 1

        if start_index != len(txt):
            tokens.append("unk_word")

    return tokens


def most_freq_vocabulary(text, tokenizer_type, vocab_size = 500):
    vocabs = None
    if tokenizer_type == 'word':
        tokens = wordTokenizer(text)
        vocabs = collections.Counter(tokens)
    else:
        vocabs = spacelessBPELearn(text, 1000)
    most_common = vocabs.most_common(vocab_size)
    return most_common

if __name__ == "__main__":
    data = None
    with open('a1_tweets.txt', 'r') as file:
        data = file.read()
        data = remove_non_ascii(data)

    print("checkpoint 1.1")
    Docs = data.split('\n')
    # # printing for wordTokenizer for the first 5 and the last documents
    for index, doc in enumerate(Docs[:5]):
        words = wordTokenizer(doc)
        print(f"Document {index + 1}: {doc}\nwordTokenized : {words}\n\n")

    # # last one
    words = wordTokenizer(Docs[-2])
    print(f"Last Document: {Docs[-2]}\nwordTokenized : {words}\n\n")

    print("checkpoint 1.2")
    vocabs = spacelessBPELearn(data, 1000)
    print("\n\nVocabulary: ")
    print(list(vocabs.keys()))


    for i in range(5):
        print(data.split('\n')[i])
        tokens = spacelessBPETokenize(data.split('\n')[i], vocabs)
        print("\n\nDoc: {} : {}".format(i + 1, data.split('\n')[i]))
        print(f"bpeTokenized: {tokens}")
