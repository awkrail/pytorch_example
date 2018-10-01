from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math

# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token

"""
Voc クラスの定義
単語をベクトルとして表現するために、データセットから引き出した単語の順番に, 単語をone-hotのベクトルとして表現する
Vocクラスは単語をインデックスに、インデックスを単語に変換する役割を持つ。

`addWord` : ボキャブラリを追加する
`addSentence` : 文に含まれる一文をボキャブラリとして追加する
`trim` : トリミングを行う
"""

class Voc:
  def __init__(self, name):
    self.name = name
    self.trimmed = False
    self.word2index = {}
    self.index2count = {}
    self.index2word = {PAD_token : "PAD", SOS_token : "SOS", EOS_token : "EOS"}
    self.num_words = 3
  
  def addSentence(self, sentence):
    for word in sentence.split(" "):
      self.addWord(word)
  
  def addWord(self, word):
    if word not in self.word2index:
      self.word2index[word] = self.num_words
      self.word2count[word] = 1
      self.index2word[self.num_words] = word
      self.num_words += 1
    else:
      self.word2count[word] += 1
  
  # Remove words below a certain count threshold
  def trim(self, min_count):
    if self.trimmed:
      return
    self.trimmed = False

    keep_words = []

    for k, v in self.word2count.items():
      if v >= min_count:
        keep_words.append(k)
    
    print("keep words {} / {} = {:4f}".format(
      len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
    ))

    # Reinitialize dictionary
    self.word2index = {}
    self.word2count = {}
    self.index2word = { PAD_token : "PAD", SOS_token : "SOS", EOS_token : "EOS" }

    for word in keep_words:
      self.addWord(word)

MAX_LENGTH = 10 # Maximum sentence length to consider

"""
前処理として
- Unicode => ASCIIへ変更
- 全ての単語をlowercaseへ
- MAX_LENGTHを超える文章はフィルターする
"""

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

def readVocs(datafile, corpus_name):
    print("Reading lines...")
    # Read the file and split into lines
    lines = open(datafile, encoding='utf-8').\
        read().strip().split('\n')
    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    voc = Voc(corpus_name)
    return voc, pairs

# Returns True iff both sentences in a pair 'p' are under the MAX_LENGTH threshold
def filterPair(p):
    # Input sequences need to preserve the last word for EOS token
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH

# Filter pairs using filterPair condition
def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

# Using the functions defined above, return a populated voc object and pairs list
def loadPrepareData(corpus, corpus_name, datafile, save_dir):
    print("Start preparing training data ...")
    voc, pairs = readVocs(datafile, corpus_name)
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filterPairs(pairs)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    print("Counted words:", voc.num_words)
    return voc, pairs


# Load/Assemble voc and pairs
corpus_name = "cornell movie-dialogs corpus"
corpus = os.path.join("data", corpus_name)
datafile = os.path.join(corpus, "formatted_movie_lines.txt")

save_dir = os.path.join("data", "save")
voc, pairs = loadPrepareData(corpus, corpus_name, datafile, save_dir)
# Print some pairs to validate
print("\npairs:")
for pair in pairs[:10]:
    print(pair)


# 学習中に早く収束させるために、めったに使われない単語を削っていく
# Voc.trim()を利用する
def trimRareWords(voc, pairs, MIN_COUNT):
  voc.trim(MIN_COUNT) 
  keep_pairs = []

  for pair in pairs:
    input_sentence = pair[0]
    output_sentence = pair[1]
    keep_input = True
    keep_output = True

    for word in input_sentence.split(" "):
      if word not in voc.word2index:
        keep_output = False
        break
    
    for word in output_sentence.split(" "):
      if word not in voc.word2index:
        keep_output = False
        break
    
    if keep_input and keep_output:
      keep_pairs.append(pairs)
  
  print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
  return keep_pairs

# Trim voc and pairs
pairs = trimRareWords(voc, pairs, MIN_COUNT)