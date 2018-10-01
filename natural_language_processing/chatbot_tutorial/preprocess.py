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


USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

# データの中身をみる
corpus_name = "cornell movie-dialogs corpus"
corpus = os.path.join("data", corpus_name)

def printLines(file, n=10):
    with open(file, 'rb') as datafile:
        lines = datafile.readlines()
    for line in lines[:n]:
        print(line)

printLines(os.path.join(corpus, "movie_lines.txt"))

"""
データを適切な形にする
- loadLines
- loadConversations
- extractSentencePairs
"""
def loadLines(fileName, fields):
  lines = {}
  with open(fileName, "r", encoding="iso-8859-1") as f:
    for line in f:
      values = line.split(" +++$+++ ")
      lineObj = {}
      for i, field in enumerate(fields):
        lineObj[field] = values[i]
      lines[lineObj['lineID']] = lineObj
  return lines

def loadConversations(fileName, lines, fields):
  conversations = []
  with open(fileName, "r", encoding="iso-8859-1") as f:
    for line in f:
      values = line.split(" +++$+++ ")

      # Extract Field
      convObj = {}
      for i, field in enumerate(fields):
        convObj[field] = values[i]

      lineIds = eval(convObj["utteranceIDs"])
      convObj["lines"] = []

      for lineId in lineIds:
        convObj["lines"].append(lines[lineId])
      conversations.append(convObj)
  return conversations

def extractSentencePairs(conversations):
  qa_pairs = []
  for conversation in conversations:
    for i in range(len(conversation["lines"]) - 1):
      inputLine = conversation["lines"][i]["text"].strip()
      targetLine = conversation["lines"][i+1]["text"].strip()

      if inputLine and targetLine:
        qa_pairs.append([inputLine, targetLine])
  return qa_pairs


# 前処理 & 新しいファイル生成
datafile = os.path.join(corpus, "formatted_movie_lines.txt")
delimiter = '\t'
# Unescape the delimiter
delimiter = str(codecs.decode(delimiter, "unicode_escape"))

# Initialize lines dict, conversations list, and field ids
lines = {}
conversations = []
MOVIE_LINES_FIELDS = ["lineID", "characterID", "movieID", "character", "text"]
MOVIE_CONVERSATIONS_FIELDS = ["character1ID", "character2ID", "movieID", "utteranceIDs"]

# Load lines and process conversations
print("\nProcessing corpus...")
lines = loadLines(os.path.join(corpus, "movie_lines.txt"), MOVIE_LINES_FIELDS)
print("\nLoading conversations...")
conversations = loadConversations(os.path.join(corpus, "movie_conversations.txt"),
                                  lines, MOVIE_CONVERSATIONS_FIELDS)


# Write new csv file
"""
input(i) => target(i+1)
...の対訳のようにペアができる
"""
print("\nWriting newly formatted file...")
with open(datafile, 'w', encoding='utf-8') as outputfile:
    writer = csv.writer(outputfile, delimiter=delimiter)
    for pair in extractSentencePairs(conversations):
        writer.writerow(pair)

# Print a sample of lines
print("\nSample lines from file:")
printLines(datafile)