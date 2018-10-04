# Prepare data for model
"""
やるべきことは,
1. 文章中の単語列 => 対応する語彙中のインデックスに変換
2. GPUで学習できるように, ミニバッチ処理にすること
  a. バッチ中の文章の長さに気をつける。同じバッチに異なる大きさの文章を入れるために, max_lengthでpaddingする
3. (batchsize, max_length) => バッチサイズ分の文章
   (max_length, batchsize) => 各時刻単位のstepでのバッチサイズ分
"""
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


# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token
MAX_LENGTH = 10 # Maximum sentence length to consider


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
    self.word2count = {}
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

def indexesFromSentence(voc, sentence):
  return [voc.word2index[word] for word in sentence.split(" ")] + [EOS_token]

def zeroPadding(l, fillvalue=PAD_token):
  return list(itertools.zip_longest(*l, fillvalue=fillvalue))

def binaryMatrix(l, value=PAD_token):
  m = []
  for i, seq in enumerate(l):
    m.append([])
    for token in seq:
      if token == PAD_token:
        m[i].append(0)
      else:
        m[i].append(1)
  return m

# Returns padded input sequence and lengths
def inputVar(l, voc):
  indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
  lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
  padList = zeroPadding(indexes_batch)
  padVar = torch.LongTensor(padList)
  return padVar, lengths

# Returns padded target sequence tensor, padding mask, and max target length
def outputVar(l, voc):
  indexes_batch = [indexesFromSentence(voc, sequence) for sequence in l]
  max_target_len = max([len(indexes) for indexes in indexes_batch])
  padList = zeroPadding(indexes_batch)
  mask = binaryMatrix(padList)
  mask = torch.ByteTensor(mask)
  padVar = torch.LongTensor(padList)
  return padVar, mask, max_target_len

# Returns all items for a given batch of pairs
def batch2TrainData(voc, pair_batch):
  pair_batch.sort(key=lambda x : len(x[0].split(" ")), reverse=True) # 長い順番にならべかえる
  input_batch, output_batch = [], []
  for pair in pair_batch:
    input_batch.append(pair[0])
    output_batch.append(pair[1])

  inp, lengths = inputVar(input_batch, voc)
  output, mask, max_target_len = outputVar(output_batch, voc)
  return inp, lengths, output, mask, max_target_len


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

MIN_COUNT = 3
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
      keep_pairs.append(pair)
  
  print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
  return keep_pairs

# Trim voc and pairs
pairs = trimRareWords(voc, pairs, MIN_COUNT)

# Example for validation
small_batch_size = 5
random_pairs = [random.choice(pairs) for _ in range(small_batch_size)]
batches = batch2TrainData(voc, random_pairs)
input_variable, lengths, target_variable, mask, max_target_len = batches

print("input_variable:", input_variable)
print("lengths:", lengths)
print("target_variable:", target_variable)
print("mask:", mask)
print("max_target_len:", max_target_len)


# Define the model
"""
Seq2Seq Model
- seq2seq modelは可変長の入力を受け取り、可変長のシーケンスを出力する。
- RNN : Encoder => 入力文を固定長のベクトルへ変換(RNNの最終層に意味を入力のベクトルの潜在的な情報が含まれていると考えられる)
        Decoder => contextベクトルを出力する

Encoder :
  Encoderは入力xを毎時受け取り、出力ベクトルと隠れ層のベクトルを出力する
  Encoderは各ステップのコンテキストを高次元の点として表現し、デコーダはそれを使って意味のある出力をすることができるようになる。
  EncoderとしてはGRUを利用する : bidirectional GRU

  Note :
    1. Embeddingレイヤ => word indices => 人気の特徴空間へ
    2. RNNモジュールにpadded batch sequenceを流すときは、 torch.nn.utils.rnn.pack_padded_sequence と torch.nn.utils.rnn.pad_packed_sequence を使う
  
  Encoder側の計算グラフ
    1. word => Embeddingへ
    2. padded batch sequenceをpackする <- ?
    3. GRUに流し込む
    4. Unpack padding <- ?
    5. GRUの双方向の合計をとって出力
    6. 出力と最終層の隠れベクトルを出す
"""
class EncoderRNN(nn.Module):
  def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
    super(EncoderRNN, self).__init__()
    self.n_layers = n_layers
    self.hidden_size = hidden_size
    self.embedding = embedding
    self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                      dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
  
  def forward(self, input_seq, input_lengths, hidden=None):
    # Convert word indexes to embedding
    embedded = self.embedding(input_seq)
    # Pack padded batch of sequence for RNN module
    """
    このpack_padded_sequenceに何の意味があるのかよくわかってない
    """
    packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
    # Forward pass through GRU
    outputs, hidden = self.gru(packed, hidden)
    outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
    # Sum bidirectional GRU outputs
    """
    TODO : GRUの出力の形をチェックする
    """
    outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
    return outputs, hidden

"""
Decoder
  Decoderはtokenごとに返答を生成する。Encoderのコンテキストベクトルを使い、隠れ層を用いて次の単語を出力する => <EOS>が出るまで出力を続ける
  seq2seq Decoderの共通の問題はEnocderが入力の文章を全て単に一つのベクトルにエンコードしてしまった場合は, 情報の損失が発生してしまう : 長文のシーケンスを見るときほどこの問題が発生する

  この問題に対処するため Attention Mechanism が生まれた。 => Decoderが出力するときに全部の文章を見ない方法
  LuongらはBahdanauらの方法を, Global Attentionを作って改善した。キーとなる違いは, Encoderの全ての隠れ層を生成に利用するかどうか
  2つめは, attention重みやエナジーを, Decoderの今のステップの隠れ層を使って計算するかどうか .. score(h_t, h_s)で計算 : そのステップでのDecoderの隠れ層, h_sはすべてのEncoderの隠れ層
"""
class Attn(torch.nn.Module):
  def __init__(self, method, hidden_size):
    super(Attn, self).__init__()
    self.method = method
    if self.method not in ['dot', 'general', 'concat']:
      raise ValueError(self.method, "is not an appropriate attention method")
    self.hidden_size = hidden_size
    if self.method == "general":
      self.attn = torch.nn.Linear(self.hidden_size, hidden_size)
    elif self.method == "concat":
      self.attn = torch.nn.Linear(self.hidden_size * 2, hidden_size)
      self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))
  
  def dot_score(self, hidden, encoder_output):
    return torch.sum(hidden * encoder_output, dim=2)
  
  def general_score(self, hidden, encoder_output):
    energy = self.attn(encoder_output)
    return torch.sum(hidden * energy, dim=2)
  
  def concat_score(self, hidden, encoder_output):
    energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
    return torch.sum(self.v * energy, dim=2)
  
  def forward(self, hidden, encoder_outputs):
    # Calculate the attention weights
    if self.method == "general":
      attn_energies = self.general_score(hidden, encoder_outputs)
    elif self.method == "concat":
      attn_energies = self.concat_score(hidden, encoder_outputs)
    elif self.method == "dot":
      attn_energies = self.dot_score(hidden, encoder_outputs)
    
    # Transpose max_length and batch_size dimensions
    attn_energies = attn_energies.t()

    # Return the softmax normalized probability score
    return F.softmax(attn_energies, dim=1).unsqueeze(1)


"""
Decoderの計算グラフ
1. 現在の入力のEmbeddingを受け取る
2. GRUを計算する
3. 2からGRUの出力と、アテンションを計算する
4. 新しいコンテキストベクトルを得るために、attention重みとエンコーダーの出力を掛け合わせる
5. GRUの出力とweighted contextをconcat
6. 次の単語を予測する
7. 出力と隠れ層
"""
class LuongAttnDecoderRNN(nn.Module):
  def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
    super(LuongAttnDecoderRNN, self).__init__()
    # Keep for reference
    self.attn_model = attn_model
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.n_layers = n_layers
    self.dropout = dropout
  
    # Define layers
    self.embedding = embedding
    self.embedding_dropout = nn.Dropout(dropout)
    self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
    self.concat = nn.Linear(hidden_size*2, hidden_size)
    self.out = nn.Linear(hidden_size, output_size)
    self.attn = Attn(attn_model, hidden_size)

  def forward(self, input_step, last_hidden, encoder_outputs):
    # Note: we run this one step (word) at a time
    # Get embedding of current input word
    embedded = self.embedding(input_step)
    embedded = self.embedding_dropout(embedded)

    # Forward through unidirectional GRU
    rnn_output, hidden = self.gru(embedded, last_hidden)
    # Calculate attention weights from the current GRU output
    attn_weights = self.attn(rnn_output, encoder_outputs)
    # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
    context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
      
    # Concatenate weighted context vector and GRU output using Luong eq. 5
    rnn_output = rnn_output.squeeze(0)
    context = context.squeeze(1)
    concat_input = torch.cat((rnn_output, context), 1)
    concat_output = torch.tanh(self.concat(concat_input))
    # Predict next word using Luong eq. 6
    output = self.out(concat_output)
    output = F.softmax(output, dim=1)
    # Return output and final hidden state
    return output, hidden    

def maskNLLoss(inp, target, mask):
  nTotal = mask.sum()
  # TODO : crossEntropyの計算は詳しく手で追う
  crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)))
  loss = crossEntropy.masked_select(mask).mean()
  loss = loss.to(device)
  return loss, nTotal.item()


"""
Define Training Procedure
1. Masked Loss  
 - Paddingした部分のLossは計算しないようにする

2. Single Training Iteration
 - teacher forcingの利用 : decoder側でguessを出すが、そのguessをそのまま次の入力へ渡すのではなく、
                          教師データの系列を渡してLossを計算する
                          しかし、その方法ではDecoder側が不安定になるため、教師データ系列を渡す割合を設定する

  - gradient clipping : 勾配爆発を防ぐために、勾配が一定以上の大きさになったらclippingを行う

3. Sequence Operation
  1. Encoder側に入力バッチ分を渡す
  2. decoder側の入力を<SOS>とし、Encoderの最終層の隠れ層を初期状態として渡す
  3. Encoder側でForward計算
  4. teacher_forcing : 今のtargetの単語をDecoderに渡す, teacher_forcing無しの時はDecoderの出力を次へ渡す
  5. Loss計算
  6. 誤差逆伝播
  7. 勾配クリッピング
  8. パラメータのアップデート
"""
# 各ステップごとに呼び出され, バッチ分のデータを処理する
def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, embedding,
          encoder_optimizer, decoder_optimizer, batch_size, clip, max_length=MAX_LENGTH):
  
  # Zero gradients
  encoder_optimizer.zero_grad()
  decoder_optimizer.zero_grad()

  # Set device options
  input_variable = input_variable.to(device)
  lengths = lengths.to(device)
  target_variable = target_variable.to(device)
  mask = mask.to(device)

  # Initialize variables
  loss = 0
  print_losses = []
  n_totals = 0

  # Forward pass through encoder
  encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

  # Create initial decoder input
  decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
  decoder_input = decoder_input.to(device)

  # Set initial decoder hidden state to the encoder's final hidden state
  decoder_hidden = encoder_hidden[:decoder.n_layers]

  # Determine if we are using teacher forcing this iteration
  use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

  # Forward batch of sequences through decoder one time step at time
  if use_teacher_forcing:
    for t in range(max_target_len):
      decoder_output, decoder_hidden = decoder(
        decoder_input, decoder_hidden, encoder_outputs
      )
      # Teacher Forcing : Next input is current target
      decoder_input = target_variable[t].view(1, -1)
      # Calculate and accumulate loss
      mask_loss, nTotal = maskNLLoss(decoder_output, target_variable[t], mask[t])
      loss += mask_loss
      print_losses.append(mask_loss.item() * nTotal)
      n_totals += nTotal
  else:
    for t in range(max_target_len):
      decoder_output, decoder_hidden = decoder(
        decoder_input, decoder_hidden, encoder_outputs
      )
      # No Teacher Forcing
      # 一番大きいところのインデックスを取得する
      _, topi = decoder_output.topk(1)
      decoder_input = torch.LongTensor([topi[i][0] for i in range(batch_size)])
      decoder_input = decoder_input.to(device)

      # Calculate the Loss
      mask_Loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
      loss += mask_Loss
      print_losses.append(mask_loss.item() * nTotal)
      n_totals += nTotal
  
  # Perform backpropatation
  loss.backward()

  # Clip model weights
  _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
  _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

  # Adjust model weights
  encoder_optimizer.step()
  decoder_optimizer.step()

  return sum(print_losses) / n_totals

"""
Training Iterations
"""
def trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
              embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size,
              print_every, save_every, clip, corpus_name, loadFilename):
  # Load batches for each iteration
  # => これランダムにとっちゃうと全データ学習に使えないような気がするが...
  training_batches = [batch2TrainData(voc, [random.choice(pairs) for _ in range(batch_size)])
                    for _ in range(n_iteration)]
  
  # Initialize
  print("Initializing...")
  start_iteration = 1
  print_loss = 0
  if loadFilename:
    start_iteration = checkpoint["iteration"] + 1
  
  # Training loop
  print("Training...")
  for iteration in range(start_iteration, n_iteration + 1):
    training_batch = training_batches[iteration - 1]
    # Extract fields, from batch
    input_variable, lengths, target_variable, mask, max_target_len = training_batch

    # Run a training iteration with batch
    loss = train(input_variable, lengths, target_variable, mask, max_target_len, encoder,
                decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size, clip)
    print_loss += loss

    # Print Progress
    if iteration % print_every == 0:
      print_loss_avg = print_loss / print_every
      print("Iteration : {}; Percent complete : {:.1f}%; Average loss : {:.4f}".format(
        iteration, iteration / n_iteration * 100, print_loss_avg
      ))
      print_loss = 0
    
    # Save checkpoints
    if (iteration % save_every == 0):
      directory = os.path.join(save_dir, model_name, corpus_name, "{}-{}_{}".format(
        encoder_n_layers, decoder_n_layers, hidden_size))
      if not os.path.exists(directory):
        os.makedirs(directory)
      torch.save({
        'iteration': iteration,
        'en': encoder.state_dict(),
        'de': decoder.state_dict(),
        'en_opt': encoder_optimizer.state_dict(),
        'de_opt': decoder_optimizer.state_dict(),
        'loss': loss,
        'voc_dict': voc.__dict__,
        'embedding': embedding.state_dict()
      }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))

"""
Define Evaluation
- Encoderを入力として与えたときに, Decoderがどういう振る舞いをして出力するかを決める必要がある
- Greedy Decoding: decoderの出力の中で、もっともsoftmaxが高くなるものを選択する

計算グラフ
  1. Encoderへ入力文を渡す
  2. Encoderの最終層の隠れ層をDecoderへ渡す
  3. Decoderの最初の入力を<SOS>で渡す
  4. 毎時1単語ずつ渡し, 
    4.a Decoderを計算
    4.b softmax計算, tokenとscoreを保持
    4.c 4.bで保持したtokenを元に次のdecoderの出力を計算する
"""
class GreedySearchDecoder(nn.Module):
  def __init__(self, encoder, decoder):
    super(GreedySearchDecoder, self).__init__()
    self.encoder = encoder
    self.decoder = decoder

  def forward(self, input_seq, input_length, max_length):
    # Forward Encoder(1)
    encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)

    # set encoder's final hidden layer to be first hidden input to the decoder
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    # Initialize decoder input with SOS token
    decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token
    # Initialize tensors to append decoded words to
    all_tokens = torch.zeros([0], device=device, dtype=torch.long)
    all_scores = torch.zeros([0], device=device)
    # Iteratively decode one word token at a time
    for _ in range(max_length):
      # Forward pass through decoder
      decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
      # Obtain most likely word token and its softmax score
      decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
      # Record token and score
      all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
      all_scores = torch.cat((all_scores, decoder_scores), dim=0)
      # Prepare current token to be next decoder input (add a dimension)
      """
      torch.unsqueeze
      特定の位置に1サイズ分次元を追加した新しいTensorを返す
      """
      decoder_input = torch.unsqueeze(decoder_input, 0)
      # Return collections of word tokens and scores
    return all_tokens, all_scores


"""
Run Model
  1. Run Training
"""
# Configure models
model_name = 'cb_model'
attn_model = 'dot'
#attn_model = 'general'
#attn_model = 'concat'
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 64

# Set checkpoint to load from; set to None if starting from scratch
loadFilename = None
checkpoint_iter = 4000
#loadFilename = os.path.join(save_dir, model_name, corpus_name,
#                            '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
#                            '{}_checkpoint.tar'.format(checkpoint_iter))


# Load model if a loadFilename is provided
if loadFilename:
    # If loading on same machine the model was trained on
    checkpoint = torch.load(loadFilename)
    # If loading a model trained on GPU to CPU
    #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    voc.__dict__ = checkpoint['voc_dict']


print('Building encoder and decoder ...')
# Initialize word embeddings
embedding = nn.Embedding(voc.num_words, hidden_size)
if loadFilename:
    embedding.load_state_dict(embedding_sd)
# Initialize encoder & decoder models
encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
if loadFilename:
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
# Use appropriate device
encoder = encoder.to(device)
decoder = decoder.to(device)
print('Models built and ready to go!')


# Configure training/optimization
clip = 50.0
teacher_forcing_ratio = 1.0
learning_rate = 0.0001
decoder_learning_ratio = 5.0
n_iteration = 4000
print_every = 1
save_every = 500

# Ensure dropout layers are in train mode
encoder.train()
decoder.train()

# Initialize optimizers
print('Building optimizers ...')
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
if loadFilename:
    encoder_optimizer.load_state_dict(encoder_optimizer_sd)
    decoder_optimizer.load_state_dict(decoder_optimizer_sd)

# Run training iterations
print("Starting Training!")
trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
           embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size,
           print_every, save_every, clip, corpus_name, loadFilename)

"""
Evaluate text
"""
def evaluate(encoder, decoder, searcher, voc, sentence, max_length=MAX_LENGTH):
    ### Format input sentence as a batch
    # words -> indexes
    indexes_batch = [indexesFromSentence(voc, sentence)]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, max_length)
    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words


def evaluateInput(encoder, decoder, searcher, voc):
    input_sentence = ''
    while(1):
        try:
            # Get input sentence
            input_sentence = input('> ')
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit': break
            # Normalize sentence
            input_sentence = normalizeString(input_sentence)
            # Evaluate sentence
            output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
            # Format and print response sentence
            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            print('Bot:', ' '.join(output_words))

        except KeyError:
            print("Error: Encountered unknown word.")


# チャットをはじめる
# Set dropout layers to eval mode
encoder.eval()
decoder.eval()

# Initialize search module
searcher = GreedySearchDecoder(encoder, decoder)
# Begin chatting (uncomment and run the following line to begin)
evaluateInput(encoder, decoder, searcher, voc)