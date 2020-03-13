#!/usr/bin/env python
# coding: utf-8
import hashlib
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import pandas as pd
import re
from tqdm import trange
import random

from keras.preprocessing.sequence import pad_sequences
from collections import defaultdict
from sklearn.preprocessing import minmax_scale
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


# In[25]:


from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, WeightedRandomSampler
from torchvision import transforms, utils

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        #         return self.decode(self.encode(src, src_mask), src_mask,
        #                             tgt, tgt_mask)
        out = self.encode(src, src_mask)
        return out

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class EncoderDecoder_ForClassification(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder_ForClassification, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        #         return self.decode(self.encode(src, src_mask), src_mask,
        #                             tgt, tgt_mask)
        out = self.encode(src, src_mask)
        return out

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.d_model = d_model
        self.proj = nn.Linear(self.d_model, vocab)

    def forward(self, x):
        # print(torch.mean(x, axis=1).shape)
        out = self.proj(x[:, 0, :])
        # out = self.proj(torch.mean(x, axis=1))
        # print(out.shape)
        return out


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return self.norm(x + self.dropout(sublayer(x)))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


class Batch:
    "Object for holding a batch of data with mask during training."

    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg
            self.trg_y = trg
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask

class Batch_binary_classification:

    def __init__(self, src, trg=None, pad=0, labels=None):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)

        self.labels = labels



        if trg is not None:
            self.trg = trg
            self.trg_y = trg
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()


    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


class LogTokenizer:
    def __init__(self, filters='([ |:|\(|\)|=|,])|(core.)|(\.{2,})'):
        self.filters = filters
        self.word2index = {'<PAD>': 0, '<CLS>': 1, '<MASK>': 2}
        self.index2word = {0: '<PAD>', 1: '<CLS>', 2: '<MASK>'}
        self.n_words = 3  # Count SOS and EOS

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1

    def tokenize(self, sent):
        sent = sent.replace('\'', '')
        filtered = re.split(self.filters, sent)
        new_filtered = []
        for f in filtered:
            if f != None and f != '':
                new_filtered.append(f)
        for w in range(len(new_filtered)):
            self.addWord(new_filtered[w])
            new_filtered[w] = self.word2index[new_filtered[w]]
        return new_filtered


class MaskedDataset(Dataset):

    def __init__(self, data, tokenizer, mask_percentage=0.2, transforms=None, pad=0, pad_len=64, targets=None):
        self.c = copy.deepcopy

        self.data = data


        self.padded_data = self._get_padded_data(data, pad_len)

        self.mask_percentage = mask_percentage
        self.transforms = transforms
        self.pad = pad
        self.pad_len = pad_len
        self.tokenizer = tokenizer
        self.targets = targets

        self.token_id = self.tokenizer.tokenize('<MASK>')[0]

    def get_sample_weights(self):
        def changeTokenToCount(token, dictInfo):
            if token == 0:
                return 0
            else:
                return dictInfo[token]

        d = self.c(self.padded_data)
        data_token_idx_df = pd.DataFrame(d)
        storeColumnInfo = defaultdict(dict)
        cnt = 0

        for column in range(self.pad_len):
            val_cnt = pd.value_counts(data_token_idx_df.iloc[:, column])
            storeColumnInfo[column] = val_cnt.to_dict()
            data_token_idx_df.iloc[:, column] = data_token_idx_df.iloc[:, column].apply(
                lambda x: changeTokenToCount(x, storeColumnInfo[column]))
        #         weights = minmax_scale(np.divide(np.ones(data_token_idx_df.shape[0]),
        #                                          data_token_idx_df.sum(axis=1)),
        #                                feature_range=(0.005, 0.995))
        weights = 1 - minmax_scale(data_token_idx_df.sum(axis=1), feature_range=(0.0, 0.75))
        return weights

    @staticmethod
    def subsequent_mask(size, trg):
        "Mask out subsequent positions."
        attn_shape = (size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        t = torch.from_numpy(subsequent_mask) == 0

        return t & trg

    def make_std_mask(self, trg):
        "Create a mask to hide padding and future words."
        trg_mask = (trg != self.pad)
        trg_mask = self.subsequent_mask(trg.shape[0], trg_mask)
        return trg_mask

    def _get_padded_data(self, data, pad_len):
        d = self.c(data)
        pd = pad_sequences(d, maxlen=pad_len, dtype="long",
                           truncating="post", padding="post")
        return pd

    def __getitem__(self, index):
        masked_data = self.c(self.padded_data[index])
        # print(masked_data)
        src = self.padded_data[index]

        offset = 1


        #print(self.targets)

        if isinstance(self.targets, type(None)):
            data_len = len(self.data[index]) - 1 if len(self.data[index]) < self.pad_len else self.pad_len - 1
            return src, offset, data_len, index
        else:
            data_len = len(self.data[index]) - 1 if len(self.data[index]) < self.pad_len else self.pad_len - 1
            tg = self.targets[index]
            # print("EXECUTE HERE")
            return src, offset, data_len, index, tg

    def __len__(self):
        return self.padded_data.shape[0]


class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion, opt=None, is_test=False):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        self.is_test = is_test

    def __call__(self, x, y, norm):
        x = self.generator(x)
        # print(x.size())
        #y = y.reshape(-1)
        # print(x)
        # print("########")
        # print(y)
        loss = self.criterion(x, y)
        if not self.is_test:
            loss.backward()
            if self.opt is not None:
                self.opt.step()
                #                 self.opt.optimizer.zero_grad()
                self.opt.zero_grad()

        return loss.item() * norm


class LogParser:
    def __init__(self, indir, outdir, filters, k, log_format):
        self.path = indir
        self.logName = None
        self.savePath = outdir
        self.filters = filters
        self.k = k
        self.df_log = None
        self.log_format = log_format
        self.tokenizer = LogTokenizer(filters)

        self.results_fine_tune = []


    def num_there(self, s):
        digits = [i.isdigit() for i in s]
        return True if np.mean(digits) > 0.0 else False

    def parse(self, logName, batch_size=5, mask_percentage=1.0, pad_len=150, N=1, d_model=256,
              dropout=0.1, lr=0.001, betas=(0.9, 0.999), weight_decay=0.005, nr_epochs=5, num_samples=0, step_size=10):
        self.logName = logName
        self.mask_percentage = mask_percentage
        self.pad_len = pad_len
        self.batch_size = batch_size
        self.N = N
        self.d_model = d_model
        self.dropout = dropout
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.num_samples = num_samples
        self.nr_epochs = nr_epochs
        self.step_size = step_size
        self.load_data()

        if not os.path.exists(self.savePath):
            os.makedirs(self.savePath)

        df_len = self.df_log.shape[0]

        data_tokenized = []

        for i in trange(0, df_len):
            tokenized = self.tokenizer.tokenize('<CLS> ' + self.df_log.iloc[i].Content)
            data_tokenized.append(tokenized)

        train_dataloader, test_dataloader = self.get_dataloaders(data_tokenized)




        criterion = nn.CrossEntropyLoss()
        model = self.make_model(self.tokenizer.n_words, self.tokenizer.n_words, N=self.N, d_model=self.d_model,
                                d_ff=self.d_model,
                                dropout=self.dropout, max_len=self.pad_len)
        model.cuda()
        model_opt = torch.optim.Adam(model.parameters(), lr=self.lr, betas=self.betas, weight_decay=self.weight_decay)

        for epoch in range(self.nr_epochs):
            model.train()
            print("Epoch", epoch)
            self.run_epoch(train_dataloader, model, SimpleLossCompute(model.generator, criterion, model_opt))
            torch.save(model.state_dict(), self.savePath + 'model_parser_' + self.logName + str(epoch) + '.pt')

        # model.load_state_dict(torch.load('./AttentionParserResult/model_parser_BGL_2k.log3.pt'))
        # model.cuda()
        results = self.run_test(test_dataloader, model, SimpleLossCompute(model.generator, criterion, None, is_test=True))

        data_words = []
        indices_from = []

        for i, (x, y, ind) in enumerate(results):

            # print(ind)
            for j in range(len(x)):
                if not self.num_there(self.tokenizer.index2word[y[j]]):
                    if y[j] in x[j][-self.k:]:
                        data_words.append(self.tokenizer.index2word[y[j]])
                    else:
                        data_words.append("<*>")
                else:
                    data_words.append("<*>")

            indices_from += ind.tolist()

        p = pd.DataFrame({"indices": indices_from, "predictions": data_words})
        p = p.groupby('indices')['predictions'].apply(list).reset_index()

        parsed_logs = []
        for i in p.predictions.values:
            parsed_logs.append(str(''.join(i)).strip())

        df_event = self.outputResult(parsed_logs)
        df_event.to_csv(self.savePath + self.logName + "_structured.csv", index=False)

    def fine_tune(self, logName, path_stored_model, targets, batch_size=5, mask_percentage=1.0, pad_len=150, N=1, d_model=256,
              dropout=0.1, lr=0.001, betas=(0.9, 0.999), weight_decay=0.005, nr_epochs=5, num_samples=0, step_size=10):
        self.logName = logName
        self.mask_percentage = mask_percentage
        self.pad_len = pad_len
        self.batch_size = batch_size
        self.N = N
        self.d_model = d_model
        self.dropout = dropout
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.num_samples = num_samples
        self.nr_epochs = nr_epochs
        self.step_size = step_size
        self.load_data()

        if not os.path.exists(self.savePath):
            os.makedirs(self.savePath)

        df_len = self.df_log.shape[0]

        data_tokenized = []

        for i in trange(0, df_len):
            tokenized = self.tokenizer.tokenize('<CLS> ' + self.df_log.iloc[i].Content)
            data_tokenized.append(tokenized)



        train_dataloader, test_dataloader = self.get_dataloaders_fine_tune([data_tokenized, targets])


        criterion = nn.CrossEntropyLoss()

        model = self.make_fine_tune(self.tokenizer.n_words, self.tokenizer.n_words, N=self.N, d_model=self.d_model, d_ff=self.d_model,  dropout=self.dropout, max_len=self.pad_len)

        model.load_state_dict(torch.load(path_stored_model))

        model.generator = Generator(d_model, 2)

        model.cuda()
        model_opt = torch.optim.Adam(model.parameters(), lr=self.lr, betas=self.betas, weight_decay=self.weight_decay)

        for epoch in range(self.nr_epochs):
            model.train()
            print("Epoch", epoch)
            self.run_epoch_fine_tune(train_dataloader, model, SimpleLossCompute(model.generator, criterion, model_opt))
            torch.save(model.state_dict(), self.savePath + 'fine_tuned_model_BGL_10_' + self.logName + str(epoch) + '.pt')

        # model.load_state_dict(torch.load('./AttentionParserResult/model_parser_BGL_2k.log3.pt'))
        # model.cuda()
        self.results_fine_tune = self.run_test_fine_tuning(test_dataloader, model, SimpleLossCompute(model.generator, criterion, None, is_test=True))

        data_words = []
        indices_from = []


        # self.predictions = []
        # self.true = []

        # for i, (x, y) in enumerate(results):
        #     for j in range(len(x)):
        #         predictions.append(x[j])
        #         true.append(y[j])
        #
        #
        # # print(len(predictions))
        # # print(len(true))
        # #
        # # print(true[0].shape)
        # #
        # #
        # #
        # # print(predictions[0].shape)
        # #
        # # print(predictions[0])
        # # print(predictions[0])
        #
        #
        # print("Accuracy score: {}".format(accuracy_score(true, predictions)))
        # print("Precision score: {}".format(precision_score(true, predictions)))
        # print("Recall score: {}".format(recall_score(true, predictions)))
        # print("F1 score: {}".format(f1_score(true, predictions)))
        # #
        #     # print(ind)
        #     for j in range(len(x)):
        #         if not self.num_there(self.tokenizer.index2word[y[j]]):
        #             if y[j] in x[j][-self.k:]:
        #                 data_words.append(self.tokenizer.index2word[y[j]])
        #             else:
        #                 data_words.append("<*>")
        #         else:
        #             data_words.append("<*>")
        #
        #     indices_from += ind.tolist()
        #
        # p = pd.DataFrame({"indices": indices_from, "predictions": data_words})
        # p = p.groupby('indices')['predictions'].apply(list).reset_index()
        #
        # parsed_logs = []
        # for i in p.predictions.values:
        #     parsed_logs.append(str(''.join(i)).strip())
        #
        # df_event = self.outputResult(parsed_logs)
        # df_event.to_csv(self.savePath + self.logName + "_structured.csv", index=False)





    def get_dataloaders(self, data_tokenized):
        transform_to_tensor = transforms.Lambda(lambda lst: torch.tensor(lst))


        train_data = data_tokenized[:int(0.8*len(data_tokenized))]
        test_data = data_tokenized[int(0.8*len(data_tokenized)):]

        train_data = MaskedDataset(train_data, self.tokenizer, mask_percentage=self.mask_percentage,
                                   transforms=transform_to_tensor, pad_len=self.pad_len)
        weights = train_data.get_sample_weights()
        if self.num_samples != 0:
            train_sampler = WeightedRandomSampler(weights=list(weights), num_samples=self.num_samples, replacement=True)
        if self.num_samples == 0:
            train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.batch_size)

        test_data = MaskedDataset(test_data, self.tokenizer, mask_percentage=self.mask_percentage,
                                  transforms=transform_to_tensor, pad_len=self.pad_len)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=self.batch_size)
        return train_dataloader, test_dataloader


    def get_dataloaders_fine_tune(self, data_tokenized):
        transform_to_tensor = transforms.Lambda(lambda lst: torch.tensor(lst))


        train_data_x = data_tokenized[0][:int(0.8*len(data_tokenized[0]))]
        print(len(train_data_x))
        train_data_y = data_tokenized[1][:int(0.8 * len(data_tokenized[0]))]
        test_data_x = data_tokenized[0][int(0.8 * len(data_tokenized[0])):]
        test_data_y = data_tokenized[1][int(0.8 * len(data_tokenized[0])):]



        #print(len(data_tokenized))
        train_data = MaskedDataset(train_data_x, self.tokenizer, mask_percentage=self.mask_percentage, transforms=transform_to_tensor, pad_len=self.pad_len, targets=train_data_y)
        weights = train_data.get_sample_weights()
        if self.num_samples != 0:
            train_sampler = WeightedRandomSampler(weights=list(weights), num_samples=self.num_samples, replacement=True)
        if self.num_samples == 0:
            train_sampler = RandomSampler(train_data)


        # print(train_data.data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.batch_size)

        test_data = MaskedDataset(test_data_x, self.tokenizer, mask_percentage=self.mask_percentage, transforms=transform_to_tensor, pad_len=self.pad_len, targets=test_data_y)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=self.batch_size)

        return train_dataloader, test_dataloader


    def load_data(self):
        headers, regex = self.generate_logformat_regex(self.log_format)
        self.df_log = self.log_to_dataframe(os.path.join(self.path, self.logName), regex, headers, self.log_format)

    def log_to_dataframe(self, log_file, regex, headers, logformat):
        """ Function to transform log file to dataframe
        """
        log_messages = []
        linecount = 0
        with open(log_file, 'r') as fin:
            for line in fin.readlines():
                try:
                    match = regex.search(line.strip())
                    message = [match.group(header) for header in headers]
                    log_messages.append(message)
                    linecount += 1
                except Exception as e:
                    pass
        logdf = pd.DataFrame(log_messages, columns=headers)
        logdf.insert(0, 'LineId', None)
        logdf['LineId'] = [i + 1 for i in range(linecount)]
        return logdf

    def generate_logformat_regex(self, logformat):
        """ Function to generate regular expression to split log messages
        """
        headers = []
        splitters = re.split(r'(<[^<>]+>)', logformat)
        regex = ''
        for k in range(len(splitters)):
            if k % 2 == 0:
                splitter = re.sub(' +', '\\\s+', splitters[k])
                regex += splitter
            else:
                header = splitters[k].strip('<').strip('>')
                regex += '(?P<%s>.*?)' % header
                headers.append(header)
        regex = re.compile('^' + regex + '$')
        return headers, regex

    def do_mask(self, batch):
        c = copy.deepcopy
        token_id = self.tokenizer.tokenize('<MASK>')[0]
        srcs, offsets, data_lens, indices = batch
        src, trg, idxs = [], [], []

        for i, _ in enumerate(data_lens):
            data_len = c(data_lens[i].item())
            # print(data_lens[i].item())
            dg = c(indices[i].item())
            masked_data = c(srcs[i])
            offset = offsets[i].item()
            num_masks = round(self.mask_percentage * data_len)
            if self.mask_percentage < 1.0:
                masked_indices = np.random.choice(np.arange(offset, offset + data_len),
                                                  size=num_masks if num_masks > 0 else 1, replace=False)
            else:
                masked_indices = np.arange(offset, offset + data_len)

            masked_indices.sort()

            for j in masked_indices:
                tmp = c(masked_data)
                label_y = c(tmp[j])
                tmp[j] = token_id
                src.append(c(tmp))

                trg.append(label_y)
                idxs.append(dg)
        return torch.stack(src), torch.stack(trg), torch.Tensor(idxs)


    def do_mask_fine_tune(self, batch):
        c = copy.deepcopy
        token_id = self.tokenizer.tokenize('<MASK>')[0]
        srcs, offsets, data_lens, indices, labels = batch
        src, trg, idxs, lab = [], [], [], []



        for i, _ in enumerate(data_lens):
            data_len = c(data_lens[i].item())
            # print(data_lens[i].item())
            dg = c(indices[i].item())
            masked_data = c(srcs[i])
            offset = offsets[i].item()
            num_masks = round(self.mask_percentage * data_len)
            if self.mask_percentage < 1.0:
                masked_indices = np.random.choice(np.arange(offset, offset + data_len),
                                                  size=num_masks if num_masks > 0 else 1, replace=False)
            else:
                masked_indices = np.arange(offset, offset + data_len)

            masked_indices.sort()

            # print(type(labels[i]))
            # exit(-1)


            # print(len(masked_indices))

            lab.append(labels[i])
            src.append(srcs[i])

            # for j in masked_indices:
            #     lab.append(labels[i])
            #
            #     tmp = c(masked_data)
            #     label_y = c(tmp[j])
            #     tmp[j] = token_id
            #     src.append(c(tmp))
            #
            #
            #     trg.append(label_y)
            #     idxs.append(dg)



        return torch.stack(src), None, None, torch.stack(lab)

    def run_epoch_fine_tune(self, dataloader, model, loss_compute):

        start = time.time()
        total_tokens = 0
        total_loss = 0
        tokens = 0


        #print(dataloader.dataset)
        for i, batch in enumerate(dataloader):
            # print(len(batch))
            b_input, b_labels, _, labels = self.do_mask_fine_tune(batch)


            #print("The shape of the input is {}".format(b_input.shape))
            #print("The shape of the b_labels is {}".format(b_labels.shape))
            #print("The shape of the labels is {}".format(labels.shape))


            batch = Batch_binary_classification(b_input, labels, 0, labels=labels)

            out = model.forward(batch.src.cuda(), batch.trg.cuda(), batch.src_mask.cuda(), batch.trg_mask.cuda())

            # print(out.shape)

            loss = loss_compute(out, labels.cuda(), batch.ntokens)
            total_loss += loss
            total_tokens += batch.ntokens
            tokens += batch.ntokens

            if i % self.step_size == 1:
                elapsed = time.time() - start
                #print("Epoch Step: %d / %d Loss: %f Tokens per Sec: %f" %
                #      (i, len(dataloader), loss / batch.ntokens, tokens / elapsed))
                start = time.time()
                tokens = 0
        return total_loss / total_tokens

    def run_epoch(self, dataloader, model, loss_compute):

        start = time.time()
        total_tokens = 0
        total_loss = 0
        tokens = 0
        for i, batch in enumerate(dataloader):
            # print(type(batch))
            # print(len(batch))
            b_input, b_labels, _ = self.do_mask(batch)
            batch = Batch(b_input, b_labels, 0)

            out = model.forward(batch.src.cuda(), batch.trg.cuda(),   batch.src_mask.cuda(), batch.trg_mask.cuda())

            loss = loss_compute(out, batch.trg_y.cuda(), batch.ntokens)
            total_loss += loss
            total_tokens += batch.ntokens
            tokens += batch.ntokens

            if i % self.step_size == 1:
                elapsed = time.time() - start
                print("Epoch Step: %d / %d Loss: %f Tokens per Sec: %f" %
                      (i, len(dataloader), loss / batch.ntokens, tokens / elapsed))
                start = time.time()
                tokens = 0
        return total_loss / total_tokens

    def run_test(self, dataloader, model, loss_compute):
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                b_input, b_labels, ind = self.do_mask(batch)

                batch = Batch(b_input, b_labels, 0)
                out = model.forward(batch.src.cuda(), batch.trg.cuda(),
                                    batch.src_mask.cuda(), batch.trg_mask.cuda())
                out_p = model.generator(out)  # batch_size, hidden_dim

                if i % self.step_size == 1:
                    print("Epoch Step: %d / %d" % (i, len(dataloader)))
                yield out_p.cpu().numpy().argsort(axis=1), b_labels.cpu().numpy(), ind.cpu()

    def run_test_fine_tuning(self, dataloader, model, loss_compute):
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                b_input, b_labels, ind, labels = self.do_mask_fine_tune(batch)

                batch = Batch(b_input, labels, 0)
                out = model.forward(batch.src.cuda(), batch.trg.cuda(), batch.src_mask.cuda(), batch.trg_mask.cuda())
                out_p = model.generator(out)  # batch_size, hidden_dim

                # print(out_p)
                # print(labels)
                # print(ind)

                if i % self.step_size == 1:
                    print("Epoch Step: %d / %d" % (i, len(dataloader)))
                    print(out_p.shape, labels.shape)
                    print(out_p.cpu().numpy().argmax(axis=1), labels.cpu().numpy())
                yield out_p.cpu().numpy().argmax(axis=1), labels.cpu().numpy()



    def outputResult(self, pred):
        df_events = []
        templateids = []
        for pr in pred:
            template_id = hashlib.md5(pr.encode('utf-8')).hexdigest()
            templateids.append(template_id)
            df_events.append([template_id, pr])

        df_event = pd.DataFrame(df_events, columns=['EventId', 'EventTemplate'])
        return df_event

    def make_model(self, src_vocab, tgt_vocab, N=3,
                   d_model=512, d_ff=2048, h=8, dropout=0.1, max_len=20):
        "Helper: Construct a model from hyperparameters."
        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout, max_len)
        model = EncoderDecoder(
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
            Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
            nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
            nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
            Generator(d_model, tgt_vocab))
        # print(model)

        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)
        return model


    def make_fine_tune(self, src_vocab, tgt_vocab=2, N=3,
                   d_model=512, d_ff=2048, h=8, dropout=0.1, max_len=20):
        "Helper: Construct a model from hyperparameters."
        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout, max_len)
        model = EncoderDecoder_ForClassification(
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
            Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
            nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
            nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
            Generator(d_model, tgt_vocab))
        # print(model)

        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)
        return model



class FineTune_LogParser(LogParser):
    def __init__(self, path_stored_model):
        super(LogParser, self).__init__()
        model = self.make_fine_tune(self.tokenizer.n_words, self.tokenizer.n_words, N=self.N, d_model=self.d_model, d_ff=self.d_model, dropout=self.dropout, max_len=self.pad_len)
        model.load_state_dict(torch.load(path_stored_model))
        model.generator = Generator(self.d_model, 2)
        self.log_parser = model


    def forward(self, src, src_mask, labels):


        output = self.log_parser.encode(src=src, src_mask=src_mask)
        output = self.log_parser.genrator(output[0])


        return output
