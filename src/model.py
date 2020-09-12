import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn as nn
import torch.optim as optim

"""
versi ini pake batch_first = True: 
Without batch_first=True it will use the first dimension as the sequence dimension.
With batch_first=True it will use the second dimension as the sequence dimension

"""


class LSTM_2(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)


class LSTM_1(nn.Module):
    def __init__(self, text_field, dimension=128):
        super(LSTM_1, self).__init__()

        self.embedding = nn.Embedding(len(text_field.vocab), 300)
        self.dimension = dimension
        self.lstm = nn.LSTM(
            input_size=300,
            hidden_size=dimension,
            num_layers=5,
            batch_first=True,
            bidirectional=True,
        )
        self.drop = nn.Dropout(p=0.5)
        self.fc = nn.Linear(2 * dimension, 1)

    def forward(self, text, text_len):
        text_emb = self.embedding(text)
        packed_input = pack_padded_sequence(
            text_emb, text_len, batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        out_forward = output[range(len(output)), text_len - 1, : self.dimension]
        out_reverse = output[:, 0, self.dimension :]
        out_reduced = torch.cat((out_forward, out_reverse), 1)
        text_features = self.drop(out_reduced)

        text_features = self.fc(text_features)
        text_features = self.fc1(text_features)
        text_features = torch.squeeze(text_features, 1)
        text_out = torch.sigmoid(text_features)
        return text_out


class LSTM_word2vec(nn.Module):
    def __init__(self, text_field, dimension=128):
        super(LSTM_word2vec, self).__init__()
        self.embedding = nn.Embedding(len(text_field.vocab), 300).from_pretrained(
            text_field.vocab.vectors
        )

        # self.embedding = nn.Embedding(len(w.vocab), 300, padding_idx=0)
        # self.embedding.weight.data.copy_(t)
        self.embedding.weight.requires_grad_ = False
        self.dimension = dimension
        self.lstm = nn.LSTM(
            input_size=300,
            hidden_size=dimension,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.drop = nn.Dropout(p=0.2)
        self.fc = nn.Linear(2 * dimension, 1)

    def forward(self, text, text_len):
        text_emb = self.embedding(text)
        packed_input = pack_padded_sequence(
            text_emb, text_len, batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        out_forward = output[range(len(output)), text_len - 1, : self.dimension]
        out_reverse = output[:, 0, self.dimension :]
        out_reduced = torch.cat((out_forward, out_reverse), 1)
        text_features = self.drop(out_reduced)

        text_features = self.fc(text_features)
        text_features = torch.squeeze(text_features, 1)
        text_out = torch.sigmoid(text_features)
        return text_out
