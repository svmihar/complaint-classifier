import torchtext
from tqdm import tqdm
import pandas as pd
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn as nn
import torch.optim as optim
from model import LSTM_word2vec
from torchtext.vocab import Vectors
""" ini dibutuhkan ketika pertama kali init
 from gensim.models import Word2Vec
 word_vecs = Word2Vec.load('./word2vec_weights/idwiki_word2vec_300.model')
 word_vecs.wv.save_word2vec_format('./word2vec_weights/w2v.bin')
 """



device = torch.device("cuda:0") # TODO: if cpu, then cpu, else cuda
bs = 32

# def load_dataset():
df = pd.read_csv("../data/label_processed.csv")
label_field = torchtext.data.Field(
    sequential=False, use_vocab=False, batch_first=True, dtype=torch.float
)
text_field = torchtext.data.Field(
    tokenize=(lambda s: s.split()),
    lower=True,
    include_lengths=True,
    batch_first=True,
)
fields = [("label", label_field), ("text", text_field)]

train, valid, test = torchtext.data.TabularDataset.splits(
    path="../data/",
    train="train.csv",
    validation="valid.csv",
    test="test.csv",
    format="CSV",
    fields=fields,
    skip_header=True,
)


train_iter = torchtext.data.BucketIterator(
    train,
    batch_size=bs,
    sort_key=lambda x: len(x.text),
    device=device,
    sort_within_batch=True,
)
test_iter = torchtext.data.BucketIterator(
    test,
    batch_size=bs,
    sort_key=lambda x: len(x.text),
    device=device,
    sort_within_batch=True,
)
valid_iter = torchtext.data.BucketIterator(
    valid,
    batch_size=bs,
    sort_key=lambda x: len(x.text),
    device=device,
    sort_within_batch=True,
)

vectors = Vectors(cache='word2vec_weights', name='w2v.bin')
text_field.build_vocab(train, min_freq=1)
text_field.vocab.set_vectors(vectors.stoi, vectors.vectors, vectors.dim)

class LSTM(nn.Module):
    def __init__(self, dimension=128):
        super(LSTM, self).__init__()

        self.embedding = nn.Embedding(len(text_field.vocab), 300)
        self.dimension = dimension
        self.lstm = nn.LSTM(
            input_size=300,
            hidden_size=dimension,
            num_layers=1,
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
        text_features = torch.squeeze(text_features, 1)
        text_out = torch.sigmoid(text_features)
        return text_out


def save_(save_path, model, optimizer, valid_loss):
    if save_path is None:
        return
    state_dict = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "valid_loss": valid_loss,
    }

    torch.save(state_dict, save_path)
    print(f"model saved to {save_path}")


def load_metrics():
    # TODO: metric evaluation mode
    pass


def train(
    model,
    opt,
    criterion=nn.BCEWithLogitsLoss(),
    train_loader=train_iter,
    valid_loader=valid_iter,
    num_epochs=5,
    eval_every=len(train_iter) // 2,
    file_path="./models",
    best_valid_loss=float("inf"),
):
    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list, valid_loss_list, valid_loss_list = [], [], []
    global_steps_list = []

    model.train()
    for epoch in tqdm(range(num_epochs)):
        for (labels, (text, text_len)), _ in train_loader:
            labels = labels.to(device)
            text = text.to(device)
            text_len = text_len.to(device)
            output = model(text, text_len)

            loss = criterion(output, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()

            running_loss += loss.item()
            global_step += 1

            if global_step % eval_every == 0:
                model.eval()
                with torch.no_grad():
                    # val loop
                    for (labels, (text, text_len)), _ in valid_loader:
                        labels = labels.to(device)
                        text = text.to(device)
                        text_len = text_len.to(device)

                        output = model(text, text_len)
                        loss = criterion(output, labels)
                        valid_running_loss += loss.item()

            average_train_loss = running_loss / eval_every
            average_valid_loss = valid_running_loss / len(valid_loader)
            train_loss_list.append(average_train_loss)
            valid_loss_list.append(average_valid_loss)
            global_steps_list.append(global_step)

            # reset all running values
            running_loss = 0.0
            valid_running_loss = 0.0

            model.train()

            # print(f"train_loss: {average_train_loss}\nvalid_loss: {average_valid_loss}")

            # save checkpoint if best
            if best_valid_loss > average_valid_loss:
                best_valid_loss = average_valid_loss
                save_("./models/model.pt", model, opt, best_valid_loss)
                # TODO: save metrics


if __name__ == "__main__":
    # model = LSTM_1(text_field=text_field).to(device)
    # model = LSTM().to(device)
    model = LSTM_word2vec(text_field=text_field).to(device)
    opt = optim.AdamW(model.parameters(), lr=1e-2)
    train(model, opt=opt, num_epochs=300)
