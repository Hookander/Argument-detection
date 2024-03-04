from encoder import *
from decoder import *
from positional_encoding import *
import numpy as np
import torch.optim as optim
from torchinfo import summary
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
import torchtext
import pandas as pd

class Transformer(nn.Module):
    def __init__(self, vocab1_size, vocab2_size, d_model, n_heads, n_layers):
        super().__init__()
        self.input_embedding = nn.Embedding(vocab1_size, d_model)
        self.output_embedding = nn.Embedding(vocab2_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)

        self.encoder = TransformerEncoder(d_model, n_heads, n_layers)
        self.decoder = TransformerDecoder(d_model, n_heads, n_layers)
        self.last_layer = nn.Linear(d_model, vocab2_size)


    def forward(self, input_ids, decoder_input_ids, key_padding_mask = None):
        """
        input_ids: (batch_size, seq_len_x)
        decoder_input_ids: (batch_size, seq_len_y)
        """
        encoder_input = self.input_embedding(input_ids)
        decoder_input = self.output_embedding(decoder_input_ids)

        encoder_input = self.pos_enc(encoder_input)
        decoder_input = self.pos_enc(decoder_input)

        #print("encoder_input", encoder_input.shape)
        encoder_output = self.encoder(encoder_input, key_padding_mask = key_padding_mask)
        decoder_output = self.decoder(decoder_input, encoder_output, key_padding_mask = key_padding_mask)
        output = self.last_layer(decoder_output) # (batch_size, seq_len_y, vocab_size)
        return output
    
    def train_m(self, X_train, Y_train, key_padding_mask,loss_padding_mask, epochs):
        optimizer = optim.Adam(self.parameters(), lr=1e-5)
        loss_fn = nn.CrossEntropyLoss(reduction = "none")

        for epoch in range(epochs):
            for i in range(len(X_train)):
                optimizer.zero_grad()
                X_train_batch = X_train[i] # (batch_size, seq_len_x)
                Y_train_batch = Y_train[i] # (batch_size, seq_len_y)

                Y_train_input = Y_train_batch[:, :-1] # (batch_size, seq_len_y-1)
                Y_train_target = Y_train_batch[:, 1:] # (batch_size, seq_len_y-1)

                output = transformer(X_train_batch, Y_train_input, key_padding_mask[i]) # (batch_size, seq_len_y-1, vocab_size)
                output = output.permute(0, 2, 1) # (batch_size, vocab_size, seq_len_y-1), crossentropy requires this

                loss = loss_fn(output, Y_train_target)
                loss = loss * (1 - loss_padding_mask[i, :, 1:].float()) # (batch_size, seq_len_y-1)
                loss = loss.sum()
                if loss_padding_mask[i, :, 1:].sum() != 0 :

                    loss = loss/loss_padding_mask[i, :, 1:].sum()

                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch}, loss {loss.item()}")

df = pd.read_csv('./docs/csv/csvsum.csv')

X_train = df['PAROLES'].values
Y_train = df['Dimension Dialogique'].values
#print(X_train)
tokenizer = get_tokenizer("basic_english")

vocab = build_vocab_from_iterator(map(tokenizer, np.concatenate([X_train, Y_train])), specials=["<pad>", "<start>", "<end>", "<unk>"])

X_train = [torch.tensor(vocab(tokenizer(x)), dtype=torch.long) for x in X_train]
Y_train = [torch.tensor(vocab(tokenizer("<start> " + y + " <end>")), dtype=torch.long) for y in Y_train]

# pad
X_train = torch.nn.utils.rnn.pad_sequence(X_train, batch_first=True, padding_value=vocab["<pad>"])  # (batch_size, seq_len)
Y_train = torch.nn.utils.rnn.pad_sequence(Y_train, batch_first=True)

# batch, first
batch_size = 1
seq_len_x = X_train.shape[1]
seq_len_y = Y_train.shape[1]
X_train = X_train.reshape(-1, batch_size, seq_len_x)
Y_train = Y_train.reshape(-1, batch_size, seq_len_y)


# key padding mask
key_padding_mask = (X_train == vocab["<pad>"]) # (-1, batch_size, seq_len_x)
loss_padding_mask = (Y_train == vocab["<pad>"]) # (-1, batch_size, seq_len_y)

# transformer
transformer = Transformer(len(vocab), len(vocab), 128, 4, 2)
transformer.load_state_dict(torch.load('./Transformer/models/model_0_300.pt'))
#transformer.train_m(X_train, Y_train, key_padding_mask, loss_padding_mask, 30)

#x_test = "il faut faire cela parce que il a raison "
x_test = "je pense que il faut faire ceci parce que bien"

x_test = torch.tensor(vocab(tokenizer(x_test)), dtype=torch.long).reshape(1, -1) # (1, seq_len_x)

output = [vocab["<start>"]]

for i in range(30):
    y_test = torch.tensor(output, dtype=torch.long).reshape(1, -1) # (1, seq_len_y)

    y_pred = transformer(x_test, y_test) # (1, 1, vocab_size)
    y_pred = y_pred[0, -1, :].argmax().item()
    output.append(y_pred)

    if y_pred == vocab["<end>"]:
        break

vocab_itos = vocab.get_itos()
output = [vocab_itos[i] for i in output]
print(" ".join(output))
