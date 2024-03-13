from pprint import pprint
import functools
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoModelForSequenceClassification, CamembertForMaskedLM, AutoTokenizer, AutoConfig
from datasets import load_dataset
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.manifold import TSNE
#import matplotlib.pyplot as plt
#import seaborn as sns
import plotly.express as px
#from tqdm.notebook import tqdm
import sys

from data_handler import *
sys.path.append('./docs/csv') # not clean but ok for now
from csv_handler import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

camembert = AutoModelForSequenceClassification.from_pretrained('camembert-base')
camembert = camembert.to(device)


def take_first_embedding(embeddings, attention_mask=None):
    return embeddings[:, 0]

def average_embeddings(embeddings, attention_mask):
    return torch.sum(embeddings * attention_mask.unsqueeze(-1), dim=1) / torch.sum(attention_mask, dim=1).unsqueeze(-1)

sentences, cleaned_labels = get_data()

tokenized_sentences = tokenize_sentences(sentences)

ratio = [0.1, 0.5]
train_dl, val_dl, test_dl = get_dataloaders(tokenized_sentences, cleaned_labels, ratio=ratio, batch_size=16)

all_representations = torch.tensor([], device=device)
with torch.no_grad():
    for tokenized_batch in val_dl:
        model_output = camembert(
            input_ids = tokenized_batch["input_ids"],
            attention_mask = tokenized_batch["attention_mask"],
            output_hidden_states=True
        )
        batch_representations = average_embeddings(model_output["hidden_states"][-1], tokenized_batch["attention_mask"])
        all_representations = torch.cat((all_representations, batch_representations), 0)

labels = get_data(clear_labels=False)[1]
val_labels = get_labels_from_ratio(labels, ratio)[1]
tsne = TSNE()
all_representations_2d = tsne.fit_transform(all_representations)
print(all_representations_2d.shape)
scatter_plot = px.scatter(x=all_representations_2d[:, 0], y=all_representations_2d[:, 1], color=val_labels)
scatter_plot.show(config={'staticPlot': True})
"""

dataset = load_dataset("miam", "loria")
train_dataset, val_dataset, test_dataset = dataset.values()
print(train_dataset)

#val_dataloader = DataLoader(val_dataset, batch_size=16, collate_fn=functools.partial(tokenize_batch, tokenizer=tokenizer), shuffle=True)
#print(next(iter(val_dataloader)))

sentences = []
labels = []
str_labels = []
all_representations = torch.Tensor()


with torch.no_grad():
    for tokenized_batch in val_dataloader:
        model_output = camembert(
            input_ids = tokenized_batch["input_ids"],
            attention_mask = tokenized_batch["attention_mask"],
            output_hidden_states=True
        )
        batch_representations = average_embeddings(model_output["hidden_states"][-1], tokenized_batch["attention_mask"])
        sentences.extend(tokenized_batch["sentences"])
        labels.extend(tokenized_batch["labels"])
        str_labels.extend(tokenized_batch["str_labels"])
        all_representations = torch.cat((all_representations, batch_representations), 0)

print(all_representations.shape)

tsne = TSNE()
all_representations_2d = tsne.fit_transform(all_representations)
print(all_representations_2d.shape)
scatter_plot = px.scatter(x=all_representations_2d[:, 0], y=all_representations_2d[:, 1], color=str_labels)
scatter_plot.show(config={'staticPlot': True})
"""
