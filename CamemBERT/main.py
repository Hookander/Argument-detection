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
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from tqdm.notebook import tqdm

from tokenizer import *

camembert = AutoModelForSequenceClassification.from_pretrained('camembert-base')


def take_first_embedding(embeddings, attention_mask=None):
    return embeddings[:, 0]

def average_embeddings(embeddings, attention_mask):
    return torch.sum(embeddings * attention_mask.unsqueeze(-1), dim=1) / torch.sum(attention_mask, dim=1).unsqueeze(-1)

def tokenize_batch(samples, tokenizer):
    text = [sample["Utterance"] for sample in samples]
    labels = torch.tensor([sample["Label"] for sample in samples])
    str_labels = [sample["Dialogue_Act"] for sample in samples]
    # The tokenizer handles
    # - Tokenization (amazing right?)
    # - Padding (adding empty tokens so that each example has the same length)
    # - Truncation (cutting samples that are too long)
    # - Special tokens (in CamemBERT, each sentence ends with a special token </s>)
    # - Attention mask (a binary vector which tells the model which tokens to look at. For instance it will not compute anything if the token is a padding token)
    tokens = tokenizer(text, padding="longest", return_tensors="pt")

    return {"input_ids": tokens.input_ids, "attention_mask": tokens.attention_mask, "labels": labels, "str_labels": str_labels, "sentences": text}

batch_sentences = [
    "Vous savez où est la <mask> la plus proche?",
    "La Seine est un <mask>.",
    "Je cherche urgemment un endroit où retirer de l'<mask>.",
]
tokenizer_output = tokenizer(
    batch_sentences, 
    padding = "max_length",
    truncation = True,
    return_tensors = "pt"
)

dataset = load_dataset("miam", "loria")
train_dataset, val_dataset, test_dataset = dataset.values()
val_dataloader = DataLoader(val_dataset, batch_size=16, collate_fn=functools.partial(tokenize_batch, tokenizer=tokenizer), shuffle=True)
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

scatter_plot = px.scatter(x=all_representations_2d[:, 0], y=all_representations_2d[:, 1], color=str_labels)
scatter_plot.show(config={'staticPlot': True})
