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

def take_first_embedding(embeddings, attention_mask=None):
    return embeddings[:, 0]

def average_embeddings(embeddings, attention_mask):
    return torch.sum(embeddings * attention_mask.unsqueeze(-1), dim=1) / torch.sum(attention_mask, dim=1).unsqueeze(-1)

def show_tsne(model, clean_props = True, ratio = [0.7, 0.15]):
    sentences, cleaned_labels = get_data()

    tokenized_sentences = tokenize_sentences(sentences)

    train_dl, val_dl, test_dl = get_dataloaders(tokenized_sentences, cleaned_labels, ratio=ratio, batch_size=16)

    all_representations = torch.tensor([], device=device)
    with torch.no_grad():
        for tokenized_batch in val_dl:
            model_output = model(
                input_ids = tokenized_batch["input_ids"],
                attention_mask = tokenized_batch["attention_mask"],
                output_hidden_states=True
            )
            batch_representations = average_embeddings(model_output["hidden_states"][-1], tokenized_batch["attention_mask"])
            all_representations = torch.cat((all_representations, batch_representations), 0)

    labels = get_data(clear_labels=False)[1]
    val_labels = get_labels_from_ratio(labels, ratio)[1]
    if clean_props:
        for i in range(len(val_labels)):
            val_labels[i] = val_labels[i].strip()
            if val_labels[i][0] == 'P' and val_labels[i][1:].isdigit(): # Pn
                val_labels[i] = 'Pn'
    tsne = TSNE()
    all_representations_2d = tsne.fit_transform(all_representations)
    print(all_representations_2d.shape)
    scatter_plot = px.scatter(x=all_representations_2d[:, 0], y=all_representations_2d[:, 1], color=val_labels)
    scatter_plot.show(config={'staticPlot': True})

#show_tsne(camembert, ratio = [0.1, 0.1])

class LightningModel(pl.LightningModule):
    def __init__(self, model_name, num_labels, lr, weight_decay, from_scratch=False):
        super().__init__()
        self.save_hyperparameters()
        if from_scratch:
            # Si `from_scratch` est vrai, on charge uniquement la config (nombre de couches, hidden size, etc.) 
            # et pas les poids du modèle 
            config = AutoConfig.from_pretrained(
                model_name, num_labels=num_labels
            )
            self.model = AutoModelForSequenceClassification.from_config(config)
        else:
            # Cette méthode permet de télécharger le bon modèle pré-entraîné directement 
            # depuis le Hub de HuggingFace sur lequel sont stockés de nombreux modèles
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=num_labels
            ).to(device)
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_labels = self.model.num_labels

    def forward(self, batch):
        return self.model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device)
        )

    def training_step(self, batch):
        out = self.forward(batch)

        logits = out.logits
        # -------- MASKED --------
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits.view(-1, self.num_labels), batch["labels"].view(-1))

        # ------ END MASKED ------

        self.log("train/loss", loss)

        return loss

    def validation_step(self, batch, batch_index):
        labels = batch["labels"]
        out = self.forward(batch)

        preds = torch.max(out.logits, -1).indices
        # -------- MASKED --------
        acc = (batch["labels"] == preds).float().mean()
        # ------ END MASKED ------
        self.log("valid/acc", acc)

        f1 = f1_score(batch["labels"].cpu().tolist(), preds.cpu().tolist(), average="macro")
        self.log("valid/f1", f1)

    def predict_step(self, batch, batch_idx):
        """La fonction predict step facilite la prédiction de données. Elle est 
        similaire à `validation_step`, sans le calcul des métriques.
        """
        out = self.forward(batch)

        return torch.max(out.logits, -1).indices

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

num_labels = 9
lightning_model = LightningModel("camembert-base", num_labels, lr=3e-5, weight_decay=0.)

model_checkpoint = pl.callbacks.ModelCheckpoint(monitor="valid/acc", mode="max")

camembert_trainer = pl.Trainer(
    max_epochs=1,
    callbacks=[
        pl.callbacks.EarlyStopping(monitor="valid/acc", patience=4, mode="max"),
        model_checkpoint,
    ]
)


##sentences, cleaned_labels = get_data()
#tokenized_sentences = tokenize_sentences(sentences)

#train_dl, val_dl, test_dl = get_dataloaders(tokenized_sentences, cleaned_labels, ratio=[0.7, 0.15], batch_size=16)
#camembert_trainer.fit(lightning_model, train_dataloaders=train_dl, val_dataloaders=val_dl)

