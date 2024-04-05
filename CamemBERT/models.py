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
import plotly.express as px
from pytorch_lightning.loggers import WandbLogger

import sys

from results import *
from data_handler import *
sys.path.append('./docs/csv') # not clean but ok for now
from csv_handler import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model(pl.LightningModule):
    def __init__(self, model_name, num_labels, lr, weight_decay, typ, from_scratch=False):
        """
            typ = 'arg' or 'dom' for the different types of classifiction
        """
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
        self.num_labels = num_labels
        self.typ = typ

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

    def get_dico(self, typ):
        if typ == 'arg':
            return arg_dico
        elif typ == 'dom':
            return domain_dico
        else:
            print("get_dico : Invalid type")
            return

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
    
    def test_step(self, test_batch):
        labels = test_batch["labels"]
        out = self.forward(test_batch)

        preds = torch.max(out.logits, -1).indices
 
        acc = (test_batch["labels"] == preds).float().mean()

        self.log("test/acc", acc)

        f1 = f1_score(test_batch["labels"].cpu().tolist(), preds.cpu().tolist(), average="macro")
        self.log("test/f1", f1)

    def get_trainer(self, save, max_epochs, patience, wandb = True):
        if save:
            model_checkpoint = pl.callbacks.ModelCheckpoint(monitor="valid/acc", mode="max")
            if wandb:
                trainer = pl.Trainer(
                    max_epochs=max_epochs,
                    callbacks=[
                        pl.callbacks.EarlyStopping(monitor="valid/acc", patience=patience, mode="max"),
                        model_checkpoint,
                    ],
                    logger = WandbLogger(project="camembert_"+self.typ, log_model=False)
                )
            else:
                trainer = pl.Trainer(
                    max_epochs=max_epochs,
                    callbacks=[
                        pl.callbacks.EarlyStopping(monitor="valid/acc", patience=patience, mode="max"),
                        model_checkpoint,
                    ]
                )
        else:
            if wandb:
                trainer = pl.Trainer(
                    max_epochs=max_epochs,
                    callbacks=[
                        pl.callbacks.EarlyStopping(monitor="valid/f1", patience=patience, mode="max"),
                    ],
                    logger = WandbLogger(project="camembert_"+self.typ, log_model=False)
                )
            else:
                trainer = pl.Trainer(
                    max_epochs=max_epochs,
                    callbacks=[
                        pl.callbacks.EarlyStopping(monitor="valid/acc", patience=patience, mode="max"),
                    ]
                )
        return trainer
    
    def train_model(self, batch_size=16, patience = 10, max_epochs = 50, test = True, ratio = [0.8, 0.1], wandb = True, save = False):


        camembert_trainer = self.get_trainer(save, max_epochs, patience, wandb)
        sentences, args_labels, domains = get_data_with_simp_labels(shuffle = False)
        tokenized_sentences = tokenize_sentences(sentences)
        if self.typ == 'arg':
            cleaned_labels = args_labels
        elif self.typ == 'dom':
            cleaned_labels = domains
        else:
            print("Invalid type")
            return
        train_dl, val_dl, test_dl = get_dataloaders(tokenized_sentences, cleaned_labels, ratio=ratio, batch_size=batch_size)
        print('ok')
        camembert_trainer.fit(self, train_dataloaders=train_dl, val_dataloaders=val_dl)

        if test:
            ret = camembert_trainer.test(model = self, dataloaders=test_dl)
            print(ret)
            see_results(self, test_dl, self.get_dico(self.typ))
            return ret



num_labels = 21
lightning_model = Model("camembert-base", num_labels, lr=3e-4, weight_decay=0., typ = 'dom')
lightning_model.train_model(batch_size=16, patience=10, max_epochs=30, test=True, wandb = True, ratio=[0.7, 0.15], save = False)



