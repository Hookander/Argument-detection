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

from data_handler import *
sys.path.append('./docs/csv') # not clean but ok for now
from csv_handler import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ArgDetector(pl.LightningModule):
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
    
    def test_step(self, test_batch):
        labels = test_batch["labels"]
        out = self.forward(test_batch)

        preds = torch.max(out.logits, -1).indices
 
        acc = (test_batch["labels"] == preds).float().mean()

        self.log("test/acc", acc)

        f1 = f1_score(test_batch["labels"].cpu().tolist(), preds.cpu().tolist(), average="macro")
        self.log("test/f1", f1)

    
    def train_model(self, batch_size=16, patience = 10, max_epochs = 50, test = True, ratio = [0.7, 0.15], wandb = True):


        model_checkpoint = pl.callbacks.ModelCheckpoint(monitor="valid/acc", mode="max")
        if wandb:
            camembert_trainer = pl.Trainer(
                max_epochs=max_epochs,
                callbacks=[
                    pl.callbacks.EarlyStopping(monitor="valid/acc", patience=patience, mode="max"),
                    model_checkpoint,
                ],
                logger = WandbLogger(project="camembert")
            )
        else:
            camembert_trainer = pl.Trainer(
                max_epochs=max_epochs,
                callbacks=[
                    pl.callbacks.EarlyStopping(monitor="valid/acc", patience=patience, mode="max"),
                    model_checkpoint,
                ]
            )
        sentences, cleaned_labels = get_data_with_simp_labels(shuffle = False)
        tokenized_sentences = tokenize_sentences(sentences)

        train_dl, val_dl, test_dl = get_dataloaders(tokenized_sentences, cleaned_labels, ratio=ratio, batch_size=batch_size)
        camembert_trainer.fit(lightning_model, train_dataloaders=train_dl, val_dataloaders=val_dl)

        if test:
            ret = camembert_trainer.test(model = lightning_model, dataloaders=test_dl)
            print(ret)
            return ret



#num_labels = 3
#lightning_model = ArgDetector("camembert-base", num_labels, lr=3e-5, weight_decay=0.)
#lightning_model.train_model(batch_size=16, patience=10, max_epochs=50, test=True, wandb = True, ratio=[0.7, 0.15])
