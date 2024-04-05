import torch
import pandas as pd

import sys
sys.path.append('./docs/csv') # not clean but ok for now
from csv_handler import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def see_results(model, dataloader, dico):
    """
        Affiche les résultats du modèle sur le jeu de données de validation
    """
    inv_dico = {dico[key]: key for key in dico}
    model = model.to(device)
    s, p, t = [], [], [] #setences, predictions, targets
    for batch in dataloader:
        out = model.forward(batch)
        preds = torch.max(out.logits, -1).indices
        s += detokenize_sentences(batch['input_ids'])

        p += [inv_dico[pred.item()] for pred in preds]
        t += [inv_dico[true.item()] for true in batch['labels']]

        
