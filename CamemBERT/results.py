import torch
import pandas as pd
from data_handler import *
import numpy as np

import sys
sys.path.append('./docs/csv') # not clean but ok for now
from csv_handler import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def see_results(model, dataloader, dico, out_path = './CamemBERT/results.csv'):
    """
        Donne un csv avec les résultats du modèle sur le jeu de données 

    """
    inv_dico = {dico[key]: key for key in dico}

    #After the training the model is moved back to the cpu, we need to move it back to the device
    model = model.to(device)
    s, p, t = [], [], [] #sentences, predictions, targets
    for batch in dataloader:
        out = model.forward(batch)
        preds = torch.max(out.logits, -1).indices
        s += detokenize_sentences(batch['input_ids'].numpy())

        p += [inv_dico[pred.item()] for pred in preds]
        t += [inv_dico[true.item()] for true in batch['labels']]


    df = pd.DataFrame({'Sentences': s, 'Predictions': p, 'Targets': t})
    df.to_csv(out_path)

