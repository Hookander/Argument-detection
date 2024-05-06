# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, CamembertForMaskedLM, AutoTokenizer, AutoConfig
from datasets import load_dataset
from sklearn.metrics import confusion_matrix, f1_score

import sys
sys.path.append('./docs/csv') # not clean but ok for now
from csv_handler import *
sys.path.append('./CamemBERT') # not clean but ok for now
from data_handler import *

import json

 

sentences_init, labels_init, domains_init = get_data_with_simp_labels(shuffle = False)
sentences_aug, labels_aug, domains_aug =  get_data_aug()
sentences, labels, domains = sentences_init + sentences_aug, labels_init + labels_aug, domains_init + domains_aug

data_load = False
if data_load:
    camembert = AutoModelForSequenceClassification.from_pretrained(
                    "camembert-base", num_labels=3
                )
    tokenizer = AutoTokenizer.from_pretrained('camembert-base')
    tokenized_sentences = tokenize_sentences(sentences_init)

    # Create the list of the embedding of tokenized_sentences
    token_embeddings = []
    train_dl, val_dl, test_dl = get_dataloaders('arg',use_data_aug = True, batch_size = 1, ratio = [1., 0.])
    with torch.no_grad():
        i = 0
        for batch in train_dl:
            output = camembert(**batch, output_hidden_states = True)
            token_embeddings.append(output.hidden_states[-1][:,0,:].detach().numpy())
            if i % 100 == 0:
                print(i)
            i += 1
    print(len(token_embeddings))
    final_embeddings = [token_embeddings[i][0] for i in range(len(token_embeddings))]

    mat = [[[float(token_embeddings[i][j][k]) for i in range(0, len(token_embeddings))] for j in range(0, len(token_embeddings[0]))] for k in range(0, len(token_embeddings[0][0]))]
    # Serializing json
    with open("data.json", "w") as f:
        json.dump({"embedding" : mat, "domains":domains , "labels":labels}, f)
else:
    with open("data.json", "r") as f:
        data = json.load(f)
    token_embeddings = data["embedding"]
    final_embeddings = [token_embeddings[i][0] for i in range(len(token_embeddings))]
    cleaned_embeddings = [final_embeddings[0:1700][i] for i in range(len(final_embeddings[0:1700])) if labels[i] != 0]
    cleaned_domains = [domains[0:1700][i] for i in range(len(final_embeddings[0:1700])) if labels[i] != 0]
    cleaned_labels = [labels[0:1700][i] for i in range(len(final_embeddings[0:1700])) if labels[i] != 0]


def essai(domains = cleaned_domains, sentences = cleaned_embeddings):
    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(sentences,domains[0:len(sentences)], test_size=0.15, random_state=1) # 70% training and 30% test

    # Create Decision Tree classifer object
    clf = DecisionTreeClassifier(criterion="entropy")

    # Train Decision Tree Classifer
    clf = clf.fit(X_train,y_train)

    #Predict the response for test dataset
    y_pred = clf.predict(X_test)

    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


for i in range(20):
    essai()
