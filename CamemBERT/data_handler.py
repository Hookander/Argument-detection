import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import Dataset
import numpy as np
import sys
sys.path.append('./docs/csv') # not clean but ok for now
from csv_handler import *


tokenizer = AutoTokenizer.from_pretrained('camembert-base')

def detokenize_sentences(sentences, tokenizer = tokenizer):
    return [tokenizer.decode(sent) for sent in sentences]

def tokenize_sentences(sentences, tokenizer = tokenizer):
    tokens = tokenizer(sentences, padding="longest", return_tensors="np")
    
    return tokens

def get_labels_from_ratio(labels, ratio = [0.7, 0.15]):
    """
     -- outdated -- 
    Args:
        labels (np list): the labels
        ratio (list, optional): The repartition between the train/validation/test. Defaults to [0.7, 0.15]. 
                    The rest is for the test set.
    """
    size = len(labels)
    
    train_size = int(ratio[0] * size)
    val_size = int(ratio[1] * size)
    
    return labels[:train_size], labels[train_size:train_size+val_size], labels[train_size+val_size:]

def get_equal_distribution(sentences, labels, ratio = [0.8, 0.1]):
    """
    sentence : dictionary with the tokenized sentences and the attention masks
    """
    split_labels_indices = {label : [] for label in set(labels)}
    for i, label in enumerate(labels):
        split_labels_indices[label].append(i)
    
    # for each label, we split the indices into the train/validation/test set to obtain the
    # same distribution in each dataset
    train_indices, val_indices, test_indices = [], [], []

    for label in split_labels_indices:
        size = len(split_labels_indices[label])
        train_size = int(ratio[0] * size)
        val_size = int(ratio[1] * size)
        
        train_indices += split_labels_indices[label][:train_size]
        val_indices += split_labels_indices[label][train_size:train_size+val_size]
        test_indices += split_labels_indices[label][train_size+val_size:]


    train_dict = {key: [sentences[key][i] for i in train_indices] for key in sentences}
    train_dict['labels'] = [labels[i] for i in range(len(labels)) if i in train_indices]

    val_dict = {key: [sentences[key][i] for i in val_indices] for key in sentences}
    val_dict['labels'] = [labels[i] for i in range(len(labels)) if i in val_indices]

    test_dict = {key: [sentences[key][i] for i in test_indices] for key in sentences}
    test_dict['labels'] = [labels[i] for i in range(len(labels)) if i in test_indices]

    return train_dict, val_dict, test_dict

def get_labels(typ, sentences, arg_types, domains):

    """
        -- outdated -- 
        donne les bons labels en enlevant les phrases non argument si on veut les domaines
        (car dans ce cas on n'essaie pas de les trouver)
    """
    if typ == 'arg':
        labels = arg_types
    elif typ == 'dom':
        indices = [i for i, domain in enumerate(domains) if domain != 0]

        sentences = {key: [sentences[key][i] for i in indices] for key in sentences}
        labels = [domain for i, domain in enumerate(domains) if i in indices]

    else:
        print("Invalid type")
        return

    return labels

def get_dataloaders(typ, use_data_aug = True, batch_size = 16):
    """

    Renvoie les dataloaders d'entrainement et de test en foncion du type, et de si on veut 
    utiliser la data-augmentation

    """
    sentences_train, labels_train, sentences_test, labels_test = get_train_test(typ, use_data_aug = use_data_aug)
    
    sentences_train = tokenize_sentences(sentences_train)
    sentences_test = tokenize_sentences(sentences_test)
    
    train_dict = {key: sentences_train[key] for key in sentences_train}
    train_dict['labels'] = labels_train

    test_dict = {key: sentences_test[key] for key in sentences_test}
    test_dict['labels'] = labels_test

    
    train_ds = Dataset.from_dict(train_dict)
    train_ds = train_ds.with_format("torch")
    
    test_ds = Dataset.from_dict(test_dict)
    test_ds = test_ds.with_format("torch")
    
    train_dl = DataLoader(train_ds, batch_size = batch_size, shuffle = True)
    test_dl = DataLoader(test_ds, batch_size = batch_size, shuffle = False)
    
    return train_dl, test_dl

