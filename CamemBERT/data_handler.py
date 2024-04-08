import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import Dataset
import numpy as np


tokenizer = AutoTokenizer.from_pretrained('camembert-base')

def detokenize_sentences(sentences, tokenizer = tokenizer):
    return [tokenizer.decode(sent) for sent in sentences]

def tokenize_sentences(sentences, tokenizer = tokenizer):
    tokens = tokenizer(sentences, padding="longest", return_tensors="np")
    
    return tokens

def get_labels_from_ratio(labels, ratio = [0.7, 0.15]):
    """
    #! OUTDATED ?

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
    print(val_dict['labels'])
    return train_dict, val_dict, test_dict


def get_dataloaders(sentences, labels, batch_size = 16, ratio = [0.8, 0.1]):
    """_summary_

    Args:
        sentences (Dict): the tokenized sentences with the padding and the attention mask
        labels (np list): the labels
        batch_size (int, optional): _description_. Defaults to 16.
        ratio (list, optional): The repartition between the train/validation/test. Defaults to [0.7, 0.15]. 
                    The rest is for the test set.

    """
    size = len(sentences['input_ids'])
    
    train_size = int(ratio[0] * size)
    val_size = int(ratio[1] * size)
    
    train_dict = {key: sentences[key][:train_size] for key in sentences}
    train_dict['labels'] = labels[:train_size]
    
    val_dict = {key: sentences[key][train_size:train_size+val_size] for key in sentences}
    val_dict['labels'] = labels[train_size:train_size+val_size]
    
    test_dict = {key: sentences[key][train_size+val_size:] for key in sentences}
    test_dict['labels'] = labels[train_size+val_size:]
    

    train_dict, val_dict, test_dict = get_equal_distribution(sentences, labels, ratio)
    
    train_ds = Dataset.from_dict(train_dict)
    train_ds = train_ds.with_format("torch")
    
    val_ds = Dataset.from_dict(val_dict)
    val_ds = val_ds.with_format("torch")
    
    test_ds = Dataset.from_dict(test_dict)
    test_ds = test_ds.with_format("torch")
    
    train_dl = DataLoader(train_ds, batch_size = batch_size, shuffle = True)
    val_dl = DataLoader(val_ds, batch_size = batch_size, shuffle = False)
    test_dl = DataLoader(test_ds, batch_size = batch_size, shuffle = False)
    
    return train_dl, val_dl, test_dl


