import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import Dataset


tokenizer = AutoTokenizer.from_pretrained('camembert-base')

def tokenize_sentences(sentences, tokenizer = tokenizer):
    tokens = tokenizer(sentences, padding="longest", return_tensors="np")
    
    return tokens

def get_labels_from_ratio(labels, ratio = [0.7, 0.15]):
    
    size = len(labels)
    
    train_size = int(ratio[0] * size)
    val_size = int(ratio[1] * size)
    
    return labels[:train_size], labels[train_size:train_size+val_size], labels[train_size+val_size:]

def get_dataloaders(sentences, labels, batch_size = 16, ratio = [0.7, 0.15]):
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
    
    train_ds = Dataset.from_dict(train_dict)
    train_ds = train_ds.with_format("torch")
    
    val_ds = Dataset.from_dict(val_dict)
    val_ds = val_ds.with_format("torch")
    
    test_ds = Dataset.from_dict(test_dict)
    test_ds = test_ds.with_format("torch")
    
    train_dl = DataLoader(train_ds, batch_size = batch_size, shuffle = True)
    val_dl = DataLoader(val_ds, batch_size = batch_size, shuffle = True)
    test_dl = DataLoader(test_ds, batch_size = batch_size, shuffle = True)
    
    return train_dl, val_dl, test_dl



