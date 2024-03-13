
from transformers import AutoTokenizer, AutoConfig


tokenizer = AutoTokenizer.from_pretrained('camembert-base')


def tokenize_sentences(sentences, tokenizer = tokenizer):
    tokens = tokenizer(sentences, padding="longest", return_tensors="pt")
    return tokens

