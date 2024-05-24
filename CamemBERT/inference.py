from model import *
from arg_model import ArgModel
from dom_model import DomModel
import lightning
"""

[s1, s2, ...]

--> [(0, 0), (1, 2), ...]

"""
def arg_inference(sentences, path = "./CamemBERT/models/arg/arg_model"):

    sentences = tokenize_sentences(sentences)
    dataset = Dataset.from_dict(sentences).with_format("torch")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    print("done dl")
    device = torch.device("cpu")
    model = ArgModel(path, 5e-6, 0, True)
    
    trainer = model.get_trainer(save=False, max_epochs=1, patience=1, wandb=False)
    
    with torch.no_grad():
        predictions = trainer.predict(model, dataloaders=dataloader)
    print(predictions)
    preds = torch.concatenate(predictions).tolist()
    print(preds)

def dom_inference(sentences, path = "./CamemBERT/models/dom/dom_model"):
    sentences = tokenize_sentences(sentences)
    dataset = Dataset.from_dict(sentences).with_format("torch")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    print("done dl")
    device = torch.device("cpu")
    model = DomModel(path, 5e-6, 0, True)
    
    trainer = model.get_trainer(save=False, max_epochs=1, patience=1, wandb=False)
    
    with torch.no_grad():
        predictions = trainer.predict(model, dataloaders=dataloader)
    print(predictions)
    preds = torch.concatenate(predictions).tolist()
    print(preds)

def inference(sentences):
    arg = arg_inference(sentences)

    # We only check the domains if the argument is not 0
    # So we need to filter the sentences and store the indexes
    filtered_sentences = []
    indexes = []
    for i in range(len(arg)):
        if arg[i] != 0:
            filtered_sentences.append(sentences[i])
            indexes.append(i)
    dom = dom_inference(filtered_sentences)
    dom_preds = [0 for i in range(len(sentences))]
    for i in range(len(indexes)):
        dom_preds[indexes[i]] = dom[i]
    
    output = [(arg[i], dom_preds[i]) for i in range(len(arg))]

    #! need to convert to strings 

    return output
    

arg_inference(['Je pense que oui car il rendrait le projet plus acceptable', 'Je suis pas d\'accord avec toi'])