import sys
sys.path.append('./CamemBERT/')
from model import *
from arg_model import ArgModel
from dom_model import DomModel
from data_handler import *


def arg_inference(sentences, path = "./models/arg/arg_model"):

    # Rend les prédictions pour les arguments

    sentences = tokenize_sentences(sentences)
    dataset = Dataset.from_dict(sentences).with_format("torch")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    device = torch.device("cpu")
    model = ArgModel(path, 5e-6, 0, True)
    
    trainer = pl.Trainer(logger = False)
    
    with torch.no_grad():
        predictions = trainer.predict(model, dataloaders=dataloader)
    

    preds = torch.concatenate(predictions).tolist()
    
    return preds, model.get_dico()

def dom_inference(sentences, path = "./models/dom/dom_model_base3"):

    # Rend les prédictions pour les domaines

    sentences = tokenize_sentences(sentences)
    dataset = Dataset.from_dict(sentences).with_format("torch")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    device = torch.device("cpu")
    model = DomModel(path, 5e-6, 0, True)
    
    trainer = pl.Trainer(logger = False)
    
    with torch.no_grad():
        predictions = trainer.predict(model, dataloaders=dataloader)

    preds = torch.concatenate(predictions).tolist()
    
    return preds, model.get_dico()

def inference(sentences):

    # Rend les prédictions pour les arguments et les domaines

    arg, arg_dico = arg_inference(sentences)

    # We only check the domains if the argument is not 0
    # So we need to filter the sentences and store the indexes
    filtered_sentences = []
    indexes = []
    for i in range(len(arg)):
        if arg[i] != 0:
            filtered_sentences.append(sentences[i])
            indexes.append(i)
    dom, groups_dom_rev = dom_inference(filtered_sentences)
    dom_preds = [0 for i in range(len(sentences))]
    for i in range(len(indexes)):
        dom_preds[indexes[i]] = dom[i]

    
    # Re convert the indexes to the original arg/domain names
    output = [(arg[i], dom_preds[i]) for i in range(len(arg))]
    output = [(arg_dico[arg], dom) for arg, dom in output]
    output = [(arg, groups_dom_rev[dom]) for arg, dom in output]

    return output
    

#print(inference(['Oui car il rendrait le projet plus acceptable écologiquement', 'Je suis pas d\'accord avec toi']))