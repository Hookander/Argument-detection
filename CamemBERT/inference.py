from model import *
from arg_model import ArgModel
from dom_model import DomModel


def arg_inference(path, sentences):

    sentences = tokenize_sentences(sentences)
    dataset = Dataset.from_dict(sentences)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    model = ArgModel('camembert-base', 5e-6, 0, False)
    model.load_state_dict(torch.load(path))
    trainer = model.get_trainer2(50, False)
    
    with torch.no_grad():
        predictions = trainer.predict(model, dataloaders=dataloader)
    print(predictions)

arg_inference('camembert_arg/51dbsetx/checkpoints/epoch=49-step=47100.ckpt', ['Je suis d\'accord avec toi', 'Je suis pas d\'accord avec toi'])