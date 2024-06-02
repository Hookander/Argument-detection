from model import *
import wandb


class DomModel(Model):

    def __init__(self, model_name, lr, wd, from_scratch=False):

        self.num_labels = len(self.get_dico())
        print(self.num_labels)
        self.model_name = model_name
        self.lr = lr
        self.weight_decay = wd
        self.typ = 'dom'
        super().__init__(model_name, self.num_labels, lr, wd, 'dom', from_scratch=False)


    def get_dico(self):
        return domain_dico
    
    def train_model(self, batch_size, max_epochs, test = True, wandb = True, save = False, data_aug = True):
        
        super().train_model('dom', batch_size, max_epochs, test, wandb, save, data_aug)

        if save:
            self.model.save_pretrained(f"./CamemBERT/models/dom/dom_model_large")
        

sweep_config = {
    "method": "random",
    "name": "sweep",
    "metric": {"goal": "maximize", "name": "test/f1"},
    "parameters": {
        "batch_size": {"values": [4, 8, 16]},
        "patience": {"values": [5, 10, 15, 30, 40]},
        "lr": {"max": 1e-3, "min": 1e-6},
        "weight_decay": {"max": 0.2, "min": 0.},
        "model_name": {"values": ["camembert-base"]},
        "data_aug": {"values": [True, False]}
    },
}

def sweep(count):
    wandb.login()

    def get_test_f1():
        wandb.init(project="camembert_arg")
        config = wandb.config
        lr = config['lr']
        batch_size = config['batch_size']
        weight_decay = config['weight_decay']
        model_name = config['model_name']
        data_aug = config['data_aug']

        model = DomModel(model_name, lr, weight_decay)
        return model.train_model(batch_size=batch_size, patience=patience, max_epochs=150, test=True, ratio=[0.75, 0.1], wandb = True, save = False, data_aug = data_aug)

    def main():
        
        sweep_id = wandb.sweep(sweep_config, project="camembert_dom")
        wandb.agent(sweep_id, function=get_test_f1, count=count)
    main()

m = DomModel("camembert/camembert-large", 5e-6, 0)
m.train_model(8, 80, test=True, wandb=True, save = True, data_aug = True)


#sweep(60)