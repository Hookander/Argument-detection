from model import *
import wandb


class ArgModel(Model):

    def __init__(self, model_name, lr, wd, from_scratch=False):

        self.num_labels = 3 #Nothing, Arg_fact, Arg_value
        self.model_name = model_name
        self.lr = lr
        self.weight_decay = wd
        self.typ = 'arg'
        super().__init__(model_name, self.num_labels, lr, wd, 'arg', from_scratch=False)


    def get_dico(self):
        return arg_dico
    
    def train_model(self, batch_size, patience, max_epochs, test = True, ratio = [0.8, 0.1], wandb = True, save = False, data_aug = True):
        
        return super().train_model('arg', batch_size, patience, max_epochs, test, ratio, wandb, save, data_aug)

sweep_config = {
    "method": "random",
    "name": "sweep",
    "metric": {"goal": "maximize", "name": "test/f1"},
    "parameters": {
        "batch_size": {"values": [4, 8, 16, 32]},
        "patience": {"values": [5, 10, 15, 30, 40]},
        "lr": {"max": 5e-4, "min": 3e-6},
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
        patience = config['patience']
        model_name = config['model_name']
        data_aug = config['data_aug']

        model = ArgModel(model_name, lr, weight_decay)
        return model.train_model(batch_size=batch_size, patience=patience, max_epochs=150, test=True, ratio=[0.75, 0.1], wandb = True, save = False, data_aug = data_aug)

    def main():
        
        sweep_id = wandb.sweep(sweep_config, project="camembert_arg")
        wandb.agent(sweep_id, function=get_test_f1, count=count)
    main()

#model = ArgModel('flaubert/flaubert_base_cased', 5e-5, 0, False)
#model.train_model(16, 10, 50, True, [0.75, 0.1], True, False)

#sweep(60)