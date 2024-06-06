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
        return arg_dico_rev
    
    def train_model(self, batch_size, max_epochs, test = True, wandb = True, save = False, data_aug = True):
        
        super().train_model('arg', batch_size, max_epochs, test, wandb, save, data_aug)
        if save:
            self.model.save_pretrained(f"./CamemBERT/models/arg/arg_model")

sweep_config = {
    "method": "random",
    "name": "sweep",
    "metric": {"goal": "maximize", "name": "test/f1"},
    "parameters": {
        "batch_size": {"values": [4, 8, 16, 32]},
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
        model_name = config['model_name']
        data_aug = config['data_aug']

        model = ArgModel(model_name, lr, weight_decay)
        return model.train_model(batch_size=batch_size, max_epochs=100, test=True, wandb = True, data_aug = data_aug)


    def main():
        
        sweep_id = wandb.sweep(sweep_config, project="camembert_arg")
        wandb.agent(sweep_id, function=get_test_f1, count=count)
    main()

#model = ArgModel('camembert/camembert-large', 5e-7, 0, False)
#model.train_model(8, 70, test = True, wandb = True, save = True, data_aug = True)

#sweep(60)