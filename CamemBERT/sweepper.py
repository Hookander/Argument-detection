from models import *
import wandb


wandb.login()



def get_test_f1():
    wandb.init(project="camembert_arg")
    config = wandb.config
    lr = config['lr']
    batch_size = config['batch_size']
    weight_decay = config['weight_decay']
    patience = config['patience']
    model_name = config['model_name']

    model = Model(model_name, 3, lr=lr, weight_decay=weight_decay, typ="arg")
    model.train_model(batch_size=batch_size, patience=patience, max_epochs=100, test=True, ratio=[0.8, 0.2], wandb = True)

config = {
    "method": "random",
    "name": "sweep",
    "metric": {"goal": "maximize", "name": "valid/f1"},
    "parameters": {
        "batch_size": {"values": [8, 16, 32, 64]},
        "patience": {"values": [5, 10, 15, 30]},
        "lr": {"max": 1e-3, "min": 1e-5},
        "weight_decay": {"max": 0.1, "min": 0.},
        "model_name": {"values": ["camembert-base", "camembert/camembert-large"]},
    },
}

def main():
    
    sweep_id = wandb.sweep(config, project="camembert_arg")
    wandb.agent(sweep_id, function=get_test_f1, count=50)
main()