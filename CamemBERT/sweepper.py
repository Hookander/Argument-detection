from CamemBERT.models import *
import wandb


wandb.login()

def get_test_acc(config):
    lr = config['lr']
    batch_size = config['batch_size']
    weight_decay = config['weight_decay']

    model = ArgDetector("camembert-base", 3, lr=lr, weight_decay=weight_decay)
    model.train_model(batch_size=batch_size, patience=10, max_epochs=50, test=True, ratio=[0.7, 0.15], wandb = False)

def main():
    wandb.init(project="camembert")

