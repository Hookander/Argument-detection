from model import *


class DomModel(Model):

    def __init__(self, model_name, lr, wd, from_scratch=False):

        self.num_labels = len(self.get_dico)
        self.model_name = model_name
        self.lr = lr
        self.weight_decay = wd
        self.typ = 'arg'
        super().__init__(model_name, self.num_labels, lr, wd, 'dom', from_scratch=False)


    def get_dico(self):
        return dom_dico
    
    def train_model(self, batch_size, patience, max_epochs, test = True, ratio = [0.8, 0.1], wandb = True, save = False):
        
        return super().train_model('dom', batch_size, patience, max_epochs, test, ratio, wandb, save)

model = DomModel('camembert-base', 5e-5, 0, False)
model.train_model(16, 10, 50, True, [0.8, 0.2], False, False)