from model import *


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
    
    def train_model(self, batch_size, patience, max_epochs, test = True, ratio = [0.8, 0.1], wandb = True, save = False):
        
        return super().train_model('arg', batch_size, patience, max_epochs, test, ratio, wandb, save)

model = ArgModel('camembert-base', 5e-5, 0, False)
model.train_model(16, 10, 50, True, [0.8, 0.2], False, False)