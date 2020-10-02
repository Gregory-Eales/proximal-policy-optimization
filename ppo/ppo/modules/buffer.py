import torch
from torch.utils.data import Dataset


class Buffer(Dataset):

    def __init__(self, hparams, x, y):

        self.hparams = hparams

        super(Buffer, self).__init__()

        self.states = []
        self.rewards = []
        self.actions = []
        self.advantages = []
        self.pi = []
        self.old_pi = []

    def __len__(self):
        return self.states.shape[0]
    
    def __getitem__(self, index):
        return self.states[index],
        self.actions[index],
        self.rewards[index],
    
    def __iter__(self):
        num_batch = self.x.shape[0]//self.hparams.batch_size
        rem_batch = self.x.shape[0]%self.hparams.batch_size
        
        for i in range(num_batch):
            i1, i2 = i*self.hparams.batch_size, (i+1)*self.hparams.batch_size
            yield self.x[i1:i2], self.y[i1:i2]
        
        
        i1 = -rem_batch
        i2 = 0
        yield self.x[i1:i2], self.y[i1:i2]



	