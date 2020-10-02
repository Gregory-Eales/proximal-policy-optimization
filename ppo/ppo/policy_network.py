import torch
from tqdm import tqdm
import numpy as np

class PolicyNetwork(torch.nn.Module):

    def __init__(self, alpha, in_dim, out_dim, epsilon=0.1):

        super(PolicyNetwork, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.epsilon = epsilon
        self.define_network()
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')
        self.to(self.device)
        self.prev_params = self.parameters()

    def define_network(self):
        self.relu = torch.nn.ReLU()
        self.leaky_relu = torch.nn.LeakyReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.l1 = torch.nn.Linear(1024, 512)
        self.l2 = torch.nn.Linear(512, 64)
        self.l3 = torch.nn.Linear(64, self.out_dim)
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=4, stride=2)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=4, stride=2)
        self.conv4 = torch.nn.Conv2d(128, 256, kernel_size=4, stride=2)

    

    def normalize(self, x):
        x = np.array(x)
        x_mean = np.mean(x)
        x_std = np.std(x) if np.std(x) > 0 else 1
        x = (x-x_mean)/x_std
        return torch.Tensor(x)

    def loss(self, r_theta, advantages):

        clipped_r = torch.clamp(r_theta, 1.0 - self.epsilon, 1.0 + self.epsilon)
        return torch.min(r_theta*advantages, clipped_r*advantages)

    def forward(self, x):

        out = torch.Tensor(x).to(self.device)

        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.relu(out)
        out = self.conv4(out)
        out = self.relu(out)

        out = out.reshape(-1, 1024)

        
        out = self.l1(out)
        out = self.tanh(out)
        out = self.l2(out)
        out = self.tanh(out)
        out = self.l3(out)

        return out.to(torch.device('cpu:0'))

    def optimize(self, r, adv, iter=1):

        adv = self.normalize(adv)

        n_samples = r.shape[0]
        num_batch = int(n_samples/5)


        # calculate loss
        loss = self.loss(r, adv)

        l = []

        for batch in range(5):
            l.append(torch.sum(loss[batch*num_batch:(batch+1)*num_batch]))

        print("Training Policy Net:")
        for i in tqdm(range(iter)):

            for batch in range(5):


                torch.cuda.empty_cache()
                # zero the parameter gradients
                self.optimizer.zero_grad()

                # optimize
                l[batch].backward(retain_graph=True)

                self.optimizer.step()



def main():

    t1 = torch.ones(1, 3, 64, 64)
    pn = PolicyNetwork(0.01, 3, 1)
    print(pn(t1))
    print(pn.parameters())


if __name__ == "__main__":
    main()
