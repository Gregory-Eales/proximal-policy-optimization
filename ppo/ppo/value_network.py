import torch
import numpy as np
from tqdm import tqdm

class ValueNetwork(torch.nn.Module):

    def __init__(self, alpha, input_dims, output_dims):

        self.input_dims = input_dims
        self.output_dims = output_dims

        # inherit from nn module class
        super(ValueNetwork, self).__init__()

        # initialize_network
        self.initialize_network()

        # define optimizer
        self.optimizer = torch.optim.Adam(lr=alpha, params=self.parameters())

        # define loss
        self.loss = torch.nn.MSELoss()

        # get device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')
        self.to(self.device)


    # initialize network
    def initialize_network(self):

		# define network components
        self.fc1 = torch.nn.Linear(self.input_dims, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, self.output_dims)
        self.relu = torch.nn.LeakyReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        x = torch.Tensor(x).to(self.device)
        out = self.fc1(x)
        out = self.tanh(out)
        out = self.fc2(out)
        out = self.tanh(out)
        out = self.fc3(out)
        out = self.relu(out)
        return out.to(torch.device('cpu:0'))

    def normalize(self, x):
        x = np.array(x)
        x_mean = np.mean(x)
        x_std = np.std(x) if np.std(x) > 0 else 1
        x = (x-x_mean)/x_std
        return x

    def optimize(self, observations, rewards, epochs=10):


        observations = self.normalize(observations.tolist())
        rewards = self.normalize(rewards.tolist())


        observations = torch.Tensor(observations.tolist())
        rewards = torch.Tensor(rewards.tolist())

        n_samples = rewards.shape[0]
        num_batch = int(n_samples/20)

        print("Training Value Net:")
        for i in tqdm(range(epochs)):

            for batch in range(20):


                torch.cuda.empty_cache()
                # zero the parameter gradients
                self.optimizer.zero_grad()

                # make prediction
                prediction = self.forward(observations[batch*num_batch:(batch+1)*num_batch])

                # calculate loss
                loss = self.loss(prediction, rewards[batch*num_batch:(batch+1)*num_batch])

                # optimize
                loss.backward(retain_graph=True)
                self.optimizer.step()



def main():

    t1 = torch.rand(100, 3)
    vn = ValueNetwork(0.01, 3, 1)
    vn.optimize(iter=100, state=t1, disc_reward=torch.rand(100, 1))



if __name__ == "__main__":
    main()
