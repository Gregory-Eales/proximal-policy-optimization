import torch
import numpy as np

class Buffer(object):

    def __init__(self):

        self.log_probs = []
        self.k_log_probs = []
        self.states = []
        self.prev_states = []
        self.rewards = []
        self.advantages = []
        self.firsts = []

    def store(self, state, reward, prev_state, first):
        self.store_state(state)
        self.store_reward(reward)
        self.store_firsts(first)
        self.store_prev_states(prev_state)

    def store_prev_states(self, prev_state):
        self.prev_states.append(prev_state)

    def store_state(self, state):
        self.states.append(state)

    def store_reward(self, reward):
        self.rewards.append(reward)

    def store_firsts(self, first):
        self.firsts.append(first)
        
    def store_log_probs(self, log_prob, k_log_prob):
        self.log_probs.append(log_prob)
        self.k_log_probs.append(k_log_prob)

    def store_advantage(self, advantage):
        self.advantages.append(advantage)

    def clear(self):
        self.log_probs = []
        self.k_log_probs = []
        self.states = []
        self.rewards = []
        self.advantages = []

    def get(self):

        states = torch.Tensor(self.states)
        log_probs = torch.cat(self.log_probs)
        k_log_probs = torch.cat(self.k_log_probs)
        rewards = torch.Tensor(self.rewards)
        advantages = torch.Tensor(self.advantage)

        return states, actions, k_actions, rewards, advantages
