import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque

env = gym.make('CartPole-v0')
print(env.observation_space)
print(env.action_space)
# while True:
#     env.reset()
#     env.render()
#     action = env.action_space.sample()
#     env.step(action)


class Policy(nn.Module):
    def __init__(self, s_size=4, h_size=16, a_size=2):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, a_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(state)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)


policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=0.01)


def reinforce(n_episodes=1000, max_t=1000, gamma=1.0, print_every=100):
    sources_deque = deque(maxlen=100)
    scores = []
