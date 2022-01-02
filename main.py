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
    scores_deque = deque(maxlen=100)
    scores = []
    for i_episode in range(1, n_episodes + 1):
        saved_log_probs = []
        rewards = []
        state = env.reset()
        for t in range(1, max_t+1):
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            state, reward, done, _ = env.step(action)
            rewards.append(reward)
            env.render()
            if done:
                break
        scores.append(sum(rewards))
        scores.append(sum(rewards))
        discount = [gamma**i for i in range(len(rewards))]
        R = sum([a * b for a, b in zip(discount, rewards)])
        policy_loss = []
        for log_prob in saved_log_probs:
            policy_loss.append(-log_prob * R)
        policy_loss = torch.cat(policy_loss).sum()
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        if i_episode % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(
                i_episode-100, np.mean(scores_deque)))
        avg_score = np.mean(scores_deque)
        if avg_score >= 195.0:
            print('Environment solved in {} episodes!'.format(i_episode-100))
            break

        return scores


scores = reinforce()
env = gym.make('CartPole-v0')

for _ in range(100):
    state = env.reset()
    while True:
        action, _ = policy.act(state)
        env.render()
        state, reward, done, _ = env.step(action)
        if done:
            break

env.close()
