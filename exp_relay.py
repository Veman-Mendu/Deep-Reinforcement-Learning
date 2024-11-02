import gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import numpy as np

env = gym.make('CartPole-v0', render_mode="human")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(env.observation_space.shape[0], 64)
        self.fc2 = nn.Linear(64, env.action_space.n)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        
        return x
    
model = DQN().to(device)

def epsilon_greedy(state, epsilon):
    r = random.random()

    if r > epsilon:
        action = env.action_space.sample()
        #print(f'Random action {action}')
        return action
    else:
        #print(f'state is - {state}')
        with torch.no_grad():
            state = torch.Tensor(state).to(device)
            qvalues = model(state)
            print(f'qvalues are - {qvalues}')
            maxq, action = torch.max(qvalues, dim=0)
            action = action.item()
        #print(f'Algo action {action}')
        return action

episodes = 1000
discount_factor = 0.99

total_rewards = []

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for episode in range(episodes):
    print(f'Episode number - {episode}')
    state = env.reset()
    state = state[0]
    done = False
    rewards = 0

    exp_relay = []

    while done == False:
        env.render()

        epsilon = episode/episodes
        action = epsilon_greedy(state, epsilon)

        new_state, reward, done, info, _ = env.step(action)
        rewards += reward
        #result = env.step(action)
        #print(result)
        exp_relay.append([state, action, new_state, reward, done])
        state = new_state

        if done:
            model.train()

            exp_relay = np.stack(exp_relay)

            exp_newstates = exp_relay[:,2]
            exp_newstates = np.stack(exp_newstates)
            exp_newstates = torch.Tensor(exp_newstates).to(device)

            q_values = model(exp_newstates)
            maxq, index = torch.max(q_values, dim=-1)

            exp_rewards = exp_relay[:,3]
            exp_rewards = np.stack(exp_rewards)
            exp_rewards = torch.Tensor(exp_rewards).to(device)

            exp_dones = exp_relay[:,4]
            exp_dones = np.stack(exp_dones)
            exp_dones = torch.Tensor(exp_dones).to(device)

            target = exp_rewards + (discount_factor * maxq * (1-exp_dones))

            exp_states = exp_relay[:,0]
            exp_states = np.stack(exp_states)
            exp_states = torch.Tensor(exp_states).to(device)

            pred = model(exp_states)

            exp_actions = exp_relay[:,1]
            exp_actions = np.stack(exp_actions)
            exp_actions = torch.Tensor(exp_actions).to(device)

            pred = pred[torch.arange(exp_actions.size(0)), exp_actions.squeeze().long()]

            loss = criterion(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    total_rewards.append(rewards)
    print(f'Total Rewards - {rewards}')

plt.plot(range(episodes), total_rewards, marker='o')
plt.title('Rewards Collected for episode data training')
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.show()