import gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt

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

    while done == False:
        env.render()

        epsilon = episode/episodes
        action = epsilon_greedy(state, epsilon)

        new_state, reward, done, info, _ = env.step(action)
        rewards += reward
        #esult = env.step(action)
        #print(result)

        model.train()

        new_state_tensor = torch.Tensor(new_state).to(device)
        newq = model(new_state_tensor).detach()
        #print(newq)
        maxq = torch.max(newq)

        reward = torch.Tensor([reward]).to(device)

        if done:
            target = reward
        else:
            target = reward + (discount_factor * maxq)

        #print(state)
        state = torch.Tensor(state).to(device)
        pred = model(state)
        print(f'pred is - {pred[action].view(-1)} and maxpred is - {torch.max(pred)}')
        pred = pred[action].view(-1)
        
        #print(f'prediction - {maxpred}, target - {target}')
        loss = criterion(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = new_state
        #print(done)

        if done:
            total_rewards.append(rewards)
            print(f'Total Reward - {rewards}')
            break

plt.plot(range(episodes), total_rewards, marker='o')
plt.title('Rewards Collected')
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.show()