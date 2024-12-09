import gymnasium as gym
import ale_py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import cv2
import random
import copy
from plotter import update_plot

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

gym.register_envs(ale_py)

FramesToSkip = 4
image_shape = (84, 84)

def process_observation(frame):
    frame = cv2.resize(frame, image_shape)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = frame/255.0
    frame = torch.Tensor(frame)
    frame = frame.unsqueeze(0)
    frame = frame.unsqueeze(0)

    return frame

def step(action, lives):
    total_reward = 0
    done = False

    for i in range(FramesToSkip):
        observation, reward, done, truncated, info = env.step(action)
        total_reward += reward
        frame_buffer.append(observation)

        if info['lives'] < lives:
            total_reward -= 1
            lives = info['lives']

        if done:
            break

    max_frame = np.max(frame_buffer[-2:], axis=0)
    max_frame = process_observation(max_frame)

    total_reward = torch.tensor(total_reward).view(1, -1).float()
    done = torch.tensor(done).view(1, -1).float()

    return max_frame, total_reward, done, info

env = gym.make('BreakoutNoFrameskip-v4', render_mode='human')

class AtariNet(nn.Module):
    def __init__(self):
        super(AtariNet, self).__init__()

        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=(8,8), stride=(4,4))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(4,4), stride=(2,2))
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3,3), stride=(1,1))

        self.flatten = nn.Flatten()

        self.dropout = nn.Dropout(p=0.2)

        self.action_value1 = nn.Linear(3136, 1024)
        self.action_value2 = nn.Linear(1024, 1024)
        self.action_value3 = nn.Linear(1024, 4)

        self.state_value1 = nn.Linear(3136, 1024)
        self.state_value2 = nn.Linear(1024, 1024)
        self.state_value3 = nn.Linear(1024, 1)

    def forward(self, x):
        x = torch.Tensor(x)

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.flatten(x)

        state_value = self.dropout(self.relu(self.state_value1(x)))
        state_value = self.dropout(self.relu(self.state_value2(state_value)))
        state_value = self.relu(self.state_value3(state_value))

        action_value = self.dropout(self.relu(self.action_value1(x)))
        action_value = self.dropout(self.relu(self.action_value2(action_value)))
        action_value = self.action_value3(action_value)

        output = state_value + (action_value - action_value.mean())

        return output

model = AtariNet().to(device)
target_model = copy.deepcopy(model).eval().to(device)

learning_rate = 0.00001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def save_the_model(weights_filename='models/latest.pt', model=model):
    torch.save(model.state_dict(), weights_filename)

def load_the_model(weights_filename='models/latest.pt', model=model):
    try:
        model.load_state_dict(torch.load(weights_filename))
        print(f'Successfully loaded weights from {weights_filename}')
    except:
        print(f'Failed to load weights from {weights_filename}')

capacity = 1000000
memory = []
batchsize = 64

def insert(transition, memory, capacity):
    transition = [item.to('cpu') for item in transition]

    if len(memory) < capacity:
        memory.append(transition)
    else:
        memory.remove(memory[0])
        memory.append(transition)
    
    return memory

def sample_batch(memory, batchsize):
    if len(memory) > batchsize * 10:
        batch = random.sample(memory, batchsize)
        batch = zip(*batch)
        return [torch.cat(items).to(device) for items in batch]
    else:
        return None

def get_action(state):
    if torch.rand(1) < epsilon:
        return torch.randint(env.action_space.n, (1, 1))
    else:
        state = state.to(device)
        qvalues = model(state).detach()
        return torch.argmax(qvalues, dim=1, keepdim=True)

episodes = 200000

epsilon = 1.0
min_epsilon = 0.1
warmup_epsilon = 5000

gamma = 0.99
scores = []
avg_scores = []
epsilon_record = []

for episode in range(episodes):
    state, _ = env.reset()
    frame_buffer = []
    lives = 5
    state = process_observation(state)
    done = False
    game_scores = 0

    while done == False:
        #action = env.action_space.sample()
        action = get_action(state)
        newstate, reward, done, info = step(action, lives)
        lives = info['lives']

        memory = insert([state, action, reward, newstate, done], memory, capacity)

        if len(memory) > batchsize * 10:
            state_batch, action_batch, reward_batch, newstate_batch, done_batch = sample_batch(memory, batchsize)
            
            q_state = model(state_batch).gather(1, action_batch)

            q_newstate = target_model(newstate_batch)
            q_newstate = torch.max(q_newstate, dim=-1, keepdim=True)[0]

            target = reward_batch + (gamma * q_newstate * (1 - done_batch))

            loss = F.mse_loss(q_state, target)
            model.zero_grad()
            loss.backward()
            optimizer.step()

        state = newstate
        game_scores += reward.item()

    scores.append(game_scores)
    print(f'Score for game-{episode} is - {game_scores}')
    epsilon = epsilon * (1 - (((epsilon - min_epsilon)/warmup_epsilon) * 2))

    if episode % 10 == 0:
        save_the_model(model=model)
        if len(scores) > 100:
            print(f'Average for last 100 games is {np.mean(scores[-100:])}')
            avg_scores.append(np.mean(scores[-100:]))
            epsilon_record.append(epsilon)
            update_plot(avg_scores, epsilon_record, 'TargetPlot')
        else:
            print(f'Average for last {len(scores)} games is {np.mean(scores)}')
            avg_scores.append(np.mean(scores))
            epsilon_record.append(epsilon)
            update_plot(avg_scores, epsilon_record, 'TargetPlot')

    if episode % 100 == 0:
        target_model.load_state_dict(model.state_dict())