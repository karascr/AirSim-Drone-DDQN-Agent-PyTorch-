#!/usr/bin/env python
"""Environment for Microsoft AirSim Unity Quadrotor

- Author: Subin Yang
- Contact: subinlab.yang@gmail.com
"""
import math
import random
from collections import deque
import airsim
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from setuptools import glob

from env import DroneEnv

#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

env = DroneEnv()

class DQN(nn.Module):
    def __init__(self, in_channels=1, num_actions=7):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 84, kernel_size=4, stride=4)
        self.conv2 = nn.Conv2d(84, 42, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(42, 21, kernel_size=2, stride=2)
        self.fc4 = nn.Linear(21*4*4, 168)
        self.fc5 = nn.Linear(168, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc4(x))
        return self.fc5(x)


class Agent:
    def __init__(self):
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay = 200
        self.gamma = 0.8
        self.learning_rate = 0.001
        self.batch_size = 1
        self.max_episodes = 10000
        self.save_interval = 50

        self.dqn = DQN()
        self.episode = -1

        cwd = os.getcwd()
        self.model_dir = os.path.join(cwd, "saved models")

        if not os.path.exists(self.model_dir):
            os.mkdir("saved models")


        files = glob.glob(self.model_dir + '\\*.pth')
        if len(files) > 0:
            file = files[-1]
            self.dqn.load_state_dict(torch.load(file))
            self.dqn.eval()

            f = open("last_episode.txt", "r")
            self.episode = int(f.read())
            print("checkpoint loaded: ", file, "last episode was: ", self.episode)

        #self.dqn = self.dqn.to(device) # to use GPU

        """Get observation"""
        responses = env.client.simGetImages(
            [airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)]
        )

        """Transform input binary array to image"""
        response = responses[0]
        img1d = np.fromstring(
            response.image_data_uint8, dtype=np.uint8
        )
        img_rgba = img1d.reshape(
            response.height, response.width, 3
        )
        image = Image.fromarray(img_rgba)
        im_final = np.array(image.resize((84, 84)).convert("L"))

        """oimg = image = Image.fromarray(im_final)
        oimg.save("a.jpg")"""

        #tensor = torch.from_numpy(im_final)
        tensor = self.transformToTensor(im_final)

        self.model = self.dqn.forward(tensor)
        self.memory = deque(maxlen=10000)
        self.optimizer = optim.Adam(self.dqn.parameters(), self.learning_rate)
        self.steps_done = 0

    def transformToTensor(self, img):
        tensor = torch.Tensor(img)
        tensor = tensor.unsqueeze(0)
        tensor = tensor.unsqueeze(0)
        tensor = tensor.float()
        return tensor

    def act(self, state):
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -1.0 * self.steps_done / self.eps_decay
        )
        self.steps_done += 1
        if random.random() > eps_threshold:
            data = self.dqn(state).data
            action = np.argmax(data.squeeze().numpy())

            return torch.LongTensor([action])
        else:
            action = [random.randrange(0, 7)]
            return torch.LongTensor([action])

    def memorize(self, state, action, reward, next_state):
        self.memory.append(
            (
                state,
                action,
                torch.FloatTensor([reward]),
                self.transformToTensor(next_state),
            )
        )

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states = zip(*batch)

        states = torch.cat(states)
        actions = torch.cat(actions)
        rewards = torch.cat(rewards)
        next_states = torch.cat(next_states)

        current_q = self.dqn(states)
        max_next_q = self.dqn(next_states).detach().max(1)[0]
        expected_q = rewards + (self.gamma * max_next_q)

        """print(current_q)
        print("current:", current_q.shape)
        print("current s:", current_q.squeeze().shape)
        print("expected:", expected_q.shape)"""

        loss = F.mse_loss(current_q.squeeze(), expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self):

        score_history = []
        reward_history = []
        score = 0
        if self.episode == -1:
            self.episode = 1

        for e in range(1, self.max_episodes + 1):
            state = env.reset()
            steps = 0
            while True:
                #state = torch.FloatTensor([state])
                state = self.transformToTensor(state)

                action = self.act(state)
                next_state, reward, done = env.step(action)

                self.memorize(state, action, reward, next_state)
                self.learn()

                state = next_state
                steps += 1
                score += reward
                if done:
                    print("episode:{0}, reward: {1}, score: {2}".format(self.episode, reward, score))
                    print("----------------------------------------------------")
                    score_history.append(steps)
                    reward_history.append(reward)
                    f = open("log.txt", "a")
                    f.write("episode:{0}, reward: {1}, score: {2}".format(e, reward, score))
                    f.close()
                    break

            if self.episode % self.save_interval == 0:
                torch.save(self.dqn.state_dict(), self.model_dir + '//model_EPISODES_DQN_DRONE{}.pth'.format(self.episode))
                f = open("last_episode.txt", "w")
                f.write(str(self.episode))
                f.close()
            self.episode += 1