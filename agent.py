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
from torch.utils.tensorboard import SummaryWriter
import time

writer = SummaryWriter()  #"runs/Mar03_14-55-58_DESKTOP-QGNSALL"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))

class DQN(nn.Module):
    def __init__(self, in_channels=1, num_actions=4):
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
    def __init__(self, useGPU=False):
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay = 30000
        self.gamma = 0.8
        self.learning_rate = 0.001
        self.batch_size = 64
        self.max_episodes = 10000
        self.save_interval = 25
        self.dqn = DQN()
        self.episode = -1
        self.useGPU = useGPU

        self.env = DroneEnv()

        # LOGGING
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
            f.close()
            f = open("log.txt", "r")
            txt = f.read()
            for i in range(len(txt)):
                if i == ' ':
                    self.eps_start = int(txt)

        else:
            if os.path.exists("log.txt"):
                open('log.txt', 'w').close()
            if os.path.exists("last_episode.txt"):
                open('last_episode.txt', 'w').close()

        if self.useGPU:
            self.dqn = self.dqn.to(device)  # to use GPU

        """Get observation"""
        responses = self.env.client.simGetImages(
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
        self.tensor = self.transformToTensor(im_final)

        self.memory = deque(maxlen=10000)
        self.optimizer = optim.Adam(self.dqn.parameters(), self.learning_rate)
        self.steps_done = 0

    def transformToTensor(self, img):
        if self.useGPU:
            tensor = torch.cuda.FloatTensor(img)
        else:
            tensor = torch.Tensor(img)
        tensor = tensor.unsqueeze(0)
        tensor = tensor.unsqueeze(0)
        tensor = tensor.float()
        return tensor

    def convert_size(self, size_bytes):
        if size_bytes == 0:
            return "0B"
        size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return "%s %s" % (s, size_name[i])

    def act(self, state):
        self.eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -1.0 * self.steps_done / self.eps_decay
        )
        self.steps_done += 1
        if random.random() > self.eps_threshold:
            print("greedy")
            if self.useGPU:
                action = np.argmax(self.dqn(state).cpu().data.squeeze().numpy())
                return torch.cuda.FloatTensor([action])
            else:
                data = self.dqn(state).data
                action = np.argmax(data.squeeze().numpy())
                return torch.LongTensor([action])
        else:
            action = [random.randrange(0, 4)]
            if self.useGPU:
                return torch.cuda.FloatTensor([action])
            else:
                return torch.LongTensor([action])

    def memorize(self, state, action, reward, next_state):
        self.memory.append(
            (
                state,
                action,
                torch.cuda.FloatTensor([reward]) if self.useGPU else torch.FloatTensor([reward]),
                self.transformToTensor(next_state),
            )
        )

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states = zip(*batch)

        states = torch.cat(states)
        #actions = torch.cat(actions)
        rewards = torch.cat(rewards)
        next_states = torch.cat(next_states)

        if self.useGPU:
            next_q_values = self.dqn(next_states).cpu().detach().numpy()
            actions = np.argmax(next_q_values, 1)
            max_next_q = torch.cuda.FloatTensor(next_q_values[[range(0, self.batch_size)], [actions]])
            current_q = torch.cuda.FloatTensor(self.dqn(states)[[range(0, self.batch_size)], [actions]])
            expected_q = rewards.to(device) + (self.gamma * max_next_q).to(device)
        else:
            next_q_values = self.dqn(next_states).detach().numpy()
            actions = np.argmax(next_q_values, 1)
            max_next_q = next_q_values[[range(0,self.batch_size)], [actions]]
            current_q = self.dqn(states)[[range(0,self.batch_size)], [actions]]
            expected_q = rewards + (self.gamma * max_next_q)

        loss = F.mse_loss(current_q.squeeze(), expected_q.squeeze())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self):

        score_history = []
        reward_history = []
        if self.episode == -1:
            self.episode = 1

        for e in range(1, self.max_episodes + 1):
            start = time.time()
            state = self.env.reset()
            steps = 0
            score = 0
            while True:
                state = self.transformToTensor(state)

                action = self.act(state)
                next_state, reward, done = self.env.step(action)

                self.memorize(state, action, reward, next_state)
                self.learn()

                state = next_state
                steps += 1
                score += reward
                if done:
                    print("episode:{0}, reward: {1}, mean reward: {2}, score: {3}, epsilon: {4}".format(self.episode, reward, round(score/steps, 2), score, self.eps_threshold))
                    print("----------------------------------------------------------------------------------------")
                    score_history.append(score)
                    reward_history.append(reward)
                    with open('log.txt', 'a') as file:
                        file.write("episode:{0}, reward: {1}, mean reward: {2}, score: {3}, epsilon: {4}\n".format(self.episode, reward, round(score/steps, 2), score, self.eps_threshold))

                    if self.useGPU:
                        print('Total Memory:', self.convert_size(torch.cuda.get_device_properties(0).total_memory))
                        print('Allocated Memory:', self.convert_size(torch.cuda.memory_allocated(0)))
                        print('Cached Memory:', self.convert_size(torch.cuda.memory_reserved(0)))
                        print('Free Memory:', self.convert_size(torch.cuda.get_device_properties(0).total_memory - (torch.cuda.max_memory_allocated() + torch.cuda.max_memory_reserved())))

                        # tensorboard --logdir=runs
                        memory_usage_allocated = np.float64(round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1))
                        memory_usage_cached = np.float64(round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1))

                        writer.add_scalar("memory_usage_allocated", memory_usage_allocated, e)
                        writer.add_scalar("memory_usage_cached", memory_usage_cached, e)

                        writer.add_scalar('epsilon_value', self.eps_threshold, e)
                        writer.add_scalar('score_history', score, e)
                        writer.add_scalar('reward_history', reward, e)
                        writer.add_scalars('General Look', {'epsilon_value': self.eps_threshold,
                                                            'score_history': score,
                                                            'reward_history': reward}, e)

                        writer.add_graph(self.dqn, self.tensor)

                    else:
                        writer.add_scalar('epsilon_value', self.eps_threshold, e)
                        writer.add_scalar('score_history', score, e)
                        writer.add_scalar('reward_history', reward, e)
                        writer.add_scalars('General Look', {'epsilon_value': self.eps_threshold,
                                                        'score_history': score,
                                                        'reward_history': reward}, e)

                        writer.add_graph(self.dqn, self.tensor)

                    break

            if self.episode % self.save_interval == 0:
                torch.save(self.dqn.state_dict(), self.model_dir + '//model_EPISODES_DQN_DRONE{}.pth'.format(self.episode))
                with open("last_episode.txt", "w") as file:
                    file.write(str(self.episode))
                with open('log.txt', 'a') as file:
                    file.write("**********************************************************\n"
                               "last {0} episodes: score mean: {1}, reward mean: {2}\n"
                               "**********************************************************\n"
                               .format(self.save_interval, round(sum(score_history[-self.save_interval:])/self.save_interval, 2), round(sum(reward_history[-self.save_interval:])/self.save_interval, 2)))

            self.episode += 1
            end = time.time()
            stopWatch = end - start
            print("Episode is done, episode time: ", stopWatch)
        writer.close()