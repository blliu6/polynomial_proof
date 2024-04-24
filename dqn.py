import sys

sys.path.append('/home/blliu/pythonproject/polynomial_proof')

import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from Env import Env


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出

    def add(self, state, action, reward, next_state, done):  # 将数据加入buffer
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), np.array(action), reward, next_state, done

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)


class Qnet(nn.Module):
    def __init__(self, s_dim, dense=4, units=128):
        super().__init__()
        self.seq = nn.Sequential()
        s = s_dim
        for i in range(dense):
            self.seq.add_module(f'dense{i}', nn.Linear(s, units))
            self.seq.add_module(f'relu{i}', nn.ReLU())
            s = units
        self.seq.add_module(f'dense_{dense}', nn.Linear(units, 1))

    def forward(self, x):
        return self.seq(x)


class DQN:
    def __init__(self, state_dim, units, action_dim, learning_rate, gamma, epsilon, target_update, device, name,
                 load=False):
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim * 3, dense=4, units=units).to(device)
        self.target_q_net = Qnet(state_dim * 3, dense=4, units=units).to(device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        # self.optimizer = torch.optim.RMSprop(self.q_net.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.count = 0
        self.device = device
        self.steps = 1e6
        self.name = name

        if load:
            self.q_net.load_state_dict(torch.load(f'./model/{self.name}.pth'))
            print('Parameters loaded successfully!')

    def take_action(self, state, action):
        if np.random.random() > self.epsilon:
            action = np.random.randint(len(action))
        else:
            input = np.concatenate((np.array([state] * len(action)), action), axis=1)
            state = torch.tensor(input, dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action

    def get_next_q(self, next_state, action_map):
        res = torch.empty((len(next_state), 1)).to(self.device)
        for i, state in enumerate(next_state):
            action = action_map[tuple(state)]
            input = np.concatenate((np.array([state] * len(action)), action), axis=1)
            state = torch.tensor(input, dtype=torch.float).to(self.device)
            res[i] = self.q_net(state).max()
        return res

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = transition_dict['next_states']
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        input = torch.cat((states, actions), dim=1)
        q_values = self.q_net(input)

        max_next_q_values = self.get_next_q(next_states, env.map)

        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)  # TD误差目标
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())  # 更新目标网络
        self.count += 1

    def save(self):
        torch.save(self.q_net.state_dict(), f'./model/{self.name}.pth')
        print('Model saved successfully!')


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []
    min_episode = 0
    for i_episode in range(num_episodes):
        if i_episode % 10 == 0:
            agent.epsilon = min(agent.epsilon + 0.01, 1)
        episode_return = 0
        state, info = env.reset()
        done, truncated = False, False
        while not done and not truncated:
            action = agent.take_action(state, env.action)
            next_state, reward, done, truncated, info = env.step(action)
            action = env.action[action]

            if done and agent.steps > info and agent.epsilon == 1:
                agent.steps, min_episode = info, i_episode
                agent.save()

            if reward > 0:
                for i in range(9):
                    replay_buffer.add(state, action, reward, next_state, done)

            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            episode_return += reward
            if replay_buffer.size() > minimal_size:
                b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r,
                                   'dones': b_d}
                agent.update(transition_dict)
        return_list.append(episode_return)
        print(f'Sum of reward: {episode_return}')
    print(f'Minimum number of proof steps:{agent.steps}, Minimum episode:{min_episode}')
    return return_list


if __name__ == '__main__':
    random.seed(2024)
    np.random.seed(2024)
    torch.manual_seed(2024)

    lr = 1e-5
    num_episodes = 300
    units = 128
    gamma = 0.99
    epsilon = 0.9
    target_update = 20
    buffer_size = 10000
    minimal_size = 1000
    batch_size = 1000
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    objective = [1, -1, -1, -1, -1]
    n = 4
    deg = 1
    max_episode = 500
    env = Env(objective, n, deg, max_episode)

    env_name = 'proof_4'
    replay_buffer = ReplayBuffer(buffer_size)
    state_dim = n + 1
    action_dim = None
    agent = DQN(state_dim, units, action_dim, lr, gamma, epsilon, target_update, device, env_name)

    return_list = train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size)

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DQN on {}'.format(env_name))
    plt.savefig(f'./picture/{env_name}_1.png')
    plt.show()

    mv_return = moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DQN on {}'.format(env_name))
    plt.savefig(f'./picture/{env_name}_2.png')
    plt.show()
