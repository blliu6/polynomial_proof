import time
import timeit

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from proof.ReplayBuffer import ReplayBuffer
from proof.feature_vector_net import FeatureVectorNet


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


def handle(states):
    prefix = [0]
    ans = 0
    for item in states:
        ans += len(item)
        prefix.append(ans)

    res = np.concatenate(states, axis=0)
    return res, prefix


class DQN:
    def __init__(self, state_dim, units, action_dim, learning_rate, gamma, epsilon, target_update, device, name,
                 load=False):
        self.feature = FeatureVectorNet(state_dim).to(device)
        self.feature_optimizer = torch.optim.AdamW(self.feature.parameters(), lr=learning_rate)
        self.action_dim = action_dim
        self.q_net = Qnet(500 + action_dim, dense=4, units=units).to(device)
        self.target_q_net = Qnet(500 + action_dim, dense=4, units=units).to(device)
        # self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.optimizer = torch.optim.RMSprop(self.q_net.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.count = 0
        self.device = device
        self.steps = 1e6
        self.name = name

        if load:
            self.q_net.load_state_dict(torch.load(f'../model/{self.name}.pth'))
            print('Parameters loaded successfully!')

    def take_action(self, state, action):
        state = state[1]
        if np.random.random() > self.epsilon:
            pos = np.random.randint(len(action))
        else:
            state, prefix = handle([state])
            state = torch.tensor(state, dtype=torch.float).to(self.device)
            # print(state.shape, prefix)
            state = self.feature(state, prefix).cpu().detach().numpy()[0]
            # print(state.shape)
            input = np.concatenate((np.array([state] * len(action)), action), axis=1)
            state = torch.tensor(input, dtype=torch.float).to(self.device)
            pos = self.q_net(state).argmax().item()
        print(f'pos:{pos}')
        return action[pos]

    def get_next_q(self, next_states_original, next_state, action_map):
        res = []
        # res = torch.empty((len(next_state), 1)).to(self.device)
        for i, state in enumerate(next_state):
            action = action_map[next_states_original[i]]
            input = np.concatenate((np.array([state] * len(action)), action), axis=1)
            state = torch.tensor(input, dtype=torch.float).to(self.device)
            # res[i] = self.target_q_net(state).max()
            res.append(self.target_q_net(state).max().reshape(-1, 1))

        return torch.cat(res, 0)

    def update(self, transition_dict, env):
        states_ = transition_dict['states']
        next_states_ = transition_dict['next_states']

        states_original, states = [item[0] for item in states_], [item[1] for item in states_]
        next_states_original, next_states = [item[0] for item in next_states_], [item[1] for item in next_states_]

        f_states, s_prefix = handle(states)
        f_next, next_prefix = handle(next_states)
        f_states = torch.tensor(f_states, dtype=torch.float).to(self.device)
        f_next = torch.tensor(f_next, dtype=torch.float).to(self.device)
        # print(f_states.shape, f_next.shape)
        states = self.feature(f_states, s_prefix)
        next_states = self.feature(f_next, next_prefix).cpu().detach().numpy()

        # print(len(states[0]), len(states[0][0]))

        # states = torch.tensor(states, dtype=torch.float).to(self.device)
        # print(states.shape, next_states.shape)
        # time.sleep(1000)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        # next_states = torch.tensor(next_states, dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        input = torch.cat((states, actions), dim=1)
        q_values = self.q_net(input)

        max_next_q_values = self.get_next_q(next_states_original, next_states, env.map)

        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)  # TD误差目标
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
        self.optimizer.zero_grad()
        self.feature_optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()
        self.feature_optimizer.step()
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())  # 更新目标网络
        self.count += 1

    def save(self):
        torch.save(self.q_net.state_dict(), f'../model/{self.name}.pth')
        print('Model saved successfully!')


def train_off_policy_agent(env, agent, num_episodes, buffer_size, minimal_size, batch_size, epsilon_step=0.01,
                           multiple_rewards=10):
    return_list = []
    min_episode, end = 0, 0
    replay_buffer = ReplayBuffer(buffer_size)
    for i_episode in range(num_episodes):
        if i_episode % 10 == 0 and agent.epsilon < 1:
            agent.epsilon = min(agent.epsilon + epsilon_step, 1)
        episode_return = 0
        state, info = env.reset()
        print(f'Episode:{i_episode},len_memory:{env.len_memory},len_action:{len(env.action)}')
        done, truncated = False, False
        while not done and not truncated:
            action = agent.take_action(state, env.action)
            next_state, reward, done, truncated, info = env.step(action)

            # if done and agent.steps > info and agent.epsilon == 1:
            if done and agent.steps > info:
                agent.steps, min_episode, end = info, i_episode, timeit.default_timer()
                print(f'Proof steps:{agent.steps}, episode:{min_episode}')
                agent.save()

            if reward > 0:
                for i in range(multiple_rewards - 1):
                    replay_buffer.add(state, action, reward, next_state, done)

            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            episode_return += reward

            if replay_buffer.size() > minimal_size and replay_buffer.size() >= 10000:  # and i_episode > 40:
                b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r,
                                   'dones': b_d}
                agent.update(transition_dict, env)

        return_list.append(episode_return)
        print(f'Sum of reward: {episode_return},agent_steps: {agent.steps}')
    print(f'Minimum number of proof steps:{agent.steps}, Minimum episode:{min_episode}')
    return return_list, end


if __name__ == '__main__':
    pass
