import sys

sys.path.append('/home/blliu/pythonproject/polynomial_proof')

from Env import Env
from dqn import DQN
import torch
import random
import numpy as np

random.seed(2024)
np.random.seed(2024)
torch.manual_seed(2024)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
lr = 1e-3
num_episodes = 1000
units = 128
gamma = 0.98
epsilon = 0.95
target_update = 50

env_name = 'proof_4'
objective = [1, -1, -1, -1, -1]
n = 4
deg = 1
state_dim = n + 1
action_dim = None

max_episode = 1000
env = Env(objective, n, deg, max_episode)

agent = DQN(state_dim, units, action_dim, lr, gamma, epsilon, target_update, device, env_name, load=True)

episode_return = 0
state, info = env.reset()
done, truncated = False, False
while not done and not truncated:
    action = agent.take_action(state, env.action)
    next_state, reward, done, truncated, info = env.step(action)
    episode_return += reward
    print('state:', next_state, 'reward:', reward, 'done:', done)
    state = next_state
