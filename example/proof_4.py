import sys

sys.path.append('/home/blliu/pythonproject/polynomial_proof')

import random
import timeit
import numpy as np
import torch
from proof.proof_config import ProofConfig
from proof.dqn import DQN, train_off_policy_agent
from proof.Env import Env
from proof.plot import plot
from proof.reappear import reappear


def main():
    load = False
    begin = timeit.default_timer()
    opts = {
        'epsilon_step': 0.025,
        'num_episodes': 50
    }
    config = ProofConfig(**opts)
    lr, num_episodes, units, gamma, epsilon, epsilon_step, target_update, buffer_size, minimal_size, batch_size, device, multiple_rewards = config.get_config()

    objective = [1, -1, -1, -1, -1]
    n = 4
    deg = 1
    max_steps = 100
    env = Env(objective, n, deg, max_steps)

    env_name = 'proof_4'
    state_dim = n + 1
    action_dim = None

    agent = DQN(state_dim, units, action_dim, lr, gamma, epsilon, target_update, device, env_name, load=load)

    if not load:
        return_list, end = train_off_policy_agent(env, agent, num_episodes, buffer_size, minimal_size, batch_size,
                                                  epsilon_step)
        print(f'Total time: {end - begin}s')
        plot(return_list, env_name)
    else:
        reappear(agent, env)


if __name__ == '__main__':
    random.seed(2024)
    np.random.seed(2024)
    torch.manual_seed(2024)
    main()
