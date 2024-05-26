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
from proof.Example import get_examples_by_name


def main():
    env_name = 'case_1'
    example = get_examples_by_name(env_name)
    load = False
    begin = timeit.default_timer()
    opts = {
        'example': example,
        'epsilon_step': 0.02,
        'num_episodes': 50,
        'epsilon': 0.6,
        'multiple_rewards': 10
    }
    config = ProofConfig(**opts)
    lr, num_episodes, units, gamma, epsilon, epsilon_step, target_update, buffer_size, minimal_size, batch_size, device, multiple_rewards = config.get_config()

    max_steps = 500
    env = Env(example, max_steps)

    state_dim = env.len_vector
    action_dim = env.len_vector

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
