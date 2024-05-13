from polynomial_mul import polynomial_mul, polynomial_mul_
from monomials_generate import monomials
import numpy as np
import cvxpy as cp
import sympy as sp


class Env:
    def __init__(self, objective, n, deg, max_episode):
        self.objective = objective
        self.n = n  # the number of variable
        self.deg = deg  # the highest degree of polynomial
        self.poly, self.poly_list = monomials(self.n, self.deg)
        self.sp_poly = np.array([sp.sympify(e) for e in self.poly])
        self.len_vector = len(self.poly)
        self.dic = {}
        for i, e in enumerate(self.poly_list):
            self.dic[tuple(e)] = i
        self.max_episode = max_episode
        self.episode = 0
        self.s = None
        self.memory = []
        self.len_memory = 0
        self.coefficient_matrix = None
        self.set_memory = None
        self.last_gamma = None
        self.state = None
        self.action = None
        self.action_ = None
        self.map = {}

    def reset(self):
        self.episode = 0
        self.memory = []
        self.action = None
        for i in range(self.n):
            tmp = [0] * self.len_vector
            tmp[i + 1] = 1
            self.memory.append(tmp)
            tmp = [0] * self.len_vector
            tmp[0], tmp[i + 1] = 1, -1
            self.memory.append(tmp)
        self.len_memory = len(self.memory)

        self.action_ = np.array(self.memory)
        for item in self.memory:
            tmp = np.concatenate((np.array([item] * self.n * 2), self.action_), axis=1)
            if self.action is None:
                self.action = tmp
            else:
                self.action = np.concatenate((self.action, tmp), axis=0)

        self.coefficient_matrix = np.array(self.memory).T
        self.set_memory = set([tuple(e) for e in self.memory])
        self.last_gamma, _ = self.compute_linear_programming()

        self.map[tuple(self.state)] = self.action
        # print(self.coefficient_matrix)
        # print(self.memory)
        return self.state, _

    def step(self, action):
        self.episode += 1
        # action [0,2*n*|M|-1]
        pos = action // (2 * self.n)
        pos_x = action % (2 * self.n)
        if pos_x % 2:
            new_poly = polynomial_mul_(self.memory[pos], (1, pos_x // 2 + 1), self.poly_list, self.dic)
        else:
            new_poly = polynomial_mul_(self.memory[pos], (0, pos_x // 2 + 1), self.poly_list, self.dic)

        self.add_memory(new_poly)
        print(f'The iteration:{self.episode}')
        print(f'The action:{action}')
        gamma, coff = self.compute_linear_programming()

        reward = gamma - self.last_gamma
        reward = -0.2 if reward == 0 else reward
        self.last_gamma = gamma
        done = True if gamma >= 0 else False
        reward = reward + 1 if done else reward
        truncated = True if self.episode > self.max_episode else False
        print('reward:', reward, 'done:', done, 'len_memory:', self.len_memory)
        # self.visualization(done, coff)
        self.map[tuple(self.state)] = self.action
        return self.state, reward, done, truncated, self.episode

    def compute_linear_programming(self):
        x = cp.Variable((self.len_memory, 1))
        y = cp.Variable()
        no_constant = self.coefficient_matrix[1:]
        constant = self.coefficient_matrix[0:1]

        b = np.array([self.objective[1:]]).T
        A = np.diag(np.ones(self.len_memory))

        obj = cp.Maximize(y)
        zero = np.zeros((self.len_memory, 1))

        constraints = [A @ x >= zero, no_constant @ x == b, constant @ x == self.objective[0] - y]

        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.GUROBI)
        if prob.status == cp.OPTIMAL:
            # print('Lambda:', y.value)
            # print('Reward:', x.value)
            s = self.coefficient_matrix @ x.value
            self.state = list(s.T[0])
            # print('s:', s)
            print('state:', self.state)
            print('sum:', sum(self.sp_poly @ s))
            return y.value, x.value
        else:
            return None, None
        # print(A)
        # print(b)

    def add_memory(self, memory):
        if sum(memory) != 0 and (not tuple(memory) in self.set_memory):
            self.set_memory.add(tuple(memory))
            self.memory.append(memory)
            self.len_memory += 1
            self.coefficient_matrix = np.concatenate((self.coefficient_matrix, np.array([memory]).T), axis=1)
            self.action = np.concatenate(
                (self.action, np.concatenate((np.array([memory] * self.n * 2), self.action_), axis=1)), axis=0)
        # print(self.coefficient_matrix)


if __name__ == '__main__':
    # ep = env([2, -1, -1, 1, 0, 0, 0, 0, 0, 0], 2, 3, 1500)
    # ep.reset()
    # ma = ep.coefficient_matrix
    # print(ma)
    # ep.compute_reward()
    pass
