from proof.polynomial_mul import polynomial_mul, polynomial_mul_
from proof.monomials_generate import monomials
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
        self.tuple_memory = []

    def reset(self):
        self.episode = 0
        self.memory = []
        self.action = []
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
            for i in range(self.n):
                new_poly1 = polynomial_mul_(item, (0, i + 1), self.poly_list, self.dic)
                new_poly2 = polynomial_mul_(item, (1, i + 1), self.poly_list, self.dic)
                self.action.extend([new_poly1, new_poly2])
        self.action = np.array(self.action)

        self.coefficient_matrix = np.array(self.memory).T
        self.set_memory = set([tuple(e) for e in self.memory])
        self.tuple_memory = [tuple(e) for e in self.memory]
        self.last_gamma, state = self.compute_linear_programming()

        self.state = (tuple(self.tuple_memory), state)
        self.map[tuple(self.tuple_memory)] = self.action
        # print(self.coefficient_matrix)
        # print(self.memory)
        return self.state, None

    def step(self, action):
        self.episode += 1
        # action [0,2*n*|M|-1]
        # pos = action // (2 * self.n)
        # pos_x = action % (2 * self.n)
        # if pos_x % 2:
        #     new_poly = polynomial_mul_(self.memory[pos], (1, pos_x // 2 + 1), self.poly_list, self.dic)
        # else:
        #     new_poly = polynomial_mul_(self.memory[pos], (0, pos_x // 2 + 1), self.poly_list, self.dic)

        self.add_memory(action)
        print(f'The iteration:{self.episode}')
        print(f'The action:{action}')

        gamma, state = self.compute_linear_programming()
        self.state = (tuple(self.tuple_memory), state)

        reward = gamma - self.last_gamma
        reward = -0.2 if reward == 0 else reward
        self.last_gamma = gamma
        done = True if gamma >= 0 else False
        reward = reward + 1 if done else reward
        truncated = True if self.episode >= self.max_episode else False
        print('reward:', reward, 'done:', done, 'len_memory:', self.len_memory)
        # self.visualization(done, coff)
        self.map[tuple(self.tuple_memory)] = self.action
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
            state = list(s.T[0])
            # print('s:', s)
            print('state:', state)
            print('sum:', sum(self.sp_poly @ s))
            return y.value, state
        else:
            return None, None
        # print(A)
        # print(b)

    def add_memory(self, memory):
        memory = list(memory)
        if sum(memory) != 0 and (not tuple(memory) in self.set_memory):
            self.set_memory.add(tuple(memory))
            self.tuple_memory.append(tuple(memory))
            self.memory.append(memory)
            self.len_memory += 1
            self.coefficient_matrix = np.concatenate((self.coefficient_matrix, np.array([memory]).T), axis=1)
            tmp = []
            for i in range(self.n):
                new_poly1 = polynomial_mul_(memory, (0, i + 1), self.poly_list, self.dic)
                new_poly2 = polynomial_mul_(memory, (1, i + 1), self.poly_list, self.dic)
                tmp.extend([new_poly1, new_poly2])
            self.action = np.concatenate((self.action, np.array(tmp)), axis=0)
        # print(self.coefficient_matrix)


if __name__ == '__main__':
    # ep = env([2, -1, -1, 1, 0, 0, 0, 0, 0, 0], 2, 3, 1500)
    # ep.reset()
    # ma = ep.coefficient_matrix
    # print(ma)
    # ep.compute_reward()
    pass
