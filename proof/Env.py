from proof.polynomial_mul import polynomial_mul, polynomial_mul_
from proof.monomials_generate import monomials
from proof.mapping import mul_polynomial_with_fft, get_map
from proof.Example import Example
from multiprocessing import Pool
import multiprocessing as mp
import numpy as np
import cvxpy as cp
import sympy as sp
import timeit


class Env:
    def __init__(self, example: Example, max_episode):
        self.n = example.n  # the number of variable
        self.deg = example.obj_deg  # the highest degree of polynomial
        self.l = example.l
        self.poly, self.poly_list = monomials(self.n, self.l)
        self.sp_poly = np.array([sp.sympify(e) for e in self.poly])
        self.len_vector = len(self.poly)
        self.objective = self.get_objective(example.objective)

        ### fft_poly
        self.poly_map = [get_map(e, self.l + 1) for e in self.poly_list]
        self.dic_forward = dict(zip(range(self.len_vector), self.poly_map))
        self.dic_reverse = dict(zip(self.poly_map, range(self.len_vector)))
        self.max_map = max(self.poly_map)

        # print(self.max_map, len(self.poly_list))
        # import os
        # os.system('pause')
        ###
        # self.dic = {}
        # for i, e in enumerate(self.poly_list):
        #     self.dic[tuple(e)] = i
        self.M, self.M_, self.A = None, None, None
        self.M_deg_map = {}
        self.first_deg_pos = -1

        self.max_episode = max_episode
        self.episode = 0
        self.s = None
        self.memory, self.memory_action = None, None
        self.len_memory = 0
        self.coefficient_matrix = None
        self.set_memory, self.set_action, self.set_M = None, None, None
        self.last_gamma, self.last_len = None, None
        self.state = None
        self.map = {}
        self.tuple_memory = []
        self.action = None
        self.origin_state = None
        self.memory_initialization()

    def reset(self):
        self.episode = 0
        self.memory, self.action = self.M.copy(), self.A.copy()
        self.len_memory = len(self.memory)
        self.last_len = self.len_memory

        self.coefficient_matrix = np.array(self.memory).T
        self.set_memory = set([tuple(e) for e in self.memory])
        self.set_action = set([tuple(e) for e in self.action])
        self.tuple_memory = [tuple(e) for e in self.memory]
        self.last_gamma, _ = self.compute_linear_programming()

        # print('reward:', self.last_gamma)
        # state.append(self.len_memory - len(self.M))
        state = self.origin_state.copy()
        self.state = (tuple(self.tuple_memory), state)
        self.map[tuple(self.tuple_memory)] = self.action
        # print(self.coefficient_matrix)
        # print(self.memory)
        return self.state, None

    def step(self, action):
        self.episode += 1

        self.add_memory(action)
        print(f'The iteration:{self.episode}')
        # print(f'The action:{action}')
        if self.len_memory > self.last_len:
            gamma, _ = self.compute_linear_programming()
            # state.append(self.len_memory - len(self.M))
            state = self.get_state(self.state[1], action)
            self.state = (tuple(self.tuple_memory), state)
            self.map[tuple(self.tuple_memory)] = self.action
        else:
            gamma = self.last_gamma

        done = True if gamma >= 0 else False
        # if done and self.last_gamma < 0:
        #     self.add_M(action)
        reward = self.get_reward(gamma)

        reward = reward + 1 if done else reward

        truncated = True if self.episode >= self.max_episode else False

        print('reward:', reward, 'done:', done, 'len_memory:', self.len_memory, 'len_action:', len(self.action))

        return self.state, reward, done, truncated, self.episode

    def get_reward(self, gamma):
        reward = -0.05
        # if self.len_memory > self.last_len:
        #     self.last_len = self.len_memory
        #     reward += 0.15
        if gamma > self.last_gamma:
            self.last_gamma = gamma
            reward += 1.05
        return reward

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
            # print(x.value.T)
            # for i, item in enumerate(x.value.T[0]):
            #     if abs(item) > 1e-10:
            #         print(item, i)
            #         ss = 0
            #         for j, k in zip(self.M[i], self.sp_poly):
            #             ss += j * k
            #         print(ss)
            state = list(s.T[0])
            # print('s:', s)
            # print('state:', state)
            print('sum:', sum(self.sp_poly @ s))
            return y.value, state
        else:
            return None, None
        # print(A)
        # print(b)

    def add_memory(self, memory):
        memory = list(memory)
        if tuple(memory) not in self.set_memory:
            self.set_memory.add(tuple(memory))
            self.tuple_memory.append(tuple(memory))
            self.memory.append(memory)
            self.len_memory += 1
            self.coefficient_matrix = np.concatenate((self.coefficient_matrix, np.array([memory]).T), axis=1)
            if self.M_deg_map[tuple(memory)] < self.l:
                tmp = []
                for mul in self.M_:
                    new_poly = mul_polynomial_with_fft(memory, mul, self.dic_forward, self.dic_reverse, self.len_vector,
                                                       self.max_map)
                    if tuple(new_poly) not in self.set_action:
                        self.set_action.add(tuple(new_poly))
                        tmp.append(new_poly)
                        self.M_deg_map[tuple(new_poly)] = self.M_deg_map[tuple(memory)] + 1
                if len(tmp) > 0:
                    self.action = np.concatenate((self.action, np.array(tmp)), axis=0)
        # print(self.coefficient_matrix)

    def memory_initialization(self):
        max_obj_deg = self.deg
        M_ = []
        _, poly = monomials(self.n * 2, max_obj_deg)

        for i in range(self.n):
            tmp = [0] * self.len_vector
            tmp[i + 1] = 1
            M_.append(tmp)
        for i in range(self.n):
            tmp = [0] * self.len_vector
            tmp[0], tmp[i + 1] = 1, -1
            M_.append(tmp)
        self.M_ = M_
        # for item in poly:
        #     print(sum(item))
        poly = poly[1:]
        pool = Pool(processes=mp.cpu_count() // 3)
        res = pool.map(self.compute_memory, poly)
        pool.close()
        pool.join()
        # for item in poly:
        #     tmp = [1] + [0] * (self.len_vector - 1)
        #     for i in range(len(item)):
        #         if item[i] > 0:
        #             for j in range(item[i]):
        #                 tmp = mul_polynomial_with_fft(tmp, M_[i], self.dic_forward, self.dic_reverse, self.len_vector,
        #                                               self.max_map)
        #     M.append(tmp)

        for x, y in zip(res, poly):
            self.M_deg_map[tuple(x)] = sum(y)

        for i, x in enumerate(poly):
            if self.first_deg_pos < 0 and sum(x) == self.deg:
                self.first_deg_pos = i
                break

        self.M = res
        self.set_M = set([tuple(e) for e in self.M])
        self.origin_state = list(np.max(np.array(res), axis=0))
        # print(len(self.M), self.first_deg_pos)

        self.memory_action = self.M[self.first_deg_pos:]
        action = []
        self.set_action = set()
        for item in self.memory_action:
            if self.M_deg_map[tuple(item)] < self.l:
                for mul in self.M_:
                    new_poly = mul_polynomial_with_fft(item, mul, self.dic_forward, self.dic_reverse, self.len_vector,
                                                       self.max_map)
                    if tuple(new_poly) not in self.set_action:
                        self.set_action.add(tuple(new_poly))
                        action.append(new_poly)
                        self.M_deg_map[tuple(new_poly)] = self.M_deg_map[tuple(item)] + 1
        self.A = np.array(action)

        print('self.M', len(res))
        print('self.A', len(self.A))
        # print(len(action))
        # for item in res:
        #     s = 0
        #     for x, y in zip(item, self.sp_poly):
        #         s += x * y
        #     print(s)

    def compute_memory(self, item):
        res = [1] + [0] * (self.len_vector - 1)
        for k in range(len(item)):
            if item[k] > 0:
                for j in range(item[k]):
                    res = mul_polynomial_with_fft(res, self.M_[k], self.dic_forward, self.dic_reverse, self.len_vector,
                                                  self.max_map)
        return res

    def get_objective(self, item: dict):
        dic = {}
        for i, e in enumerate(self.poly):
            dic[e] = i
        res = [0] * self.len_vector
        for key, value in item.items():
            res[dic[key]] += value
        return res

    def get_state(self, a, b):
        res = [max(x, y) for x, y in zip(a, b)]
        return res

    def add_M(self, m):
        if tuple(m) not in self.set_M:
            self.set_M.add(tuple(m))
            self.M.append(m)


if __name__ == '__main__':
    from proof.Example import get_examples_by_name

    ex = get_examples_by_name('case_3')
    env = Env(ex, 100)
    env.reset()
    # env.memory_initialization()
