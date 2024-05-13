import collections
import numpy as np
import random


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出

    def add(self, state, action, reward, next_state, done):  # 将数据加入buffer
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return state, np.array(action), reward, next_state, done

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)


if __name__ == '__main__':
    replay = ReplayBuffer(10)
    for i in range(10):
        replay.add([i], [i], [i], [i], [i])

    replay.sample(2)
