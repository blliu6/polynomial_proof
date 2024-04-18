import numpy as np


def cauchy_mul(a, b, deg):
    # deg是b的次数
    ans = np.array(a)
    cur = ans
    for i in range(deg):
        cur = np.insert(cur[:-1], 0, 0)
        ans = np.vstack((ans, cur))

    b_vector = np.array(b)[:deg + 1]
    ans = np.dot(b_vector, ans)
    return ans


if __name__ == '__main__':
    a = [3, -1, 2, 0, 0, 0]
    b = [0, 1, 0, 0, 0, 0]

    res = cauchy_mul(a, b, 1)
    print(res)
