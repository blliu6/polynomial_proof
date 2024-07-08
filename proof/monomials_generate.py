import re
from functools import reduce

import numpy as np
import sympy as sp
from sklearn.preprocessing import PolynomialFeatures


def dfs(n: int, m: int):
    now = [0] * n
    while True:
        yield now
        now[0] += 1
        for i in range(n):
            if now[i] > m or sum(now) > m:
                now[i] = 0
                if i + 1 >= n:
                    return
                now[i + 1] += 1
            else:
                break


def key_sort(item: list):
    res = [sum(item)]
    for i in range(len(item)):
        res.append(-item[i])
    return tuple(res)


def monomials(variable_number, degree):
    poly = [x.copy() for x in dfs(variable_number, degree)]
    poly.sort(key=key_sort)
    x = sp.symbols([f'x{i + 1}' for i in range(variable_number)])
    polynomial = [str(reduce(lambda a, b: a * b, [x[i] ** exp for i, exp in enumerate(e)])) for e in poly]
    return polynomial, poly


def replacer(match):
    return 'x' + str(int(match.group()[1:]) + 1)


def tran(s: str):
    s = s.replace(' ', '*')
    s = s.replace('^', '**')
    s = re.sub(r'x\d+', replacer, s)
    return s


def monomials_(variable_number, degree):
    poly = PolynomialFeatures(degree)
    poly.fit_transform(np.ones((1, variable_number)))
    polynomial = poly.get_feature_names_out()
    polynomial = [tran(e) for e in polynomial]
    print(polynomial)

    return polynomial


if __name__ == '__main__':
    import timeit

    t1 = timeit.default_timer()
    pol1 = monomials_(5, 3)
    t2 = timeit.default_timer()
    pol2 = monomials(5, 3)
    t3 = timeit.default_timer()
    print(pol1 == pol2)
    print(t2 - t1, t3 - t2)
