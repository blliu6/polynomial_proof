import sympy as sp

from proof.FFT import polynomial_fft
from proof.monomials_generate import monomials


def dfs(n: int, m: int):
    now = [0] * n
    while True:
        yield now
        now[0] += 1
        for i in range(n):
            if now[i] > m:
                now[i] = 0
                if i + 1 >= n:
                    return
                now[i + 1] += 1
            else:
                break


def get_map(item: list, deg):
    ans = item[-1]
    for i in range(len(item) - 2, -1, -1):
        ans = ans * deg + item[i]
    return ans


def get_poly(num, deg):
    ans = []
    while num > 0:
        ans.append(num % deg)
        num //= deg
    return ans


def get_fft_pol(item, dic, max_pos):
    res = [0] * (max_pos + 1)
    for i, e in enumerate(item):
        if e != 0:
            res[dic[i]] += e
    return res


def get_poly_fft(item, p, dic):
    res = [0] * p
    for i, e in enumerate(item):
        if abs(e) > 1e-10 and (i in dic):
            res[dic[i]] += e
    return res


def mul_polynomial_with_fft(x, y, dic_forward, dic_reverse, len_p, max_map):
    # poly_map = [get_map(e, max_deg + 1) for e in poly]
    # dic_forward = dict(zip(range(len(poly)), poly_map))
    # dic_reverse = dict(zip(poly_map, range(len(poly))))

    # len_p, max_map = len(poly), max(dic_forward.values())
    x_f = get_fft_pol(x, dic_forward, max_map)
    y_f = get_fft_pol(y, dic_forward, max_map)
    result = polynomial_fft(x_f, y_f, max_map, max_map)

    result = get_poly_fft(result, len_p, dic_reverse)

    result = [round(e) for e in result]

    return result


if __name__ == '__main__':
    import timeit

    deg = 5
    poly_list, poly = monomials(2, deg)

    p = len(poly_list)
    a = [1, 2, 3] + [0] * (p - 3)
    b = [1, 0, 0, 1, 2, 3] + [0] * (p - 6)

    t1 = timeit.default_timer()
    res = mul_polynomial_with_fft(a, b, poly, deg)
    t2 = timeit.default_timer()
    print(t2 - t1)
    print(res)

    sum = 0
    poly_list = [sp.sympify(e) for e in poly_list]
    for x, y in zip(res, poly_list):
        sum += x * y
    print(sum)

    # a = [1, 2, 3]
    # b = [1, 1, 0]
    # v1 = get_map(a, 4)
    # v2 = get_map(b, 4)
    # print(v1 + v2, get_poly(v1 + v2, 4))
