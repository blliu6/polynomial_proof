from math import cos, sin, pi


def fft(a: list, inv, tot, rev: list):
    for i in range(tot):
        if i < rev[i]:
            a[i], a[rev[i]] = a[rev[i]], a[i]
    mid = 1
    while mid < tot:
        w1 = complex(cos(pi / mid), inv * sin(pi / mid))
        i = 0
        while i < tot:
            wk = complex(1, 0)
            j = 0
            while j < mid:
                x, y = a[i + j], wk * a[i + j + mid]
                a[i + j], a[i + j + mid] = x + y, x - y
                j, wk = j + 1, wk * w1
            i = i + mid * 2
        mid = (mid << 1)


def polynomial_fft(a, b, n, m):
    a, b = [complex(e, 0) for e in a], [complex(e, 0) for e in b]
    bit = 0  # 最高位
    while (1 << bit) < (n + m + 1):
        bit += 1
    tot = (1 << bit)
    a.extend([complex(0, 0) for _ in range(tot - len(a))])
    b.extend([complex(0, 0) for _ in range(tot - len(b))])
    rev = [0] * tot
    for i in range(tot):
        rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << (bit - 1))
    fft(a, 1, tot, rev)
    fft(b, 1, tot, rev)
    for i in range(tot):
        a[i] = a[i] * b[i]
    fft(a, -1, tot, rev)
    res = [e.real / tot for e in a][:n + m + 1]
    return res


if __name__ == '__main__':
    a = [1, 2, 3]
    b = [1, 3]
    res = polynomial_fft(a, b, 2, 1)
    print(res)
