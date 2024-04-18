from monomials_generate import monomials


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


def mapping(n, m, deg):
    # deg 最高次数加一
    # polynomial, poly = monomials(n, m)
    # poly = [get_map(e, deg) for e in poly]
    poly = [x.copy() for x in dfs(n, m)]
    poly = [get_map(e, deg) for e in poly]
    print(poly)


if __name__ == '__main__':
    mapping(3, 2, 3)
    # a = [1, 2, 3]
    # b = [1, 1, 0]
    # v1 = get_map(a, 4)
    # v2 = get_map(b, 4)
    # print(v1 + v2, get_poly(v1 + v2, 4))
