def polynomial_mul(a, ops, poly_list, dic):
    # dic = {}
    # for i, e in enumerate(poly_list):
    #     dic[tuple(e)] = i
    pos = ops[1]  # ops[1] 代表哪个变量
    if ops[0] == 0:  # mul x
        ans = [0] * len(a)
        for i, e in enumerate(a):
            if e != 0:
                tmp = poly_list[i].copy()
                tmp[pos - 1] += 1
                ans[dic[tuple(tmp)]] += e
    else:  # mul 1-x
        ans = a.copy()
        for i, e in enumerate(a):
            if e != 0:
                tmp = poly_list[i].copy()
                tmp[pos - 1] += 1
                ans[dic[tuple(tmp)]] -= e
    return ans


def polynomial_mul_(a, ops, poly_list, dic):  # 论文特殊情况
    # dic = {}
    # for i, e in enumerate(poly_list):
    #     dic[tuple(e)] = i
    pos = ops[1]  # ops[1] 代表哪个变量
    if ops[0] == 0:  # mul x
        ans = [0] * len(a)
        ans[pos] = a[0] + a[pos]
    else:  # mul 1-x
        ans = a.copy()
        ans[pos] += -(a[0] + a[pos])
    return ans


if __name__ == '__main__':
    # poly, poly_list = monomials(3, 2)
    # print(poly, poly_list)
    # dic1 = {}
    # for i, e in enumerate(poly_list):
    #     dic1[tuple(e)] = i
    #
    # a = [1, 2, 3, 4, 0, 0, 0, 0, 0, 0]
    # pol = 0
    # import sympy as sp
    #
    # for i in range(len(a)):
    #     pol += a[i] * sp.sympify(poly[i])
    # print(pol)
    # ans = polynomial_mul(a, [1, 3], poly_list, dic1)
    # print(ans)
    # pol = 0
    # for i in range(len(a)):
    #     pol += ans[i] * sp.sympify(poly[i])
    # print(pol)
    ans = polynomial_mul_([1, -1, 0, 0, 0], (0, 2), None, None)
    print(ans)
