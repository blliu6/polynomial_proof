import sympy as sp


class Example:
    def __init__(self, n, obj_deg, objective, l, name):
        self.n = n
        self.obj_deg = obj_deg
        self.objective = objective
        self.l = l
        self.name = name


def convert(origin: str, lb, bounds):
    for i, bound in enumerate(bounds):
        low, high = bound[0], bound[1]
        origin = origin.replace(f'x{i + 1}', f'({high - low}*x{i + 1}+{low})')
    origin = sp.sympify(origin)
    origin = sp.expand(origin)
    terms = origin.as_ordered_terms()
    dic = {}
    for term in terms:
        coef = term.as_coeff_Mul()
        dic[str(coef[1])] = coef[0]
    if '1' not in dic:
        dic['1'] = -lb
    else:
        dic['1'] += -lb
    return dic


examples = {
    1: Example(
        n=4,
        obj_deg=3,
        objective=convert('x1*x2**2+x1*x3**2-1.1*x1+1', -9.35, [[-1.5, 2]] * 3),
        l=4,
        name='case_1'
        # 42.875*t1*t2**2 - 36.75*t1*t2 + 42.875*t1*t3**2 - 36.75*t1*t3 + 11.9*t1 - 18.375*t2**2 + 15.75*t2 -
        # 18.375*t3**2 + 15.75*t3 - 4.1
    ),
    2: Example(
        n=3,
        obj_deg=2,
        objective=convert('x1-2*x2+x3+0.835634534*x2*(1-x2)', -36.71269068, [[-5, 5]] * 3),
        l=3,
        name='case_2'
    ),
    3: Example(
        n=4,
        obj_deg=3,
        objective=convert('x1*x2**2+x1*x3**2+x1*x4**2-1.1*x1+1', -20.8, [[-2, 2]] * 4),
        l=4,
        name='case_3'
    ),
    4: Example(
        n=4,
        obj_deg=6,
        objective=convert(
            '-x1*x3**3+4*x2*x3**2*x4+4*x1*x3*x4**2+2*x2*x4**3+4*x1*x3+4*x3**2-10*x2*x4-10*x4**2+2', -3.1800966258,
            [[-5, 5]] * 4),
        l=6,
        name='case_4'
    ),
    5: Example(
        n=5,
        obj_deg=5,
        objective=convert('x5**2+x1+x2+x3+x4-x5-10', -30.25, [[-5, 5]] * 5),
        l=5,
        name='case_5'
    ),
    6: Example(
        n=5,
        obj_deg=6,
        objective=convert('-1+2*x1**6-2*x2**6+2*x3**6-2*x4**6+2*x5**6', -5, [[-1, 1]] * 5),
        l=6,
        name='case_6'
    ),
    8: Example(
        n=5,
        obj_deg=5,
        objective=convert('x1*x2*x3*x4+x1*x2*x3*x5+x1*x2*x4*x5+x1*x3*x4*x5+x2*x3*x4*x5', -30000, [[-10, 10]] * 5),
        l=5,
        name='case_8'
        # 可
    ),
    9: Example(
        n=6,
        obj_deg=4,
        objective=convert('2*x1**2+2*x2**2+2*x3**2+2*x4**2+2*x5**2+x6**2-x6', -0.25, [[-5, 5]] * 6),
        l=4,
        name='case_9'
    ),
    12: Example(
        n=6,
        obj_deg=4,
        objective=convert('x3**3*x4+x2**3*x5+x1**3*x6-0.25', -1875.25, [[2, 5]] * 3 + [[-5, -2]] * 3),
        l=4,
        name='case_12'
        # 可
    ),
    14: Example(
        n=7,
        obj_deg=4,
        objective=convert('x1**2 + 2*x2**2 + 2*x3**2 + 2*x4**2 + 2*x5**2 + 2*x6**2 + 2*x7**2 - x1', -0.25,
                          [[-1, 1]] * 7),
        l=4,
        name='case_14'
    )
}


def get_examples_by_id(identify):
    return examples[identify]


def get_examples_by_name(name):
    for key in examples.keys():
        if examples[key].name == name:
            return examples[key]


if __name__ == '__main__':
    ex = get_examples_by_name('case_3')
