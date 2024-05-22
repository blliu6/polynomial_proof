import sympy as sp


class Example:
    def __init__(self, n, obj_deg, objective, l, name):
        self.n = n
        self.obj_deg = obj_deg
        self.objective = objective
        self.l = l
        self.name = name


def convert(origin: str, bounds):
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
    return dic


examples = {
    1: Example(
        n=4,
        obj_deg=3,
        objective=convert('x1*x2**2+x1*x3**2-1.1*x1+1+9.35', [[-1.5, 2]] * 3),
        l=4,
        name='case_1'
        # 42.875*t1*t2**2 - 36.75*t1*t2 + 42.875*t1*t3**2 - 36.75*t1*t3 + 11.9*t1 - 18.375*t2**2 + 15.75*t2 - 18.375*t3**2 + 15.75*t3 - 4.1
    ),
    2: Example(
        n=3,
        obj_deg=2,
        objective=convert('x1-2*x2+x3+0.835634534*x2*(1-x2)+36.71269068', [[-5, 5]] * 3),
        l=3,
        name='case_2'
    )
}


def get_examples_by_id(identify):
    return examples[identify]


def get_examples_by_name(name):
    for key in examples.keys():
        if examples[key].name == name:
            return examples[key]
