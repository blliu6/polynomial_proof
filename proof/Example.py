class Example:
    def __init__(self, n, obj_deg, objective, l, name):
        self.n = n
        self.obj_deg = obj_deg
        self.objective = objective
        self.l = l
        self.name = name


examples = {
    1: Example(
        n=4,
        obj_deg=3,
        objective={'x1*x2**2': 42.875, 'x1*x2': -36.75, 'x1*x3**2': 42.875, 'x1*x3': -36.75, 'x1': 11.9,
                   'x2**2': -18.375, 'x2': 15.75, 'x3**2': -18.375, 'x3': 15.75, '1': 5.25},
        l=5,
        name='case_1'
        # 42.875*t1*t2**2 - 36.75*t1*t2 + 42.875*t1*t3**2 - 36.75*t1*t3 + 11.9*t1 - 18.375*t2**2 + 15.75*t2 - 18.375*t3**2 + 15.75*t3 - 4.1
    ),
    2: Example(
        n=3,
        obj_deg=3,
        objective={'1': 2},
        l=3,
        name='test'
    )
}


def get_examples_by_id(identify):
    return examples[identify]


def get_examples_by_name(name):
    for key in examples.keys():
        if examples[key].name == name:
            return examples[key]
