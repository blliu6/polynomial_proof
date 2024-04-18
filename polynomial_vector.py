from monomials_generate import monomials_


class agent_state:
    # n元m次多项式
    def __init__(self, n, m):
        self.poly = monomials_(n, m)
        self.length = len(self.poly)
        self.objective = []
        self.equalities = []
        self.memory = []


if __name__ == '__main__':
    pass
