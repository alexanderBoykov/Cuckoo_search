import numpy as np
import matplotlib.pyplot as plt
from math import gamma


def fitness_1(X):
    X = np.array(X).reshape(-1, 1)
    x, y = X[0][0], X[1][0]
    f = (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2
    return f


def fitness_2(X):
    X = np.array(X).reshape(-1, 1)
    x, y = X[0][0], X[1][0]
    f = (1-x)**2+100*(y-x**2)**2
    return f









class CSO:

    def __init__(self, fitness, P=100, n=100, pa=0.25, beta=1.5, bound=None,
                 plot=False, min=True, verbose=False, Tmax=300):


        self.fitness = fitness
        self.P = P
        self.n = n
        self.Tmax = Tmax
        self.pa = pa
        self.beta = beta
        self.bound = bound
        self.plot = plot
        self.min = min
        self.verbose = verbose


        self.X = []

        if bound is not None:
            for (U, L) in bound:
                x = (U - L) * np.random.rand(P, ) + L
                self.X.append(x)
            self.X = np.array(self.X).T
        else:
            self.X = np.random.randn(P, n)

    def update_position_1(self):

    #CALCULATE THE CHANGE OF POSITION 'X = X + rand*C' USING LEVY FLIGHT METHOD

        num = gamma(1 + self.beta) * np.sin(np.pi * self.beta / 2)
        den = gamma((1 + self.beta) / 2) * self.beta * (2 ** ((self.beta - 1) / 2))
        σu = (num / den) ** (1 / self.beta)
        σv = 1
        u = np.random.normal(0, σu, self.n)
        v = np.random.normal(0, σv, self.n)
        S = u / (np.abs(v) ** (1 / self.beta))


        for i in range(self.P):
            if i == 0:
                self.best = self.X[i, :].copy()
            else:
                self.best = self.optimum(self.best, self.X[i, :])

        Xnew = self.X.copy()
        for i in range(self.P):
            extra=np.random.randn(self.n)

            Xnew[i, :] += extra * 0.01 * S * (Xnew[i, :] - self.best)
            self.X[i, :] = self.optimum(Xnew[i, :], self.X[i, :])

    def update_position_2(self):

    #REPLACE SOME NEST WITH NEW SOLUTIONS

        Xnew = self.X.copy()
        Xold = self.X.copy()
        for i in range(self.P):
            d1, d2 = np.random.randint(0, 5, 2)
            for j in range(self.n):
                r = np.random.rand()
                if r < self.pa:
                    Xnew[i, j] += np.random.rand() * (Xold[d1, j] - Xold[d2, j])
            self.X[i, :] = self.optimum(Xnew[i, :], self.X[i, :])

    def optimum(self, best, particle_x):

    #COMPARE PARTICLE'S CURRENT POSITION WITH GLOBAL BEST POSITION

        if self.min:
            if self.fitness(best) > self.fitness(particle_x):
                best = particle_x.copy()
        else:
            if self.fitness(best) < self.fitness(particle_x):
                best = particle_x.copy()
        return best

    def clip_X(self):

    # IF BOUND IS SPECIFIED THEN CLIP 'X' VALUES SO THAT THEY ARE IN THE SPECIFIED RANGE
        if self.bound is not None:
            for i in range(self.n):
                xmin, xmax = self.bound[i]
                self.X[:, i] = np.clip(self.X[:, i], xmin, xmax)

    def execute(self):



        self.fitness_time, self.time = [], []

        for t in range(self.Tmax):
            self.update_position_1()
            self.clip_X()
            self.update_position_2()
            self.clip_X()
            self.fitness_time.append(self.fitness(self.best))
            self.time.append(t)
            if self.verbose:
                print('Iteration:  ', t, '| best global fitness (cost):', round(((1-self.best[0])**2+(1-self.best[1])**2), 7))

        print('\nOPTIMUM SOLUTION\n  >', np.round(self.best.reshape(-1), 7).tolist())
        print('\nOPTIMUM FITNESS\n  >', np.round(self.fitness(self.best), 7))
        print()
        if self.plot:
            self.Fplot()

    def Fplot(self):

        # PLOTS GLOBAL FITNESS (OR COST) VALUE VS ITERATION GRAPH

        plt.plot(self.time, self.fitness_time)
        plt.title('Fitness value vs Iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Fitness value')
        plt.show()









CSO(fitness=fitness_2, bound=[(0,10),(0,10)],n=2,verbose=True,plot=True,Tmax=30).execute()