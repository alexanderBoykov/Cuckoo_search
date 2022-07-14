import numpy as np




def fitness_1(X):

    X = np.array(X).reshape(-1, 1)
    x, y = X[0][0], X[1][0]
    f = (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2
    return f




def fitness_2(X):

    X = np.array(X).reshape(-1, 1)
    x, y = X[0][0], X[1][0]
    f = (x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2
    return f




def fitness_3(X):

    X = np.array(X).reshape(-1, 1)
    x, y = X[0][0], X[1][0]
    f = (1.5 - x + x * y) ** 2 + (2.25 - x + x * (y ** 2)) ** 2 + (2.625 - x + x * (y ** 3)) ** 2
    return f


def fitness_4(X):

    X = np.array(X).reshape(-1, 1)
    x, y = X[0][0], X[1][0]
    f = 2 * x * y + 2 * x - x ** 2 - 2 * (y ** 2)
    return f


def fitness_5(X):

    X = np.array(X).reshape(-1, 1)
    m = 10
    x, y = X[0][0], X[1][0]
    f = -np.sin(x) * (np.sin(x * x / np.pi) ** (2 * m)) - np.sin(y) * (np.sin(2 * y * y / np.pi) ** (2 * m))
    return f