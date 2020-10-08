import argparse

from scipy.optimize import differential_evolution

from sklearn.svm import SVC
from imblearn.metrics import geometric_mean_score

import numpy as np
import pickle

with open('../X_train.pickle', 'rb') as f:
    X_train = pickle.load(f)

with open('../y_train.pickle', 'rb') as f:
    y_train = pickle.load(f)

with open('../X_test.pickle', 'rb') as f:
    X_test = pickle.load(f)

with open('../y_test.pickle', 'rb') as f:
    y_test = pickle.load(f)

__author__ = "Manolomon"
__license__ = "MIT"
__version__ = "1.0"

import pickle


def logger(xa, convergence):
    x_str = np.array_repr(xa).replace('\n', '')
    print(x_str)


def kernel_picker(number):
    if number <= 0.25:
        return 'linear'
    elif number <= 0.5:
        return 'poly'
    elif number <= 0.75:
        return 'rbf'
    else:
        return 'sigmoid'


def fitness_func(individual):  # Fitness Function
    global X_train
    global y_train
    classifier = SVC(
        C=individual[0],
        kernel=kernel_picker(individual[1]),
        degree=int(round(individual[2])),
        gamma=individual[3],
        probability=False if individual[4] < 0.5 else True,
        random_state=int(round(individual[5]))
    )
    classifier.fit(X_train, y_train)

    g_mean = geometric_mean_score(
        y_test, classifier.predict(X_test), average='weighted')
    del classifier
    return -1 * g_mean


bounds = [  # Parameters tuned in https://github.com/miguelfzafra/Latest-News-Classifier plus random_state
    (0, 1),  # C: float, optional (default=1.0)
    (0, 1),  # kernel : string, optional (default=’rbf’)
    (1, 5),  # degree : int, optional (default=3)
    (0, 100),  # gamma : float, optional (default=0.0)
    (0, 1),  # probability: boolean, optional (default=False)
    (0, 10)  # random_state: int or RandomState, default=None
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description='Support Vector Machine Hyperparameter tuning for software requirements categorization using Differential Evolution')
    ap.add_argument("-v", "--verbose",
                    help="increase output verbosity", action="store_true")
    ap.add_argument('--np', dest='np', type=int,
                    required=True, help='Population size')
    ap.add_argument('--max_gen', dest='max_gen', type=int,
                    required=True, help='Genarations')
    ap.add_argument('--f', dest='f', type=float,
                    required=True, help='Scale Factor')
    ap.add_argument('--cr', dest='cr', type=float,
                    required=True, help='Crossover percentage')
    ap.add_argument('--datfile', dest='datfile', type=str,
                    help='File where it will be save the score (result)')

    args = ap.parse_args()

    result = differential_evolution(fitness_func, bounds, disp=True, popsize=args.np, maxiter=args.max_gen,
                                    mutation=args.f, recombination=args.cr, strategy='rand1bin', callback=logger)

    # print("Best individual: [C=%s, kernel=%s, degree=%s, gamma=%s, probability=%s, random_state=%s] f(x)=%s" % (
    #    result.x[0],
    #    kernel_picker(result.x[1]),
    #     int(round(result.x[2])),
    #    result.x[3],
    #    False if result.x[4] < 0.5 else True,
    #    int(round(result.x[5])),
    #    result.fun*(-100)))

    if args.datfile:
        with open(args.datfile, 'w') as f:
            f.write(str(result.fun*(-100)))
