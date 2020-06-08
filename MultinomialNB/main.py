import argparse

from scipy.optimize import differential_evolution

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score

import numpy as np
import pickle

with open('../features_train.pickle', 'rb') as f:
    X_train = pickle.load(f)

with open('../labels_train.pickle', 'rb') as f:
    y_train = pickle.load(f)

__author__ = "Manolomon"
__license__ = "MIT"
__version__ = "1.0"

import pickle

def logger(xa, convergence):
    x_str = np.array_repr(xa).replace('\n', '')
    print(x_str)

def fitness_func(individual): # Fitness Function
    global X_train
    global y_train
    classifier = MultinomialNB(
        alpha=individual[0], 
        fit_prior = False if individual[1] < 0.5 else True
        )
    classifier.fit(X_train, y_train)

    acc = cross_val_score(classifier, X_train, y_train, cv=5)
    del classifier
    return -1 * acc.mean()

bounds = [
    (0, 100), # alpha: float, default=1.0
    (0, 1), # fit_prior: bool, default=True
    ]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description='MultinomialNB Hyperparameter tuning for software requirements categorization using Differential Evolution')
    ap.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    ap.add_argument('--np', dest='np', type=int, required=True, help='Population size')
    ap.add_argument('--max_gen', dest='max_gen', type=int, required=True, help='Genarations')
    ap.add_argument('--f', dest='f', type=float, required=True, help='Scale Factor')
    ap.add_argument('--cr', dest='cr', type=float, required=True, help='Crossover percentage')
    ap.add_argument('--datfile', dest='datfile', type=str, help='File where it will be save the score (result)')

    args = ap.parse_args()

    result = differential_evolution(fitness_func, bounds, disp=True, popsize=args.np, maxiter=args.max_gen, mutation=args.f, recombination=args.cr, strategy='rand1bin', callback=logger)
    
    #print("Best individual: [alpha=%s, fit_prior=%s] f(x)=%s" % (result.x[0], False if result.x[1] < 0.5 else True, result.fun*(-100)))

    if args.datfile:
        with open(args.datfile, 'w') as f:
            f.write(str(result.fun*(-100)))
