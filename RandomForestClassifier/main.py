import argparse

from scipy.optimize import differential_evolution

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

import pickle

with open('../features_train.pickle', 'rb') as f:
    X_train = pickle.load(f)

with open('../labels_train.pickle', 'rb') as f:
    y_train = pickle.load(f)

__author__ = "Manolomon"
__license__ = "MIT"
__version__ = "1.0"

import pickle

def fitness_func(individual): # Fitness Function
    global X_train
    global y_train
    classifier = RandomForestClassifier(
        n_estimators=int(round(individual[0])),
        max_depth=None if individual[1] < 2 else int(round(individual[1])),
        min_samples_split=individual[2],
        min_samples_leaf=individual[3],
        max_features="auto" if individual[4] < 0.5 else "sqrt",
        bootstrap=False if individual[5] < 0.5 else True,
        random_state=int(round(individual[6]))
        )
    classifier.fit(X_train, y_train)

    acc = cross_val_score(classifier, X_train, y_train, cv=5)
    del classifier
    return -1 * acc.mean()

bounds = [ # Parameters tuned in https://github.com/miguelfzafra/Latest-News-Classifier/blob/master/0.%20Latest%20News%20Classifier/09.%20Report/Latest%20News%20Classifier.pdf plus random_state
    (1, 100), # n_estimators: int, default=100
    (1, 10), # max_depth: int, default=None
    (0, 1), # min_samples_split: int or float, default=2
    (0, 0.5), # min_samples_leaf: int or float, default=1
    (0, 1), # max_features: {“auto”, “sqrt”, “log2”}, int or float, default=”auto”
    (0, 1), # bootstrap: bool, default=True
    (0, 10) # random_state: int or RandomState, default=None
    ]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description='Decision Tree Hyperparameter tuning for software requirements categorization using Differential Evolution')
    ap.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    ap.add_argument('--np', dest='np', type=int, required=True, help='Population size')
    ap.add_argument('--max_gen', dest='max_gen', type=int, required=True, help='Genarations')
    ap.add_argument('--f', dest='f', type=float, required=True, help='Scale Factor')
    ap.add_argument('--cr', dest='cr', type=float, required=True, help='Crossover percentage')
    ap.add_argument('--datfile', dest='datfile', type=str, help='File where it will be save the score (result)')

    args = ap.parse_args()

    result = differential_evolution(fitness_func, bounds, disp=True, popsize=args.np, maxiter=args.max_gen, mutation=args.f, recombination=args.cr, strategy='rand1bin')
    
    print("Best individual: [n_estimators=%s, max_depth=%s, min_samples_split=%s, min_samples_leaf=%s, max_features=%s, bootstrap=%s, random_state=%s] f(x)=%s" % (
        int(round(result.x[0])),
        None if result.x[1] < 2 else int(round(result.x[1])),
        result.x[2],
        result.x[3],
        "auto" if result.x[4] < 0.5 else "sqrt",
        False if result.x[5] < 0.5 else True,
        int(round(result.x[6])),
        result.fun*(-100)))

    with open(args.datfile, 'w') as f:
        f.write(str(result.fun*(-100)))
