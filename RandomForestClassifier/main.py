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
        criterion="gini" if individual[1] < 0.5 else "entropy",
        max_depth=None if individual[2] < 2 else int(round(individual[2])),
        min_samples_split=individual[3],
        min_samples_leaf=individual[4],
        min_weight_fraction_leaf=individual[5],
        max_features="auto" if individual[6] < 0.5 else "sqrt",
        max_leaf_nodes=None if individual[7] < 1 else int(round(individual[7])),
        min_impurity_decrease=individual[8],
        bootstrap=False if individual[9] < 0.5 else True,
        random_state=int(round(individual[10]))
        )
    classifier.fit(X_train, y_train)

    acc = cross_val_score(classifier, X_train, y_train, cv=5)
    del classifier
    return -1 * acc.mean()

bounds = [
    (1, 1000), # n_estimators: int, default=100
    (1, 100), # criterion: {"gini", "entropy"}, default="gini"
    (1, 100), # max_depth: int, default=None
    (0, 1), # min_samples_split: int or float, default=2
    (0, 0.5), # min_samples_leaf: int or float, default=1
    (0, 0.5), # min_weight_fraction_leaf: float, default=0.0
    (0, 1), # max_features: {“auto”, “sqrt”, “log2”}, int or float, default=”auto”
    (0, 100), # max_leaf_nodes: int, default=None
    (0, 10), # min_impurity_decrease: float, default=0.0
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
    
    print("Best individual: [n_estimators=%s, criterion=%s, max_depth=%s, min_samples_split=%s, min_samples_leaf=%s,  min_weight_fraction_leaf=%s, max_features=%s, max_leaf_nodes=%s, min_impurity_decrease=%s, bootstrap=%s, random_state=%s] f(x)=%s" % (
        int(round(result.x[0])),
        "gini" if result.x[1] < 0.5 else "entropy",
        None if result.x[2] < 2 else int(round(result.x[2])),
        result.x[3],
        result.x[4],
        result.x[5],
        "auto" if result.x[6] < 0.5 else "sqrt",
        None if result.x[7] < 1 else int(round(result.x[7])),
        result.x[8],
        False if result.x[9] < 0.5 else True,
        int(round(result.x[10])),
        result.fun*(-100)))

    with open(args.datfile, 'w') as f:
        f.write(str(result.fun*(-100)))
