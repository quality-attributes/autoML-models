
from tpot import TPOTClassifier
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import make_scorer

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

def g_mean(y_true, y_pred):
    return geometric_mean_score(y_true, y_pred)

my_custom_scorer = make_scorer(g_mean, greater_is_better=True)

pipeline_optimizer = TPOTClassifier(verbosity=2, scoring=my_custom_scorer, n_jobs=-2, log_file=open('logger.log', 'w'))

pipeline_optimizer.fit(X_train, y_train)
print(pipeline_optimizer.score(X_test, y_test))

pipeline_optimizer.export('tpot_exported_pipeline.py')