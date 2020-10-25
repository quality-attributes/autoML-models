import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectFwe, SelectPercentile, f_classif
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: 0.8255806224382443
exported_pipeline = make_pipeline(
    SelectPercentile(score_func=f_classif, percentile=91),
    StackingEstimator(estimator=MLPClassifier(alpha=0.0001, learning_rate_init=0.001)),
    StackingEstimator(estimator=GradientBoostingClassifier(learning_rate=0.01, max_depth=7, max_features=0.6000000000000001, min_samples_leaf=15, min_samples_split=4, n_estimators=100, subsample=0.05)),
    SelectFwe(score_func=f_classif, alpha=0.035),
    MultinomialNB(alpha=0.1, fit_prior=False)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
