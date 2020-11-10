import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectPercentile, VarianceThreshold, f_classif
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline, make_union
from sklearn.svm import LinearSVC
from tpot.builtins import StackingEstimator
from sklearn.preprocessing import FunctionTransformer
from copy import copy

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: 0.8329095349860353
exported_pipeline = make_pipeline(
    make_union(
        FunctionTransformer(copy),
        make_pipeline(
            StackingEstimator(estimator=LinearSVC(C=0.1, dual=True, loss="squared_hinge", penalty="l2", tol=1e-05)),
            StackingEstimator(estimator=MLPClassifier(alpha=0.0001, learning_rate_init=0.001)),
            SelectPercentile(score_func=f_classif, percentile=18)
        )
    ),
    StackingEstimator(estimator=BernoulliNB(alpha=1.0, fit_prior=False)),
    StackingEstimator(estimator=KNeighborsClassifier(n_neighbors=81, p=1, weights="uniform")),
    VarianceThreshold(threshold=0.0005),
    MultinomialNB(alpha=0.1, fit_prior=True)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
