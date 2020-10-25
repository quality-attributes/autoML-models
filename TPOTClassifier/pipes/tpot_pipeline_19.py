import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline, make_union
from sklearn.svm import LinearSVC
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: 0.8237428874016649
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=LinearSVC(C=0.1, dual=True, loss="squared_hinge", penalty="l2", tol=0.01)),
    StackingEstimator(estimator=LinearSVC(C=0.001, dual=False, loss="squared_hinge", penalty="l1", tol=1e-05)),
    StackingEstimator(estimator=MultinomialNB(alpha=100.0, fit_prior=False)),
    MLPClassifier(alpha=0.0001, learning_rate_init=0.001)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
