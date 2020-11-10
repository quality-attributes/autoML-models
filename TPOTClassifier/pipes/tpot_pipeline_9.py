import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import RFE, SelectPercentile, f_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.svm import LinearSVC
from tpot.builtins import StackingEstimator
from sklearn.preprocessing import FunctionTransformer
from copy import copy

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: 0.8301746327045152
exported_pipeline = make_pipeline(
    make_union(
        make_pipeline(
            SelectPercentile(score_func=f_classif, percentile=65),
            PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
            StackingEstimator(estimator=RandomForestClassifier(bootstrap=True, criterion="gini", max_features=0.4, min_samples_leaf=17, min_samples_split=6, n_estimators=100)),
            MinMaxScaler()
        ),
        FunctionTransformer(copy)
    ),
    RFE(estimator=ExtraTreesClassifier(criterion="gini", max_features=0.7000000000000001, n_estimators=100), step=0.8),
    LinearSVC(C=0.1, dual=True, loss="squared_hinge", penalty="l2", tol=0.001)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
