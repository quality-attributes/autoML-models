import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import MaxAbsScaler, Normalizer
from sklearn.tree import DecisionTreeClassifier
from tpot.builtins import StackingEstimator, ZeroCount
from sklearn.preprocessing import FunctionTransformer
from copy import copy

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: 0.8336787977583366
exported_pipeline = make_pipeline(
    make_union(
        make_pipeline(
            make_union(
                make_union(
                    Normalizer(norm="l2"),
                    MaxAbsScaler()
                ),
                StackingEstimator(estimator=DecisionTreeClassifier(criterion="entropy", max_depth=2, min_samples_leaf=11, min_samples_split=17))
            ),
            StackingEstimator(estimator=MLPClassifier(alpha=0.01, learning_rate_init=0.001)),
            SelectPercentile(score_func=f_classif, percentile=54)
        ),
        FunctionTransformer(copy)
    ),
    ZeroCount(),
    StackingEstimator(estimator=BernoulliNB(alpha=1.0, fit_prior=True)),
    MultinomialNB(alpha=0.01, fit_prior=True)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
