import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import PolynomialFeatures
from tpot.builtins import StackingEstimator
from sklearn.preprocessing import FunctionTransformer
from copy import copy

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: 0.8345643602435218
exported_pipeline = make_pipeline(
    make_union(
        make_pipeline(
            make_union(
                FunctionTransformer(copy),
                make_union(
                    SelectPercentile(score_func=f_classif, percentile=33),
                    FunctionTransformer(copy)
                )
            ),
            SelectPercentile(score_func=f_classif, percentile=61)
        ),
        FunctionTransformer(copy)
    ),
    PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
    StackingEstimator(estimator=LogisticRegression(C=1.0, dual=False, penalty="l2")),
    MultinomialNB(alpha=0.01, fit_prior=False)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
