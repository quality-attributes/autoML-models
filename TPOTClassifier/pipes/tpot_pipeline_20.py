import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from sklearn.preprocessing import FunctionTransformer
from copy import copy

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: 0.8326392221287445
exported_pipeline = make_pipeline(
    make_union(
        make_pipeline(
            make_union(
                make_union(
                    FunctionTransformer(copy),
                    StackingEstimator(estimator=RandomForestClassifier(bootstrap=True, criterion="entropy", max_features=0.35000000000000003, min_samples_leaf=1, min_samples_split=7, n_estimators=100))
                ),
                make_union(
                    FunctionTransformer(copy),
                    FunctionTransformer(copy)
                )
            ),
            SelectPercentile(score_func=f_classif, percentile=58)
        ),
        FunctionTransformer(copy)
    ),
    MultinomialNB(alpha=0.1, fit_prior=True)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
