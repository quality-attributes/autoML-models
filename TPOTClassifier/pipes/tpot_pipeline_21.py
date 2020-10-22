import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from sklearn.preprocessing import FunctionTransformer
from copy import copy

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: 0.650104236770097
exported_pipeline = make_pipeline(
    make_union(
        make_union(
            StackingEstimator(estimator=make_pipeline(
                PCA(iterated_power=1, svd_solver="randomized"),
                GradientBoostingClassifier(learning_rate=0.01, max_depth=5, max_features=0.2, min_samples_leaf=9, min_samples_split=9, n_estimators=100, subsample=0.4)
            )),
            PCA(iterated_power=1, svd_solver="randomized")
        ),
        FunctionTransformer(copy)
    ),
    SelectPercentile(score_func=f_classif, percentile=45),
    BernoulliNB(alpha=0.1, fit_prior=False)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
