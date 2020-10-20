import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tpot.builtins import StackingEstimator, ZeroCount

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: 0.6591370881372015
exported_pipeline = make_pipeline(
    StandardScaler(),
    StackingEstimator(estimator=RandomForestClassifier(bootstrap=False, criterion="entropy", max_features=0.05, min_samples_leaf=19, min_samples_split=15, n_estimators=100)),
    SelectPercentile(score_func=f_classif, percentile=59),
    MinMaxScaler(),
    ZeroCount(),
    ZeroCount(),
    SelectPercentile(score_func=f_classif, percentile=79),
    StackingEstimator(estimator=ExtraTreesClassifier(bootstrap=False, criterion="entropy", max_features=0.35000000000000003, min_samples_leaf=19, min_samples_split=9, n_estimators=100)),
    BernoulliNB(alpha=0.1, fit_prior=False)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
