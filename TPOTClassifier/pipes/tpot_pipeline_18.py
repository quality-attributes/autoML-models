import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectFwe, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import MinMaxScaler
from tpot.builtins import OneHotEncoder, StackingEstimator

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: 0.6268918199700481
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=LogisticRegression(C=25.0, dual=False, penalty="l2")),
    MinMaxScaler(),
    OneHotEncoder(minimum_fraction=0.05, sparse=False, threshold=10),
    StackingEstimator(estimator=MultinomialNB(alpha=10.0, fit_prior=False)),
    StackingEstimator(estimator=BernoulliNB(alpha=1.0, fit_prior=True)),
    StackingEstimator(estimator=GradientBoostingClassifier(learning_rate=0.001, max_depth=5, max_features=0.6000000000000001, min_samples_leaf=9, min_samples_split=18, n_estimators=100, subsample=0.6500000000000001)),
    SelectFwe(score_func=f_classif, alpha=0.046),
    GaussianNB()
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
