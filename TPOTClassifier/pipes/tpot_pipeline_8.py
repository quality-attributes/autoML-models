import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFwe, VarianceThreshold, f_classif
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import Binarizer, MinMaxScaler
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: 0.8285375806358702
exported_pipeline = make_pipeline(
    make_union(
        MinMaxScaler(),
        make_pipeline(
            SelectFwe(score_func=f_classif, alpha=0.048),
            Binarizer(threshold=0.05)
        )
    ),
    StackingEstimator(estimator=MLPClassifier(alpha=0.01, learning_rate_init=0.001)),
    VarianceThreshold(threshold=0.0001),
    MultinomialNB(alpha=0.1, fit_prior=True)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
