from sklearn.utils.estimator_checks import check_estimator

from mondrian_forest import MondrianForestRegressor
from mondrian_forest import MondrianTreeRegressor

mondrian_estimators = [MondrianTreeRegressor, MondrianForestRegressor]

def check_mondrian_estimator(est):
    check_estimator(est)


def test_mondrian_estimator():
    for est in mondrian_estimators:
        check_mondrian_estimator(est)
