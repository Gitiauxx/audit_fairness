from fairness.algorithms.feldman import FeldmanAlgorithm as fa
from fairness.algorithms.zafar import ZafarAlgorithm as za
from fairness.algorithms.baseline.SVM import SVM
from fairness.algorithms.baseline.LogisticRegression import LogisticRegression as LR
import pandas as pd
import numpy as np
from aufair import auditing_data as ad
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from fairlearn import moments
from fairlearn import classred as red


class audit(object):

    def __init__(self, classifier, auditor, auditing_features, protected, yname):
        self.classifier = classifier
        self.auditor = auditor
        self.auditing_features = auditing_features
        self.protected = protected
        self.yname = yname

    def standardize(self, data):
        self.standardized_features = []
        for var in self.auditing_features:
            data = data[~np.isnan(data[var])]
            data[var + '_s'] = (data[var] - data[var].mean()) / np.sqrt(data[var].var())
            self.standardized_features.append(var + '_s')

        data['attr'] = data[self.protected[0]]
        data['outcome'] = data[self.yname]
        self.data = data

    def get_violations(self, nboot=1):
        data = self.data
        audit = ad.detector_data(self.auditor, data, self.protected, self.yname, lw=0.005, niter=100, stepsize=0.01)
        audit.get_y()
        violations = pd.DataFrame()
        resu = np.zeros((nboot, 2))

        for iter in range(nboot):
            delta, test_audited = audit.get_violation(self.standardized_features)
            try:
                resu[iter, 0] = (test_audited[(test_audited.predicted == 1) & (test_audited.attr == 1)].outcome + 1).mean() / \
                         (test_audited[(test_audited.predicted == 1) & (test_audited.attr == -1)].outcome + 1).mean()
            except:
                resu[iter, 0] = np.nan
            resu[iter, 1] = self.tpp(test_audited)

            # collect worst-case violations
            for v in self.auditing_features:
                violations.loc[iter, v + '_1'] = test_audited[(test_audited.predicted == 1) & (test_audited.attr == 1)][v].mean()
                violations.loc[iter, v + '_0'] = test_audited[(test_audited.predicted == 1) & (test_audited.attr == -1)][v].mean()

            violations.loc[iter, 'attr_1'] = len(test_audited[(test_audited.predicted == 1) & (test_audited.attr == 1)])
            violations.loc[iter, 'attr_0'] = len(test_audited[(test_audited.predicted == 1) & (test_audited.attr == -1)])
            violations.loc[iter, 'delta'] = delta

        resu = resu[~np.isnan(resu[:, 0])]
        delta_m = resu[:, 0].mean()
        delta_sd = np.sqrt(resu[:, 0].var())
        tp = resu[:, 1].mean()
        tp_sd = np.sqrt(resu[:, 1].var())

        return violations, delta_m, delta_sd, tp, tp_sd

    def tpp(self, data):
        tpp = len(data[(data.attr == 1) & (data.outcome == 1) & (data.Y == 1)]) / \
            len(data[(data.attr == 1) & (data.Y == 1)])
        tpp = tpp / (len(data[(data.attr == -1) & (data.outcome == 1) & (data.Y == 1)]) / \
                     len(data[(data.attr == -1) & (data.Y == 1)]))
        return tpp
