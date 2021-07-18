from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import scipy

class LogisticRegressionWithPValues(LogisticRegression):

    def fit(self, X, y):
        super().fit(X, y)

        denom = (2.0 * (1.0 + np.cosh(self.decision_function(X))))
        denom = np.tile(denom,(X.shape[1],1)).T
        F_ij = np.dot((X / denom).T,X) ## Fisher Information Matrix
        Cramer_Rao = np.linalg.inv(F_ij) ## Inverse Information Matrix
        sigma_estimates = np.sqrt(np.diagonal(Cramer_Rao))
        z_scores = self.coef_[0] / sigma_estimates # z-score for eaach model coefficient
        p_values = [scipy.stats.norm.sf(abs(x)) * 2 for x in z_scores] ### two tailed test for p-values

        self.p_values = p_values
        try:
            self.feature_names = X.columns
        except AttributeError:
            self.feature_names = [f'X{i}' for i in range(X.shape[1])]

        return self

    def get_summary_table(self, feature_names=None):

        if feature_names is None:
            feature_names = self.feature_names

        # Same as above.
        summary_table = pd.DataFrame(columns = ['feature_name'], data = feature_names)
        summary_table['coefficient'] = np.transpose(self.coef_)
        summary_table.index = summary_table.index + 1
        summary_table.loc[0] = ['intercept', self.intercept_[0]]
        summary_table = summary_table.sort_index()
        summary_table['p_value'] = np.append(np.nan, np.array(self.p_values))
        

        return summary_table