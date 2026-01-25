import numpy as np
import statsmodels.api as sm
from statsmodels.base.model import GenericLikelihoodModel

class CauchyMLE(GenericLikelihoodModel):
    """Cauchy regression using statsmodels' GenericLikelihoodModel."""
    def nparams(self):
        return self.exog.shape[1] + 1
    def loglike(self, params):
        X, y = self.exog, self.endog
        k = X.shape[1]
        beta = params[:k]
        log_sigma = params[k]
        sigma = np.exp(log_sigma)
        r = y - X.dot(beta)
        return (-np.log(np.pi) - np.log(sigma) - np.log1p((r/sigma)**2)).sum()

def fit_cauchy_mle(X_df, y_series):
    X = sm.add_constant(X_df)
    y = y_series.to_numpy()
    ols = sm.OLS(y, X).fit()
    init = np.r_[ols.params, np.log(np.median(np.abs(ols.resid)) + 1e-6)]
    res = CauchyMLE(y, X).fit(start_params=init, method="bfgs", disp=False)
    k = X.shape[1]
    beta, sigma = res.params[:k], float(np.exp(res.params[k]))
    return res, beta, sigma
