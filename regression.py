import pandas as pd
from factors import Combinations

from portfolios import EquallyWeighted, ReturnMomentum, LowVolMomentum
from dataclasses import dataclass
from statsmodels.api import OLS
from statsmodels.regression.linear_model import RegressionResults, RegressionResultsWrapper
from statsmodels.stats.diagnostic import het_white, spec_white, het_breuschpagan, acorr_ljungbox
import numpy as np
import matplotlib.pyplot as plt


class LinearEstimator(Combinations):
    _bias = "Bias"

    @dataclass(init=False)
    class LinearRegressionResult:
        t_stats: object = None
        p_values: object = None
        coefficients: object = None
        residuals: object = None
        fitted: object = None

        @classmethod
        def from_model(cls, result: RegressionResults):

            _obj = cls()
            object.__setattr__(_obj, "result", result)
            _obj.coefficients = result.params
            _obj.t_stats = result.tvalues
            _obj.p_values = result.pvalues
            _obj.residuals = result.resid
            _obj.fitted = result.fittedvalues
            return _obj

        def to_csv(self, file_name: str):
            path = "/Users/carlpaulus/OneDrive - EDHEC/Documents/travail/EDHEC/Cours BBA/BBA4/Mémoire/Regression results/"
            to_export = {key: value for key, value in vars(self).items() if key != "result"}
            pd.DataFrame(to_export).to_csv(path + file_name + ".csv")

        def print(self):
            try:
                res = self.__getattribute__("result")
                if isinstance(res, RegressionResultsWrapper):
                    print(res.summary())
            except AttributeError:
                raise TypeError("No result in LinearResultRegression")

    def __init__(self, hasconst: bool = True):
        super().__init__()
        self._hasconst = hasconst
        self._model = None
        self._result = None

    @property
    def model(self) -> OLS:
        return self._model

    @property
    def result(self) -> LinearRegressionResult:
        return self._result

    def __call__(self, endog: np.ndarray, exog: np.ndarray):
        if len(endog) == len(exog):

            if self._hasconst:
                exog[self._bias] = 1

            self._model = OLS(endog, exog)
            self._result = self.LinearRegressionResult.from_model(self.model.fit())

        return self.result


# calculate results for each combination (all factors + carhart momentum + RMW + CMA) and FF3, FF4, FF5
class Calculator(LinearEstimator):

    def __init__(self):
        super().__init__()
        self.ew = EquallyWeighted()
        # self.rm = ReturnMomentum()
        # self.lv = LowVolMomentum()

        self.ew.compute_levels()
        # self.rm.compute_levels()
        # self.lv.compute_levels()

        self.estimator = LinearEstimator()
        self.factors = Combinations().five_factor_combinations()

    def compute_results(self):
        self.estimator(self.ew.portfolio_returns, self.three_factors).print()

        for factor in self.factors:
            print(str(self.market) + ' ' + str(factor.columns[3:5][0]) + ' ' + str(factor.columns[3:5][1]))
            result = self.estimator(self.ew.portfolio_returns, factor)
            # result.to_csv(str(self.market) + '_' + str(factor.columns[3:5][0]) + '_' + str(factor.columns[3:5][1]))
            result.print()

        # return self

        # the way to run all regressions on all portfolios :
        # portfolios_list = [self.ew, self.rm, self.lv]
        # for portfolio in portfolios_list:
        #     for factor in self.factors:
        #         result = self.estimator(portfolio.portfolio_returns, factor)
        #         # result.to_csv("result " + portfolio.__name__ + " " + factor.__name__)
        #         result.print()


# def autocorr_test(
#         model_name: str,
#         residuals: pd.Series,
#         plot: bool = True,
#         maxlags=10
# ):
#     if plot:
#         plt.acorr(residuals, maxlags=maxlags)
#         plt.savefig("auto_corr_" + f"{model_name}" + "png")
#     ljungbox = acorr_ljungbox(residuals, lags=[maxlags], return_df=True)
#     return ljungbox.to_dict() if isinstance(ljungbox, pd.DataFrame) else ljungbox


if __name__ == '__main__':
    calculator = Calculator()
    calculator.compute_results()

    residuals = pd.Series(calculator.estimator.result.residuals)
    fitted_values = calculator.estimator.result.fitted

    # homoscedascticity test
    # print(autocorr_test(model_name="ols", residuals=residuals))

    fig = plt.figure(num=0, figsize=(10, 8))
    plot_opts = dict(linestyle="None", marker="o", color="black", markerfacecolor="None")
    _ = fig.add_subplot(2, 2, 1).plot(residuals, **plot_opts)
    _ = plt.title("Regression residuals over time")
    _ = fig.add_subplot(2, 2, 2).plot(fitted_values, residuals, **plot_opts)
    _ = plt.title("Residuals vs Fitted values")
    _ = plt.tight_layout()
    plt.show()

    labels = ['Lagrange Multiplier statistic',
              'LM-Test p-value',
              'F-Statistic',
              'F-Test p-value']
    white_test_het = het_white(residuals, calculator.three_factors)
    print(dict(zip(labels, white_test_het)))

