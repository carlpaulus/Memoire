from input_data import Analysis
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.tsa.tsatools import lagmat


class PortfolioConstruction(Analysis):
    BASIS = 100

    def __init__(self):
        super().__init__()
        self.get_component_data()
        self.get_component_prices()
        self.get_component_volumes()

        self.get_benchmark_data()
        self.get_benchmark_level()
        self.get_benchmark_volumes()
        self.get_benchmark_highs()
        self.get_benchmark_lows()

        self.set_calendar([self.benchmark_prices, self.component_prices])

        self.get_component_logreturns()
        self.get_component_volatilities()
        self.get_benchmark_logreturns()
        self.get_benchmark_volatility()

        self.weights = None
        self.portfolio_returns = None
        self.portfolio_basis_value = None
        self.benchmark_basis_value = None

    def benchmark_basis_calculation(self):
        bench_basis = (self.benchmark_prices / self.benchmark_prices[0])*PortfolioConstruction.BASIS
        self.benchmark_basis_value = bench_basis

    def comp_weights(self):
        raise NotImplementedError

    def portfolio_ret(self):
        raise NotImplementedError

    def portfolio_basis_calculation(self):
        raise NotImplementedError

    def bench_vs_index(self):
        raise NotImplementedError

    def compute_levels(self):
        self.comp_weights()
        self.portfolio_ret()
        self.portfolio_basis_calculation()
        # self.benchmark_basis_calculation()
        # self.bench_vs_index()


class EquallyWeighted(PortfolioConstruction):

    def __init__(self):
        super().__init__()

    def comp_weights(self):
        self.weights = np.ones((len(self.component_log_returns), len(self.PORTFOLIO_COMPOSITION))) * 1 / len(
            self.PORTFOLIO_COMPOSITION)

    def portfolio_ret(self):
        self.portfolio_returns = (self.weights * self.component_log_returns.values).sum(axis=1)

    def portfolio_basis_calculation(self):
        final_calendar = self.calendar
        temp_final_mat = np.c_[
            self.portfolio_returns, lagmat(self.portfolio_returns, maxlag=len(self.portfolio_returns) - 1)]
        pf_basis_value = np.where(temp_final_mat == 0, 1, temp_final_mat).prod(axis=1) * PortfolioConstruction.BASIS
        pf_basis_value = pd.DataFrame(np.r_[PortfolioConstruction.BASIS, pf_basis_value], index=final_calendar,
                                      columns=["Index Value"])
        self.portfolio_basis_value = pf_basis_value

    def bench_vs_index(self):
        comp_levels = self.component_prices[self.component_prices.index.isin(self.calendar)]
        pf_level = (self.weights * comp_levels[1:]).sum(axis=1)
        pf_basis = pd.DataFrame((pf_level / pf_level[0]) * PortfolioConstruction.BASIS)

        plt.plot(self.benchmark_basis_value, color='orange', label='Bench Value')
        plt.plot(pf_basis, color='blue', label='Portfolio Value')
        plt.legend(loc='upper left', fontsize=12)
        plt.title('Bench vs Portfolio')
        plt.show()


class ReturnMomentum(PortfolioConstruction):

    def __init__(self):
        super().__init__()

    def comp_weights(self, criteria: int = 3):
        self.weights = self.component_log_returns.apply(
            lambda x: (2 * (x > x.median()) - 1) * 1 / criteria, axis=0).shift(1).fillna(0)

    def portfolio_ret(self):
        self.portfolio_returns = (pd.DataFrame(self.weights) * self.component_log_returns).sum(axis=1) + 1

    def portfolio_basis_calculation(self):
        final_calendar = self.calendar
        temp_final_mat = np.c_[
            self.portfolio_returns, lagmat(self.portfolio_returns, maxlag=len(self.portfolio_returns) - 1)]
        pf_basis_value = np.where(temp_final_mat == 0, 1, temp_final_mat).prod(axis=1) * PortfolioConstruction.BASIS
        pf_basis_value = pd.DataFrame(np.r_[PortfolioConstruction.BASIS, pf_basis_value], index=final_calendar,
                                      columns=["Index Value"])
        self.portfolio_basis_value = pf_basis_value

    def bench_vs_index(self):
        comp_levels = self.component_prices[self.component_prices.index.isin(self.calendar)]
        # pf_level = (1+(self.weights * comp_levels).sum(axis=1)).cumprod()
        pf_level = (self.weights * comp_levels[1:]).sum(axis=1)
        pf_basis = pd.DataFrame((pf_level / pf_level[0]) * PortfolioConstruction.BASIS)

        plt.plot(self.benchmark_basis_value, color='orange', label='Bench Value')
        plt.plot(pf_basis, color='blue', label='Portfolio Value')
        plt.legend(loc='upper left', fontsize=12)
        plt.title('Bench vs Portfolio')
        plt.show()


class LowVolMomentum(PortfolioConstruction):

    def __init__(self):
        super().__init__()

    def comp_weights(self, criteria: int = 3):
        self.weights = self.component_volatilities.apply(
            lambda x: (2 * (x < x.median()) - 1) * 1 / criteria, axis=1).shift(1).fillna(0)

    def portfolio_ret(self):
        self.portfolio_returns = (pd.DataFrame(self.weights) * self.component_log_returns).sum(axis=1) + 1

    def portfolio_basis_calculation(self):
        final_calendar = self.calendar
        temp_final_mat = np.c_[
            self.portfolio_returns, lagmat(self.portfolio_returns, maxlag=len(self.portfolio_returns) - 1)]
        pf_basis_value = np.where(temp_final_mat == 0, 1, temp_final_mat).prod(axis=1) * PortfolioConstruction.BASIS
        pf_basis_value = pd.DataFrame(np.r_[PortfolioConstruction.BASIS, pf_basis_value], index=final_calendar,
                                      columns=["Index Value"])
        self.portfolio_basis_value = pf_basis_value

    def bench_vs_index(self):
        comp_levels = self.component_prices[self.component_prices.index.isin(self.calendar)]
        pf_level = (self.weights * comp_levels[1:]).sum(axis=1)
        pf_basis = pd.DataFrame((pf_level / pf_level[0]) * PortfolioConstruction.BASIS)

        plt.plot(self.benchmark_basis_value, color='orange', label='Bench Value')
        plt.plot(pf_basis, color='blue', label='Portfolio Value')
        plt.legend(loc='upper left', fontsize=12)
        plt.title('Bench vs Portfolio')
        plt.show()


if __name__ == '__main__':
    ew = EquallyWeighted()
    # rm = ReturnMomentum()
    # lv = LowVolMomentum()

    ew.compute_levels()


