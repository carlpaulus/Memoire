from typing import Union, List
import yfinance as yf
import numpy as np
import pandas as pd


class Analysis:

    """ Choose the market: europe, america or japan """

    BENCHMARK_TICKER = None
    PORTFOLIO_COMPOSITION = None

    def __init__(self, market='europe'):

        if market == 'europe':
            Analysis.BENCHMARK_TICKER = "^STOXX50E"  # Eurostoxx 50
            Analysis.PORTFOLIO_COMPOSITION = list(['ASML.AS', 'MC.PA', 'SAP.DE', 'LIN.DE', 'SIE.DE', 'SAN.PA', 'OR.PA',
                                                   'SU.PA', 'ALV.DE', 'AIR.PA', 'AI.PA', 'BNP.PA', 'DAI.DE', 'BAS.DE',
                                                   'DTE.DE', 'SAN.MC', 'DPW.DE', 'DG.PA', 'ENEL.MI', 'IBE.MC', 'ADS.DE',
                                                   'EL.PA', 'CS.PA', 'IFX.DE', 'INGA.AS', 'BAYN.DE', 'KER.PA', 'ISP.MI',
                                                   'SAF.PA', 'ABI.BR', 'RI.PA', 'BN.PA', 'BBVA.MC', 'ITX.MC', 'PHIA.AS',
                                                   'STLA.MI', 'VOW.DE', 'MUV2.DE', 'CRG.IR', 'FLTR.IR', 'AD.AS',
                                                   'ENI.MI', 'DB1.DE', 'BMW.DE', 'KNEBV.HE'])

        elif market == 'america':
            Analysis.BENCHMARK_TICKER = "^NDX"  # Nasdaq 100
            Analysis.PORTFOLIO_COMPOSITION = list(['ATVI', 'ADBE', 'AMD', 'ALGN', 'GOOG', 'AMZN', 'AEP', 'AMGN', 'BIIB',
                                                   'ANSS', 'AAPL', 'AMAT', 'ASML', 'TEAM', 'ADSK', 'ADP', 'BIDU', 'ADI',
                                                   'CDNS', 'CDW', 'CERN', 'CHTR', 'CHKP', 'CTAS', 'CSCO', 'CTSH', 'EA',
                                                   'CPRT', 'COST', 'CSX', 'DXCM', 'DOCU', 'DLTR', 'KDP', 'EBAY', 'ILMN',
                                                   'EXC', 'FB', 'FAST', 'FISV', 'FOX', 'FOXA', 'GILD', 'IDXX', 'CMCSA',
                                                   'INCY', 'INTC', 'INTU', 'ISRG', 'JD', 'KLAC', 'KHC', 'LRCX', 'LULU',
                                                   'MAR', 'MRVL', 'MTCH', 'MELI', 'MCHP', 'MU', 'MSFT', 'MRNA', 'MDLZ',
                                                   'MNST', 'NTES', 'NFLX', 'NVDA', 'NXPI', 'ORLY', 'OKTA', 'PCAR',
                                                   'PYPL', 'PTON', 'PEP', 'PDD', 'BKNG', 'QCOM', 'REGN', 'ROST', 'SGEN',
                                                   'SIRI', 'SWKS', 'SPLK', 'SBUX', 'SNPS', 'TSLA', 'TXN', 'TMUS',
                                                   'VRSK', 'VRTX', 'WBA', 'WDAY', 'XEL', 'PAYX', 'VRSN'])

        elif market == 'japan':
            Analysis.BENCHMARK_TICKER = "^N225"  # Nikkei 225
            Analysis.PORTFOLIO_COMPOSITION = list(['4151.T', '4502.T', '4503.T', '4506.T', '4507.T', '4519.T', '4523.T',
                                                   '4568.T', '4578.T', '6479.T', '6501.T', '6503.T', '6504.T', '6506.T',
                                                   '6645.T', '6674.T', '6701.T', '6702.T', '6703.T', '6724.T', '6752.T',
                                                   '6753.T', '6758.T', '6762.T', '6770.T', '6841.T', '6857.T', '6861.T',
                                                   '6902.T', '6952.T', '6954.T', '6971.T', '6976.T', '6981.T', '7735.T',
                                                   '7751.T', '7752.T', '8035.T', '7201.T', '7202.T', '7203.T', '7205.T',
                                                   '7211.T', '7261.T', '7267.T', '7269.T', '7270.T', '7272.T', '4543.T',
                                                   '4902.T', '7731.T', '7733.T', '7762.T', '9432.T', '9433.T', '9434.T',
                                                   '9613.T', '9984.T', '7186.T', '8303.T', '8304.T', '8306.T', '8308.T',
                                                   '8309.T', '8316.T', '8331.T', '8354.T', '8355.T', '8411.T', '8253.T',
                                                   '8697.T', '8601.T', '8604.T', '8628.T', '8630.T', '8725.T', '8750.T',
                                                   '8766.T', '8795.T', '1332.T', '1333.T', '2002.T', '2269.T', '2282.T',
                                                   '2501.T', '2502.T', '2503.T', '2531.T', '2801.T', '2802.T', '2871.T',
                                                   '2914.T', '3086.T', '3099.T', '3382.T', '8233.T', '8252.T', '8267.T',
                                                   '9983.T', '2413.T', '2432.T', '3659.T', '4324.T', '4689.T', '4704.T',
                                                   '4751.T', '4755.T', '6098.T', '6178.T', '7974.T', '9602.T', '9735.T',
                                                   '9766.T', '1605.T', '3101.T', '3103.T', '3401.T', '3402.T', '3861.T',
                                                   '3863.T', '3405.T', '3407.T', '4004.T', '4005.T', '4021.T', '4042.T',
                                                   '4043.T', '4061.T', '4063.T', '4183.T', '4188.T', '4208.T', '4452.T',
                                                   '4631.T', '4901.T', '4911.T', '6988.T', '5019.T', '5020.T', '5101.T',
                                                   '5108.T', '5201.T', '5202.T', '5214.T', '5232.T', '5233.T', '5301.T',
                                                   '5332.T', '5333.T', '5401.T', '5406.T', '5411.T', '5541.T', '3436.T',
                                                   '5703.T', '5706.T', '5707.T', '5711.T', '5713.T', '5714.T', '5801.T',
                                                   '5802.T', '5803.T', '2768.T', '8001.T', '8002.T', '8015.T', '8031.T',
                                                   '8053.T', '8058.T', '1721.T', '1801.T', '1802.T', '1803.T', '1808.T',
                                                   '1812.T', '1925.T', '1928.T', '1963.T', '5631.T', '6103.T', '6113.T',
                                                   '6301.T', '6302.T', '6305.T', '6326.T', '6361.T', '6367.T', '6471.T',
                                                   '6472.T', '6473.T', '7004.T', '7011.T', '7013.T', '7003.T', '7012.T',
                                                   '7832.T', '7911.T', '7912.T', '7951.T', '3289.T', '8801.T', '8802.T',
                                                   '8804.T', '8830.T', '9001.T', '9005.T', '9007.T', '9008.T', '9009.T',
                                                   '9020.T', '9021.T', '9022.T', '9062.T', '9064.T', '9101.T', '9104.T',
                                                   '9107.T', '9202.T', '9301.T', '9501.T', '9502.T', '9503.T', '9531.T',
                                                   '9532.T'])

        else:
            raise AttributeError('not applicable market')

        self.market = market
        self.start = "2013-07-31"
        self.end = "2021-07-31"
        self.calendar = None

        self.component_data = None
        self.component_prices = None
        self.component_volumes = None
        self.component_log_returns = None
        self.component_volatilities = None

        self.benchmark_data = None
        self.benchmark_prices = None
        self.benchmark_volumes = None
        self.benchmark_high = None
        self.benchmark_low = None
        self.benchmark_log_returns = None
        self.benchmark_volatility = None

    """ Calendar """

    def __compute_calendar(self, series: List[Union[pd.Series, pd.DataFrame]]):
        """
        Parameters
        ----------
        series : List[Union[pd.Series, pd.DataFrame]]
            series is a list of pandas Series or DataFrame

        Returns
        -------
            a list of sorted timestamp
        """
        return sorted(set.intersection(*list(map(lambda x: set(x.index.tolist()), series))))

    def set_calendar(self, series: List[Union[pd.Series, pd.DataFrame]]):
        self.calendar = self.__compute_calendar(series)

    """ Component Information """

    def get_component_data(self):
        all_stocks = ""
        for i in range(len(Analysis.PORTFOLIO_COMPOSITION)):
            all_stocks = str(all_stocks) + " " + str(Analysis.PORTFOLIO_COMPOSITION[i])
        temp_data = yf.download(all_stocks, start=self.start, end=self.end)
        self.component_data = temp_data.ffill().dropna().drop('Adj Close', axis=1)

    def get_component_prices(self):
        self.component_prices = self.component_data["Close"]

    def get_component_volumes(self):
        self.component_volumes = self.component_data["Volume"]

    def get_component_logreturns(self):
        if self.component_prices is not None:
            temp_log_returns = np.log((self.component_prices / (self.component_prices.shift(1)))).dropna()
            self.component_log_returns = temp_log_returns[temp_log_returns.index.isin(self.calendar)]
        else:
            raise TypeError(f'self.component_prices cannot be NoneType, it must be an instance of {type(pd.DataFrame)}')

    def get_component_volatilities(self):
        self.component_volatilities = (self.component_log_returns - self.component_log_returns.mean()) ** 0.5

    """ Index Information """

    def get_benchmark_data(self):
        self.benchmark_data = yf.download(Analysis.BENCHMARK_TICKER, start=self.start, end=self.end).drop('Adj Close', axis=1)

    def get_benchmark_level(self):
        self.benchmark_prices = self.benchmark_data["Close"]

    def get_benchmark_volumes(self):
        self.benchmark_volumes = self.benchmark_data["Volume"]

    def get_benchmark_highs(self):
        self.benchmark_high = self.benchmark_data['High']

    def get_benchmark_lows(self):
        self.benchmark_low = self.benchmark_data['Low']

    def get_benchmark_logreturns(self):
        if self.benchmark_prices is not None:
            temp_log_returns = np.log((self.benchmark_prices / (self.benchmark_prices.shift(1)))).dropna()
            self.benchmark_log_returns = temp_log_returns[temp_log_returns.index.isin(self.calendar)]
        else:
            raise TypeError(f'self.benchmark_prices cannot be NoneType, it must be an instance of {type(pd.DataFrame)}')

    def get_benchmark_volatility(self):
        self.benchmark_volatility = (self.benchmark_log_returns - self.benchmark_log_returns.mean()) ** 0.5


if __name__ == '__main__':
    print(Analysis().get_component_logreturns())
