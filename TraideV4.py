import numpy as np
from numpy import random
import yfinance as yf
from networkclass import *
import matplotlib.pyplot as plt

class Stock:

    def __init__(self, *, symbol):
        self.ticker = yf.Ticker(symbol)

    def load_price_data(self, interval = "1d", period = "max"):
        self.stock_data = self.ticker.history(interval=interval, period=period)
        self.oprices = np.array(self.stock_data["Open"])
        self.cprices = np.array(self.stock_data["Close"])
        self.yesterday_prices = self.cprices[:-1]
        self.tomorrow_prices = self.cprices[1:]
        self.volumes = np.array(self.stock_data["Volume"])
        self.opens = np.array(self.stock_data["Open"])
        self.highs = np.array(self.stock_data["High"])
        self.lows = np.array(self.stock_data["Low"])

    def load_indicators(self, interval = "1d", period = "max", sma_period = 15, ema_period = 14, macd_fast_period = 12, macd_slow_period = 26, obvslope_period = 5, rsi_period = 14):
        if not hasattr(self, "stock_data"):
            self.load_price_data(interval, period)

        all_periods = [sma_period, ema_period, macd_fast_period, macd_slow_period, obvslope_period, rsi_period]
        longest_period = max(all_periods)

        self.smas = self.calc_smas(sma_period)
        self.emas = self.calc_emas(ema_period)
        self.macds = self.calc_macds(macd_slow_period, macd_fast_period)
        self.obvslopes = self.calc_obvslopes(obvslope_period)
        self.rsis = self.calc_rsis(rsi_period)
        self.directions = (self.cprices[longest_period:] - self.oprices[longest_period:] > 0) * 1

        self.oprices /= np.max(self.oprices)
        self.smas /= np.max(self.smas)
        self.emas /= np.max(self.emas)
        self.macds /= np.max(self.macds)
        self.obvslopes /= np.max(self.obvslopes)
        self.rsis /= np.max(self.rsis)

        self.network_input_data = []
        self.network_output_data = []

        total_intervals = len(self.cprices)
        intervals = total_intervals - longest_period

        #start at 2 to account for indexing and no tomorrow price for current day
        for interval in range(1, intervals):
            self.network_input_data.append([self.oprices[-interval], 
                                            self.macds[-interval], 
                                            self.obvslopes[-interval], 
                                            self.rsis[-interval]])
            
            self.network_output_data.append(self.directions[-interval])

        self.network_input_data = np.flip(np.array(self.network_input_data), axis = 0)
        self.network_output_data = np.flip(np.array(self.network_output_data), axis = 0)

    def calc_smas(self, period = 15):
        smas = []

        prices = len(self.cprices) - period

        for price in range(prices):
            sma = np.mean(self.cprices[price:price + period])
            smas.append(sma)

        return np.array(smas)

    def calc_emas(self, period = 14):
        prices = len(self.cprices) - period
        initial_ema = np.mean(self.cprices[:period])

        emas = [initial_ema]

        smoothing_factor = 2 / (1 + period)

        for price in range(1, prices):
            ema = self.cprices[price + period] * smoothing_factor + emas[price - 1] * (1 - smoothing_factor)
            emas.append(ema)

        return np.array(emas)

    def calc_macds(self, slow_period = 26, fast_period = 12):
        macds = []

        slow_emas = self.calc_emas(slow_period)
        fast_emas = self.calc_emas(fast_period)

        prices = len(self.cprices) - slow_period

        for price in range(prices):
            macd = fast_emas[price + slow_period - fast_period] - slow_emas[price]
            macds.append(macd)

        return np.array(macds)

    def calc_obvslopes(self, period = 5):
        initial_volume = self.volumes[0]

        obvs = [initial_volume]

        volumes = len(self.volumes) - 1

        for volume in range(volumes):
            obv = obvs[volume] + self.volumes[volume + 1] * ((self.cprices[volume + 1] > self.cprices[volume]) * 1. - .5) * 2
            obvs.append(obv)

        obvs_period_shifted = np.array(obvs[:-period])
        obvs_shifted = np.array(obvs[period - 1:-1])
        obvs = np.array(obvs[period:])

        obvslopes = (obvs - obvs_shifted) / (obvs - obvs_period_shifted + 1)

        return obvslopes

    def calc_rsis(self, period = 14):
        price_changes = self.tomorrow_prices - self.yesterday_prices

        initial_avg_gain = np.sum(price_changes[:period] * (price_changes[:period] > 0)) / period
        initial_avg_loss = -np.sum(price_changes[:period] * (price_changes[:period] < 0)) / period

        avg_gains = [initial_avg_gain]
        avg_losses = [initial_avg_loss]

        initial_rsi = 100 - 100 / (1 + initial_avg_gain / initial_avg_loss)

        rsis = [initial_rsi]

        prices = len(self.cprices) - period

        for price in range(prices):
            avg_gains.append((avg_gains[price] * (period - 1) + price_changes[price + period - 1] * (price_changes[price + period - 1] > 0)) / period)
            avg_losses.append((avg_losses[price] * (period - 1) - price_changes[price + period - 1] * (price_changes[price + period - 1] < 0)) / period)
            rsis.append(100 - 100 / (1 + avg_gains[price + 1] / avg_losses[price + 1]))

        return np.array(rsis)


stock_tsla = Stock(symbol = "btc-usd")
stock_tsla.load_indicators(interval = "1d", ema_period = 14)

X = stock_tsla.network_input_data
y = stock_tsla.network_output_data

keys = list(range(len(y)))
np.random.shuffle(keys)

#y = y.reshape(len(y), 1)

later_X = X.copy()

X = X[keys]
y = y[keys]

X_test = X[-500:]
y_test = y[-500:]

X = X[:-500]
y = y[:-500]

randomized_prices = (stock_tsla.oprices[27:])[keys]

data = y.reshape(-1) * randomized_prices[:-500]

model = Model()

model.add(Layer_Dense(4, 32))
model.add(Activation_ReLU())
model.add(Layer_Dense(32, 32))
model.add(Activation_ReLU())
model.add(Layer_Dense(32, 2))
model.add(Activation_Softmax())


model.set(loss = Loss_CategoricalCrossEntropy(), optimizer = Optimizer_Adam(learning_rate = 1e-4, decay = 1e-2), accuracy = Accuracy_Categorical())

model.finalize()

og_data = model.output_layer_activation.predictions(model.forward(X, training = False)) * stock_tsla.oprices[27:-500]

print(model.forward(X, training = False))

plt.plot(range(len(y)), stock_tsla.oprices[27:-500])
plt.scatter(range(len(og_data)), og_data, c="red")
plt.show()

model.train(X, y, epochs = 10, printevery = 10, validation_data = (X_test, y_test), batch_size=64)

og_data = model.output_layer_activation.predictions(model.forward(later_X[:-500], training = False)) * stock_tsla.oprices[27:-500]

plt.plot(range(len(y)), stock_tsla.oprices[27:-500])
plt.scatter(range(len(og_data)), og_data, c="red")
plt.show()

test_data = model.output_layer_activation.predictions(model.forward(X_test, training = False)) * stock_tsla.oprices[-500:]

plt.plot(range(len(y_test)), stock_tsla.oprices[-500:])
plt.scatter(range(len(test_data)), test_data, c="red")
plt.show()

print(model.forward(X, training = False))

"""model = Model()

model.add(Layer_Dense(4, 256))
model.add(Activation_ReLU())
model.add(Layer_Dense(256, 256))
model.add(Activation_ReLU())
model.add(Layer_Dense(256, 1))
model.add(Activation_Linear())

model.set(loss = Loss_MeanSquaredError(), optimizer = Optimizer_Adam(learning_rate = .01, decay = 1e-7), accuracy = Accuracy_Regression())

model.finalize()

model.train(X, y, epochs = 5, printevery = 10, validation_data = (X_test, y_test), batch_size = 128)

plt.plot(range(len(y)), y)
plt.plot(range(len(y)), model.output_layer_activation.predictions(model.forward(X, training = False)))
plt.show()

plt.plot(range(len(y_test)), y_test * divisor)
#plt.plot(range(len(y_test)), X_test[:, 0] * divisor)
plt.plot(range(len(y_test)), model.output_layer_activation.predictions(model.forward(X_test, training = False)) * divisor)
plt.show()

model.evaluate(X_test, X_test[:, 0].reshape(-1, 1))

plt.plot(range(len(stock_tsla.cprices)), stock_tsla.cprices)
plt.plot(range(len(stock_tsla.cprices)), stock_tsla.oprices)
plt.show()

print(stock_tsla.cprices)
print(stock_tsla.oprices)"""