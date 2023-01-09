import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import yahoo_fin.stock_info as si
import time
import datetime as dt
import requests

def create_data(stock, range = 3):
    #range is in months
    global earnings_dates
    global earnings_dates_actvsexp
    global prices
    global global_volumes
    global stock_data

    earnings = pd.DataFrame.from_dict(si.get_earnings_history(stock))
    if not earnings.empty:
        earnings_dates = [x[:10] for x in earnings["startdatetime"] if len(x) > 10]
        earnings_dates_actvsexp = [x for x in earnings["epsactual"] / earnings["epsestimate"]]
    else:
        earnings_dates = [0]

    ticker = yf.Ticker(stock)
    start = dt.date.today() - dt.timedelta(days = 30 * range)
    stock_data = ticker.history(interval = "5m", start = start)

    prices = np.array(stock_data["Close"])
    global_volumes = np.array(stock_data["Volume"])

def create_normal_line(prices, batches = 5):

    if hasattr(prices, "size"):
        prices = prices.tolist()

    batch_size = int(len(prices) / batches)

    normal_line = np.array([])

    for batch in range(batches - 1):
        price_range = prices[(batch + 1) * batch_size] - prices[batch * batch_size]
        time_range = batch_size

        normal_line = np.concatenate([normal_line, np.array(range(time_range)) * price_range / time_range + prices[batch * batch_size]])

    price_range = prices[-1] - prices[batch * batch_size]
    time_range = len(prices) - batch_size * (batch + 1)

    normal_line = np.concatenate([normal_line, np.array(range(time_range)) * price_range / time_range + prices[batch * batch_size]])

    plt.plot(range(len(prices)), prices)
    plt.plot(range(len(prices)), normal_line)
    plt.show()

    return normal_line

def model(prices, volumes = [], start_safe_length = 20, sens = .05, volatilities = [], rsis = [], graph = False, first = False):

    last_unsafe_index = 0
    safe_length = start_safe_length
    change = 0
    
    normal_line = np.array([])
    emas = calc_ema(prices, 20)

    if not len(volatilities):
        volatilities = np.zeros(len(prices))

    if not len(rsis):
        rsis = np.zeros(len(prices)) + 50

    if not len(volumes):
        volumes = np.ones(len(prices))

    avg_variance = np.mean(volatilities)
    rsi_emas = abs(calc_ema(rsis, 14) - 50)
    rsis = abs(rsis - 50)

    b_signals = [0]
    s_signals = [0]

    for index in range(1, len(prices)):
        vol_sens = sens + sens * (avg_variance - volatilities[index])

        if abs(prices[index] - prices[last_unsafe_index] - change) > (2 - safe_length / start_safe_length / 2) * (1 + vol_sens) * abs(change) / (index - last_unsafe_index):
            b_signals.append((((prices[index] - prices[index - 1]) - change / (index - last_unsafe_index)) >= 0) * prices[index])
            s_signals.append((((prices[index] - prices[index - 1]) - change / (index - last_unsafe_index)) < 0) * prices[index])

            normal_line = np.concatenate([normal_line, np.array(range(index - last_unsafe_index)) * change / (index - last_unsafe_index) + prices[last_unsafe_index]])

            change = ((emas[index] - prices[last_unsafe_index]) * (.8 - volatilities[index]) + (prices[index] - prices[last_unsafe_index]) * (.2 + volatilities[index])) / (index - last_unsafe_index)
            last_unsafe_index = index
            safe_length = start_safe_length

            if rsis[index] > 20:
                temp_swap = b_signals[index]
                b_signals[index] = s_signals[index]
                s_signals[index] = temp_swap


        elif safe_length:
            change = (emas[index] - emas[last_unsafe_index]) * .5 + (prices[index] - prices[last_unsafe_index]) * .5
            safe_length -= 1

            b_signals.append(0)
            s_signals.append(0)


        elif not safe_length:
            if abs(prices[index] - prices[last_unsafe_index] - change * (index - last_unsafe_index) / start_safe_length) > (1 + vol_sens) * abs(change) / start_safe_length:
                b_signals.append((((prices[index] - prices[index - 1]) - change / (index - last_unsafe_index)) >= 0) * prices[index])
                s_signals.append((((prices[index] - prices[index - 1]) - change / (index - last_unsafe_index)) < 0) * prices[index])

                normal_line = np.concatenate([normal_line, np.array(range(index - last_unsafe_index)) * change / (index - last_unsafe_index) + prices[last_unsafe_index]])

                last_unsafe_index = index
                safe_length = start_safe_length
                change = (prices[index] - prices[index - 1]) * .8 + change * .2

                if rsis[index] > 30 or rsi_emas[index] > 20:
                    temp_swap = b_signals[index]
                    b_signals[index] = s_signals[index]
                    s_signals[index] = temp_swap

            else:
                b_signals.append(0)
                s_signals.append(0)

        sens *= .9999

    if index - last_unsafe_index:
        normal_line = np.concatenate([normal_line, np.array(range(index - last_unsafe_index + 1)) * change / (index - last_unsafe_index) + prices[last_unsafe_index]])
    else:
        normal_line = np.append(normal_line, prices[-1])

    if graph:
        plt.plot(prices)
        plt.plot(normal_line)
        plt.scatter(range(len(prices)), b_signals, c = "green")
        plt.scatter(range(len(prices)), s_signals, c = "red")
        plt.show()

    if first:
        last_val_changed = False
        lookback = 0

        while not last_val_changed:
            lookback -= 1
            if b_signals[lookback]:
                b_signals[-1] = 1
                last_val_changed = True
            if s_signals[lookback]:
                s_signals[-1] = 1
                last_val_changed = True

    return b_signals, s_signals

def simulate(buys, sells, prices, starting_money = 10000, graph = False, testing = True):

    money = starting_money
    money_over_time = [money]

    bought = False
    sold = False

    for index in range(1, len(prices)):
        if bought:
            money *= prices[index] / prices[index - 1]
        if sold:
            money *= 2 - prices[index] / prices[index - 1]

        if buys[index]:
            bought = True
            sold = False
        if sells[index]:
            sold = True
            bought = False

        money_over_time.append(money)

    if graph:
        plt.plot(range(len(money_over_time)), money_over_time)
        plt.show()

    if not testing:
        return money

    highest = np.max(money_over_time)

    return highest * .1 * (money_over_time.index(highest) / len(money_over_time)) ** 5 + money * .9

def volatility(prices, sens = 20):

    ema_data = calc_ema(prices, sens)

    volatilities = []

    for index in range(sens, len(prices)):
        volatilities.append(np.mean(abs(prices[index - sens:index] - ema_data[index - sens:index])))

    volatilities = np.concatenate([np.zeros(sens), volatilities]) / prices

    return volatilities

def calc_rsis(prices, period = 14, momentum = 0, extend = 50):
    prices = np.array(prices)

    price_changes = prices[1:] - prices[:-1]

    initial_avg_gain = np.sum(price_changes[:period] * (price_changes[:period] > 0)) / period
    initial_avg_loss = -np.sum(price_changes[:period] * (price_changes[:period] < 0)) / period

    avg_gains = [initial_avg_gain]
    avg_losses = [initial_avg_loss]

    initial_rsi = 100 - 100 / (1 + initial_avg_gain / initial_avg_loss)

    rsis = [initial_rsi]

    for index in range(len(prices) - period):
        avg_gains.append((avg_gains[index] * (period - 1) + price_changes[index + period - 1] * (price_changes[index + period - 1] > 0)) / period)
        avg_losses.append((avg_losses[index] * (period - 1) - price_changes[index + period - 1] * (price_changes[index + period - 1] < 0)) / period)
        rsis.append(100 - 100 / (1 + avg_gains[index + 1] / avg_losses[index + 1]))

    rsis = np.array([50 for x in range(period - 1)] + rsis)

    momentums = [0 for x in range(len(rsis))]
    variance = [50 - abs(50 - x) for x in rsis]

    for index in range(momentum, len(rsis)):
        greatest_change = 0
        for iteration in range(momentum):
            if abs(rsis[index] - rsis[index - iteration]) > abs(greatest_change):
                greatest_change = rsis[index] - rsis[index - iteration]

        momentums[index] = greatest_change * 1.5 * (variance[index] / 50) ** 2

    shifts = np.zeros(len(prices))

    for index in range(len(shifts)):
        if str(stock_data.iloc[index].name)[:10] in earnings_dates:
            earnings_index = earnings_dates.index(str(stock_data.iloc[index].name)[:10])
            for iteration in range(extend - (index + extend - len(shifts)) * int(index + extend > len(shifts))):
                shifts[index + iteration] = -(1 - iteration / 2 / extend) * abs(50 - rsis[index + iteration])  * (1 - 2 * int(earnings_dates_actvsexp[earnings_index] < 1))

    rsis = rsis + shifts + np.array(momentums) * .2

    return rsis

def calc_rsi_variance(rsis):

    extremes = 0
    for rsi in calc_ema(rsis, 10):
        if abs(rsi - 50) > 25:
            extremes += 1

    print(extremes / len(rsis))

def calc_ema(prices, period = 14):

    initial_ema = np.mean(prices[:period])

    emas = []
    emas.append(initial_ema)

    smoothing_factor = 2 / (1 + period)

    for index in range(1, len(prices) - period):
        ema = prices[index + period] * smoothing_factor + emas[index - 1] * (1 - smoothing_factor)
        emas.append(ema)

    emas = list(prices[:period]) + emas

    return np.array(emas)

def calc_vol_avgs(volumes, period = 20):

    vol_avgs = volumes

    for index in range(period - 1, len(vol_avgs)):
        vol_avgs[index] = np.mean(volumes[index - period + 1: index + 1])
    
    return vol_avgs

def find_best_vars(prices, volume = True, volatile = True, rsi = True, graph = False):
    volatilities = []
    rsis = []
    volumes = []
    
    settings_code = 0

    if volume:
        volumes = global_volumes
        settings_code += 100

    if volatile:
        volatilities = volatility(prices)
        settings_code += 10
        
    if rsi:
        rsis = calc_rsis(prices, extend = int(50 * 6.5))
        settings_code += 1

    possible_values = []

    for test in range(1, 1001):

        buys, sells = model(prices, volumes, sens = test / 100, volatilities = volatilities, rsis = rsis)

        money = simulate(buys, sells, prices)

        possible_values.append(money)

    best_sens = np.max(np.array(possible_values))
    best_sens_value = possible_values.index(best_sens) + 1

    if graph:
        print("Best: ", best_sens, " at a value of ", best_sens_value)

    buys, sells = model(prices, volumes, sens = best_sens_value / 100, volatilities = volatilities, rsis = rsis)
    money = simulate(buys, sells, prices, testing = False)
    
    if graph:
        plt.plot(range(1, 1001), possible_values)
        plt.show()

    return best_sens_value, money, settings_code

def big_boy_best_vars(prices):
    tests = [
        find_best_vars(prices, False, False, False),
        find_best_vars(prices, True, False, False),
        find_best_vars(prices, True, True, False),
        find_best_vars(prices, True, True, True),
        find_best_vars(prices, False, True, True),
        find_best_vars(prices, False, False, True),
        find_best_vars(prices, True, False, True),
        find_best_vars(prices, False, True, False)]

    tests = np.array(tests)

    vars = tests[:,0]
    money_vals = tests[:,1]
    settings = tests[:,2]
    
    highest = list(money_vals).index(np.max(money_vals))

    return vars[highest], settings[highest]

def trade(prices, stock, sens, settings_code):
    volumes = []
    volatilities = []
    rsis = []

    if stock.last_signal == 1:
        stock.invested_amount *= prices[-1] / stock.last_bs_price
        stock.last_bs_price = prices[-1]
        print("bought", stock.invested_amount)
    if stock.last_signal == 2:
        stock.invested_amount *= 2 - prices[-1] / stock.last_bs_price
        stock.last_bs_price = prices[-1]
        print("sold", stock.invested_amount)

    if settings_code / 100:
        volumes = global_volumes

    if settings_code % 100 / 10:
        volatilities = volatility(prices)
        
    if settings_code % 100 % 10:
        rsis = calc_rsis(prices, extend = int(50 * 6.5))

    buy, sell = model(prices, sens = sens, volumes = volumes, volatilities = volatilities, rsis = rsis, first = not bool(stock.invested_amount))

    if buy[-1]:
        print("buy")
        signal(stock, 1)
    elif sell[-1]:
        print("sell")
        signal(stock, 2)
    else:
        print("hold")

def auto_trade(stocks):
    stocks = [Stock(stock, 8000) for stock in stocks]

    while True:
        for stock in stocks:
            create_data(stock.symbol, 1/5)
            print(stock.symbol, dt.datetime.now(), ":", stock.buying_power + stock.invested_amount)
            sens, settings_code = big_boy_best_vars(prices)
            trade(prices, stock, sens = sens / 100, settings_code = settings_code)
        print("--------------------------------------------")
        time.sleep(60)

def signal(stock, new_signal):

    data = {
        "systemid" : 1, #REDACTED
        "apikey" : "REDACTED",
        "signal" : {
            "market" : 1,
            "symbol" : stock.symbol.upper(),
            "typeofsymbol" : "stock",
            "action" : "",
            "duration" : "GTC",
            "quant" : stock.invested_amount / prices[-1]
            }
        }

    if stock.last_signal != new_signal:
        if stock.last_signal:
            if stock.last_signal - 1:
                data["signal"]["action"] = "BTC"
            else:
                data["signal"]["action"] = "STC"
            request = requests.post("https://api.collective2.com/world/apiv3/submitSignal", params = {}, json = data)
            txt = open("signals.txt", "a")
            txt.write(stock.symbol + " | " + data["signal"]["action"] + " | " + str(dt.datetime.now()) + "\n")
            print(stock.symbol, "Reversing: ", data["signal"]["action"], request, stock.invested_amount)
            txt.close()

            stock.buying_power += stock.invested_amount
            stock.invested_amount -= stock.invested_amount
            time.sleep(5)

        data["signal"]["quant"] = stock.buying_power / prices[-1]

        if new_signal - 1:
            data["signal"]["action"] = "STO"
        else:
            data["signal"]["action"] = "BTO"

        request = requests.post("https://api.collective2.com/world/apiv3/submitSignal", params = {}, json = data)
        txt = open("signals.txt", "a")
        txt.write(stock.symbol + " | " + data["signal"]["action"] + " | " + str(dt.datetime.now()) + "\n")
        print(stock.symbol, "Signal: ", data["signal"]["action"], request, stock.buying_power)
        txt.close()
    else:
        print("Same Signal")

    stock.invested_amount += stock.buying_power
    stock.buying_power -= stock.buying_power

    stock.last_bs_price = prices[-1]
    stock.last_signal = new_signal

class Stock:

    def __init__(self, stock, buying_power = 10000):
        self.symbol = stock
        self.buying_power = buying_power
        self.last_bs_price = 0
        #bs meaning buy/sell
        self.last_signal = 0
        self.invested_amount = 0



stocks = ["amzn", "aapl", "tsla", "tqqq", "sqqq"]
auto_trade(stocks)

"""find_best_vars(prices, [], 0, False, False)
find_best_vars(prices, volumes, 0, False, False)
find_best_vars(prices, volumes, 0, True, False)
find_best_vars(prices, volumes, 0, True, True)
find_best_vars(prices, [], 0, True, True)
find_best_vars(prices, [], 0, False, True)
find_best_vars(prices, volumes, False, True)
find_best_vars(prices, [], 0, True, False)"""

"""buys, sells = model(prices, volumes, sens = vars / 100, graph = False)

money = simulate(buys, sells, prices, graph = False, testing = False)

print("No Volatilities:", money)

buys, sells = model(prices, volumes, sens = vars / 100, volatilities = volatilities, graph = False)

money = simulate(buys, sells, prices, graph = False, testing = False)

print("Volatiliites: ", money)

buys, sells = model(prices, volumes, sens = vars / 100, rsis = rsis, graph = False)

money = simulate(buys, sells, prices, graph = False, testing = False)

print("Rsis: ", money)

buys, sells = model(prices, volumes, sens = vars / 100, volatilities = volatilities, rsis = rsis, graph = False)

money = simulate(buys, sells, prices, graph = False, testing = False)

print("Everything: ", money)"""