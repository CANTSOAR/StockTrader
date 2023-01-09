import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import yahoo_fin.stock_info as si
import time
import datetime as dt
import requests
from robin_stocks import robinhood as rh

def create_data(stock, range = 3):
    #range is in months
    global prices
    global global_volumes
    global stock_data

    ticker = yf.Ticker(stock)
    start = dt.date.today() - dt.timedelta(days = 30 * range)
    stock_data = ticker.history(interval = "1h", start = start)

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

def model(prices, volumes = [], start_safe_length = 20, sens = .05, volatilities = [], rsis = [], graph = False, first = False, stock = None):

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
        lookback = 0

        while not stock.traded:
            lookback -= 1
            if b_signals[lookback]:
                b_signals[-1] = 1
                stock.traded = True
            if s_signals[lookback]:
                s_signals[-1] = 1
                stock.traded = True

        

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
            #money *= 2 - prices[index] / prices[index - 1]
            money *= 1

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

    rsis = rsis + np.array(momentums) * .2

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

    print(money_vals)

    return vars[highest], settings[highest]

def trade(prices, stock, sens, settings_code):
    volumes = []
    volatilities = []
    rsis = []

    if stock.last_signal == 1:
        print("bought", stock.invested_amount * float(rh.crypto.get_crypto_quote(stock.symbol[:-4].upper(), info="mark_price")))
    if stock.last_signal == 0:
        print("held", stock.buying_power)

    if settings_code / 100:
        volumes = global_volumes

    if settings_code % 100 / 10:
        volatilities = volatility(prices)
        
    if settings_code % 100 % 10:
        rsis = calc_rsis(prices, extend = int(50 * 6.5))

    buy, sell = model(prices, sens = sens, volumes = volumes, volatilities = volatilities, rsis = rsis, first = not bool(stock.traded), stock = stock, graph = True)

    if buy[-1]:
        print("buy")
        signal(stock, 1)
    elif sell[-1]:
        print("sell")
        signal(stock, 0)
    else:
        print("hold")

def auto_trade(stocks):
    rh.authentication.login(username = "REDACTED", password = "REDACTED")

    print("Current Buying Power:", rh.profiles.load_account_profile("buying_power"))
    starting = float(rh.profiles.load_account_profile("buying_power")) / len(stocks)
    global stock_objects
    stock_objects = [Stock(stock, starting, get_held_value(stock[:-4].upper())) for stock in stocks]

    while True:
        for stock in stock_objects:
            create_data(stock.symbol, range = 12 / 10)
            print(stock.symbol, dt.datetime.now(), ":", stock.buying_power + stock.invested_amount * float(rh.crypto.get_crypto_quote(stock.symbol[:-4].upper(), info="mark_price")))
            sens, settings_code = big_boy_best_vars(prices)
            trade(prices, stock, sens = sens / 100, settings_code = settings_code)
            print("-----")
        print("====================================\n-----")
        time.sleep(3600 - 150 * len(stock_objects))

def rebalance_buying_power():
    untraded_stocks = 0
    for stock in stock_objects:
        if not stock.invested_amount:
            untraded_stocks += 1
    print("Divided Buying Power:", float(rh.profiles.load_account_profile("buying_power")) / untraded_stocks)
    for stock in stock_objects:
        if not stock.invested_amount:
            stock.buying_power = float(rh.profiles.load_account_profile("buying_power")) / untraded_stocks

def get_held_value(stock):
    held_stocks = [x["code"] for x in rh.crypto.get_crypto_positions("currency")]
    
    if stock in held_stocks:
        index_of_given_stock = held_stocks.index(stock)
        return float(rh.crypto.get_crypto_positions("quantity")[index_of_given_stock])

    return 0

def signal(stock, new_signal):

    if stock.last_signal != new_signal:
        symbol = stock.symbol[:-4].upper()

        if new_signal:
            print(round(stock.buying_power * .95, 2))
            #print(rh.orders.order_buy_crypto_limit_by_price(symbol, round(stock.buying_power * .95, 2), round(float(rh.crypto.get_crypto_quote(symbol, info="ask_price")), 2)))
            
            tries = 0
            while not float(get_held_value(symbol)):
                time.sleep(30)
                print("Rechecking...")
                tries += 1
                if tries == 10:
                    print(rh.orders.cancel_all_crypto_orders())
                    #print(rh.orders.order_buy_crypto_limit_by_price(symbol, round(stock.buying_power * .95, 2), round(float(rh.crypto.get_crypto_quote(symbol, info="ask_price")), 2)))
                    tries = 0
                pass

            quantity = float(get_held_value(symbol))
            print("Bought {} {} for {} USD at price {}".format(quantity, symbol, stock.buying_power * .95, round(float(rh.crypto.get_crypto_quote(symbol, info="ask_price")), 2)))

            with open("cryptotradehistory.txt", "a") as txt:
                txt.write("BOUGHT | " + str(quantity) + " of " + symbol + " for " + str(stock.buying_power * .95) + "USD at a price of " + str(round(float(rh.crypto.get_crypto_quote(symbol, info="ask_price")), 2)) + " | " + str(dt.datetime.now()) + "\n")


            stock.invested_amount += quantity
            stock.buying_power = 0
        else:
            #print(rh.orders.order_sell_crypto_limit(symbol, stock.invested_amount, round(float(rh.crypto.get_crypto_quote(symbol, info="bid_price")), 2)))

            tries = 0
            while not float(get_held_value(symbol)):
                time.sleep(30)
                print("Rechecking...")
                tries += 1
                if tries == 10:
                    print(rh.orders.cancel_all_crypto_orders())
                    #print(rh.orders.order_sell_crypto_limit(symbol, stock.invested_amount, round(float(rh.crypto.get_crypto_quote(symbol, info="bid_price")), 2)))
                    tries = 0
                pass

            usd = stock.invested_amount * float(rh.crypto.get_crypto_quote(symbol, info="bid_price"))
            print("Sold {} {} for {} USD at price {}".format(stock.invested_amount, symbol, usd, round(float(rh.crypto.get_crypto_quote(symbol, info="bid_price")), 2)))

            with open("cryptotradehistory.txt", "a") as txt:
                txt.write("SOLD | " + str(stock.invested_amount) + " of " + symbol + " for " + str(usd)  + "USD at price " + str(round(float(rh.crypto.get_crypto_quote(symbol, info="bid_price")), 2)) +  " | " + str(dt.datetime.now()) + "\n")

            stock.buying_power += float(rh.profiles.load_account_profile("buying_power"))
            stock.invested_amount = 0

            if  usd * 1.1 < float(rh.profiles.load_account_profile("buying_power")):
                rebalance_buying_power()

    else:
        print("No change")

    stock.last_bs_price = prices[-1]
    stock.last_signal = new_signal

class Stock:

    def __init__(self, stock, buying_power = 10000, invested_amount = 0):
        self.symbol = stock
        self.buying_power = buying_power
        self.last_bs_price = 0
        #bs meaning buy/sell
        self.last_signal = 0
        self.invested_amount = invested_amount
        self.traded = False

        if self.invested_amount:
            self.last_bs_price = float(rh.crypto.get_crypto_quote(self.symbol[:-4].upper(), info="mark_price")) / self.invested_amount
            self.last_signal = 1

        print("Stock", self.symbol, "created with", self.buying_power, "buying_power and", self.invested_amount, "held")



stock = [input("Pick stock for graphs: ")]

auto_trade(stock)