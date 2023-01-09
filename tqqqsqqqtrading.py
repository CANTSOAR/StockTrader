import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import time
import datetime as dt
from robin_stocks import robinhood as rh
import mathtests as vfield

names = ["sqqq", "tqqq", "neither"]

def create_data(stock, range = 12, interval = "1d"):
    #range is in months
    global prices
    global global_volumes
    global stock_data

    ticker = yf.Ticker(stock)
    start = dt.date.today() - dt.timedelta(days = 30 * range)
    stock_data = ticker.history(interval = interval, start = start)

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

    macd, macd_signal = calc_macd(prices)

    if not len(volatilities):
        volatilities = np.zeros(len(prices))

    if not len(rsis):
        rsis = np.zeros(len(prices)) + 50

    if not len(volumes):
        volumes = np.ones(len(prices))

    avg_variance = np.mean(volatilities)

    b_signals = [0]
    s_signals = [0]
    reasons = [0]

    index = 1

    for index in range(1, len(prices)):
        vol_sens = sens + sens * (avg_variance - volatilities[index])
            
        """if rsis[index] >= 80:
            b_signals.append(0)
            s_signals.append(prices[index])

        elif rsis[index] <= 20:
            b_signals.append(prices[index])
            s_signals.append(0)"""

        if rsis[index] >= 70:
            if (macd[index] - macd_signal[index]) / (macd[index - 1] - macd_signal[index - 1] + .1 ** 50) < 1:
                b_signals.append(0)
                s_signals.append(prices[index])
                reasons.append(1)

                normal_line = np.concatenate([normal_line, np.array(range(index - last_unsafe_index)) * change / (index - last_unsafe_index) + prices[last_unsafe_index]])

                last_unsafe_index = index
                safe_length = start_safe_length
                change = (prices[index] - prices[index - 1]) * .8 + change * .2
            else:
                b_signals.append(0)
                s_signals.append(0)
                reasons.append(0)

        elif rsis[index] <= 30:
            if (macd[index] - macd_signal[index]) / (macd[index - 1] - macd_signal[index - 1] + .1 ** 50) < 1:
                b_signals.append(prices[index])
                s_signals.append(0)
                reasons.append(1)

                normal_line = np.concatenate([normal_line, np.array(range(index - last_unsafe_index)) * change / (index - last_unsafe_index) + prices[last_unsafe_index]])

                last_unsafe_index = index
                safe_length = start_safe_length
                change = (prices[index] - prices[index - 1]) * .8 + change * .2
            else:
                b_signals.append(0)
                s_signals.append(0)
                reasons.append(0)

        elif abs(prices[index] - prices[last_unsafe_index] - change) > (2 - safe_length / start_safe_length / 2) * (1 + vol_sens) * abs(change) / (index - last_unsafe_index):
            b_signals.append((((prices[index] - prices[index - 1]) - change / (index - last_unsafe_index)) >= 0) * prices[index])
            s_signals.append((((prices[index] - prices[index - 1]) - change / (index - last_unsafe_index)) < 0) * prices[index])
            reasons.append(2)

            normal_line = np.concatenate([normal_line, np.array(range(index - last_unsafe_index)) * change / (index - last_unsafe_index) + prices[last_unsafe_index]])

            change = ((emas[index] - prices[last_unsafe_index]) * (.8 - volatilities[index]) + (prices[index] - prices[last_unsafe_index]) * (.2 + volatilities[index])) / (index - last_unsafe_index)
            last_unsafe_index = index
            safe_length = start_safe_length

        elif safe_length:
            change = (emas[index] - emas[last_unsafe_index]) * .5 + (prices[index] - prices[last_unsafe_index]) * .5
            safe_length -= 1

            b_signals.append(0)
            s_signals.append(0)
            reasons.append(0)


        elif not safe_length:
            if abs(prices[index] - prices[last_unsafe_index] - change * (index - last_unsafe_index) / start_safe_length) > (1 + vol_sens) * abs(change) / start_safe_length:
                b_signals.append((((prices[index] - prices[index - 1]) - change / (index - last_unsafe_index)) >= 0) * prices[index])
                s_signals.append((((prices[index] - prices[index - 1]) - change / (index - last_unsafe_index)) < 0) * prices[index])
                reasons.append(3)

                normal_line = np.concatenate([normal_line, np.array(range(index - last_unsafe_index)) * change / (index - last_unsafe_index) + prices[last_unsafe_index]])

                last_unsafe_index = index
                safe_length = start_safe_length
                change = (prices[index] - prices[index - 1]) * .8 + change * .2

            else:
                b_signals.append(0)
                s_signals.append(0)
                reasons.append(0)

    if index - last_unsafe_index:
        normal_line = np.concatenate([normal_line, np.array(range(index - last_unsafe_index + 1)) * change / (index - last_unsafe_index) + prices[last_unsafe_index]])
    else:
        normal_line = np.append(normal_line, prices[-1])

    lookback = 0
    traded = False

    while not traded:
        lookback -= 1
        if b_signals[lookback]:
            b_signals[-1] = 1
            traded = True
        if s_signals[lookback]:
            s_signals[-1] = 1
            traded = True

    if graph:
        plt.plot(prices)
        plt.plot(normal_line)
        plt.scatter(range(len(prices)), b_signals, c = "green")
        plt.scatter(range(len(prices)), s_signals, c = "red")
        plt.show()

    return b_signals, s_signals, reasons

def macd_rsi_strat(prices, rsis = [], graph = False):

    macd, macd_signal = calc_macd(prices)
    macd_diffs = macd - macd_signal
    macd_changes = [0 for x in range(len(prices) - len(macd) + 1)] + list((macd[1:] - macd_signal[1:]) - (macd[:-1] - macd_signal[:-1]))
    macd_change_ema = calc_ema(macd_changes, 9)

    b_signals = [0]
    s_signals = [0]

    open_buy = False
    open_sell = False

    open_rsi = 0
    open_rsi_diff = 0
    rsi_emas = calc_ema(rsis)
    high_low, med = rsi_quartile_highlowmeds(rsis)

    for index in range(1, len(prices)):
        #print(str(index) + ":", med[index] + high_low[index], rsis[index], med[index] - high_low[index], macd_diffs[index], macd_changes[index], open_rsi, open_rsi_diff)

        """if open_buy:
            if rsis[index] - open_rsi < .9 * open_rsi_diff and macd_changes[index] < 0 and rsis[index] > med[index] - high_low[index]:
                b_signals.append(0)
                s_signals.append(prices[index])
                open_rsi_diff = 0
                open_buy = False
                print("sold to close position at price: ", prices[index], "at index", index)
            else:
                open_rsi_diff = rsis[index] - open_rsi
                b_signals.append(0)
                s_signals.append(0)

        elif open_sell:
            if open_rsi - rsis[index] < .9 * open_rsi_diff and macd_changes[index] > 0 and rsis[index] < med[index] + high_low[index]:
                b_signals.append(prices[index])
                s_signals.append(0)
                open_rsi_diff = 0
                open_sell = False
                print("bought to close position at price: ", prices[index], "at index", index)
            else:
                open_rsi_diff = open_rsi - rsis[index]
                b_signals.append(0)
                s_signals.append(0)

        if rsis[index] > med[index] + high_low[index] * 1 and macd_diffs[index] > 0 and macd_changes[index] < 0:
            b_signals.append(0)
            s_signals.append(prices[index])
            open_sell = True
            open_rsi = rsis[index]
            print("sold to open position at price: ", prices[index], "at index", index)
        elif rsis[index] < med[index] - high_low[index] * 1 and macd_diffs[index] < 0 and macd_changes[index] > 0:
            b_signals.append(prices[index])
            s_signals.append(0)
            open_buy = True
            open_rsi = rsis[index]
            print("bought to open position at price: ", prices[index], "at index", index)
        elif rsis[index] > med[index] + high_low[index] and macd_diffs[index] < 0 and macd_changes[index] < 0:
            b_signals.append(0)
            s_signals.append(prices[index])
            open_sell = True
            open_rsi = rsis[index]
            print("sold to open position at price: ", prices[index], "at index", index)
        elif rsis[index] < med[index] - high_low[index] and macd_diffs[index] > 0 and macd_changes[index] > 0:
            b_signals.append(prices[index])
            s_signals.append(0)
            open_buy = True
            open_rsi = rsis[index]
            print("bought to open position at price: ", prices[index], "at index", index)
        else:
            b_signals.append(0)
            s_signals.append(0)"""

        if rsi_emas[index] > med[index] + high_low[index]:
            b_signals.append(0)
            s_signals.append(prices[index])
        elif rsi_emas[index] < med[index] - high_low[index]:
            b_signals.append(prices[index])
            s_signals.append(0)
        elif rsis[index] > med[index] + high_low[index] and macd_changes[index] < 0:
            if rsi_emas[index] > med[index] + high_low[index] * .5:
                b_signals.append(0)
                s_signals.append(0)
            else:
                b_signals.append(0)
                s_signals.append(prices[index])
        elif rsis[index] < med[index] - high_low[index] and macd_changes[index] > 0:
            if rsi_emas[index] > med[index] - high_low[index] * .5:
                b_signals.append(0)
                s_signals.append(0)
            else:
                b_signals.append(prices[index])
                s_signals.append(0)
        else:
            b_signals.append(0)
            s_signals.append(0)

    print("original:", simulate(b_signals, s_signals, prices))
    winning_trades(b_signals, s_signals, prices)

    if graph:
        """fig = plt.figure()

        ax1 = plt.subplot2grid((8,1), (0,0), rowspan = 5)
        ax2 = plt.subplot2grid((8,1), (5,0), sharex = ax1)
        ax3 = plt.subplot2grid((8,1), (6,0), sharex = ax1)
        ax4 = plt.subplot2grid((8,1), (7,0), sharex = ax1)

        ax1.plot(prices)
        ax1.scatter(range(len(prices)), b_signals, c = "green")
        ax1.scatter(range(len(prices)), s_signals, c = "red")
        ax2.plot(macd)
        ax2.plot(macd_signal)
        ax3.plot(macd_diffs)
        ax3.plot(macd_changes)
        ax3.plot([0 for x in range(len(prices))], c = "black")
        ax4.plot(rsis)
        ax4.plot(np.array(med) + np.array(high_low), c = "black")
        ax4.plot(np.array(med) - np.array(high_low), c = "black")
        ax4.plot(rsi_emas)
        ax4.plot(med, c = "black")

        plt.show()"""

        #plt.plot(prices)
        one, two = calc_macd(prices)
        three = one - two
        four = differentiate(three)
        five = differentiate(four)

        """for index in range(2, len(prices)):
            if (six[index - 1] - six[index]) * (six[index - 1] - six[index - 2]) > 0 and six[index - 1] > six[index]:
                maxes.append(six[index] * 10)
                mins.append(None)
                if seven[index] < 0:
                    print("price will go down:", index)
                    sells.append(prices[index])
                else:
                    sells.append(None)
                buys.append(None)
            elif (six[index - 1] - six[index]) * (six[index - 1] - six[index - 2]) > 0 and six[index - 1] < six[index]:
                maxes.append(None)
                mins.append(six[index] * 10)
                if seven[index] > 0:
                    print("price will go up:", index)
                    buys.append(prices[index])
                else:
                    buys.append(None)
                sells.append(None)

            else:
                buys.append(None)
                sells.append(None)
                maxes.append(None)
                mins.append(None)"""


        #plt.show()

        #DETERMINE WHERE RSI IS PEAKING AND AT A MINIMUM

        look_back = 20
        means, ups, downs = statistic(rsis, look_back)
        plt.plot(prices)
        updowns = [None]

        five = calc_ema(five, 14)

        buys = [0]
        sells = [0]
        rsis = calc_ema(rsis, 7)
        five = differentiate(four)

        lastsigpoint = ""

        for index in range(1, len(rsis)):
            if (rsis[index] < ups[index] and rsis[index - 1] > ups[index - 1]) or (rsis[index] < downs[index] and rsis[index - 1] > downs[index - 1]): #coming back down through top or up through bottom
                lastsigpoint = "upper"
                if four[index] + five[index] < 0:
                    sells.append(prices[index])
                else:
                    sells.append(None)
                updowns.append(prices[index])
                buys.append(None)
            elif (rsis[index] > downs[index] and rsis[index - 1] < downs[index - 1]) or (rsis[index] > ups[index] and rsis[index - 1] < ups[index - 1]): #coming back up through bottom or down through top
                lastsigpoint = "lower"
                if four[index] + five[index] > 0:
                    buys.append(prices[index])
                else:
                    buys.append(None)
                updowns.append(prices[index])
                sells.append(None)
            elif (rsis[index] - means[index]) *  (rsis[index - 1] - means[index - 1]) < 0: #reversals about mid line
                if lastsigpoint == "mid":
                    updowns.append(prices[index])
                    if rsis[index] < means[index]:
                        sells.append(prices[index])
                        buys.append(None)
                    else:
                        buys.append(prices[index])
                        sells.append(None)
                else:
                    updowns.append(None)
                    buys.append(None)
                    sells.append(None)

                lastsigpoint = "mid"
                
            else:
                updowns.append(None)
                buys.append(None)
                sells.append(None)

        plt.plot(ups)
        plt.plot(downs)
        plt.plot(means)
        plt.plot(rsis)
        plt.plot(one * 5 + 20, color = "black")
        plt.plot(two * 5 + 20, color = "blue")
        plt.plot(np.array(four) * 5 + 20, color = "green")
        plt.plot(np.zeros(len(prices)) + 20)
        plt.plot(three * 5 + 20, color = "red")
        plt.plot((np.array(four) + five) * 5 + 20)
        plt.scatter(range(len(prices)), buys, color = "green", linewidths = 5)
        plt.scatter(range(len(prices)), sells, color = "red", linewidths = 5)
        plt.scatter(range(len(prices)), updowns, color = "black")

        plt.show()

        print("current: ", simulate(buys, sells, prices, graph = True))
        winning_trades(buys, sells, prices)

def winning_trades(buys, sells, prices):
    bought = False
    sold = False
    bought_price = 0
    sold_price = 0
    winning = 0
    trades = 0
    start = 0

    while not bought and not sold:
        if buys[start]:
            bought = True
            sold = False
            bought_price = prices[start]
            trades += 1
        elif sells[start]:
            sold = False
            bought = True
            sold_price = prices[start]
            trades += 1
        start += 1

    for index in range(start, len(prices)):
        if bought and sells[index]:
            if bought_price < prices[index]:
                winning += 1
            bought = False
            sold = True
            sold_price = prices[index]
            trades += 1
        elif sold and buys[index]:
            if sold_price > prices[index]:
                winning += 1
            bought = True
            sold = False
            bought_price = prices[index]
            trades += 1

    print(winning, "winning trades out of", trades, "total trades;", winning / trades, "win ratio")
    return winning, trades

def statistic(input, look_back = 14):

    means = calc_ema(input, look_back)
    stdevs = np.zeros(look_back)
    input = calc_ema(input, int(look_back / 2))

    for index in range(look_back, len(input)):
        stdevs = np.append(stdevs, np.std(input[index - look_back: index] - means[index - look_back: index]))

    smoothdevup = calc_ema(3 * stdevs + means, 3)
    smoothdevdown = calc_ema(-3 * stdevs + means, 3)

    return means, smoothdevup, smoothdevdown


def differentiate(input):
    input = np.array(input)
    return [0] + list(input[1:] - input[:-1])

def add_momentum(data, lookback = 5, amount = .3):
    momentums = [0 for x in range(lookback)]
    
    for index in range(lookback, len(data)):
        momentums.append(data[index] + amount * np.mean(data[index - lookback + 1: index + 1] - data[index - lookback: index]))

    return momentums

def rsi_quartile_highlowmeds(rsis, look_back = 20):
    high_low = [20 for x in range(look_back)]
    med = [50 for x in range(look_back)]
    
    for index in range(look_back, len(rsis)):
        subset = rsis[index - look_back: index]
        subset = np.sort(subset)
        high_low.append((30 + subset[int(3/4 * len(subset))] - subset[int(1/4 * len(subset))]) / 4)
        med.append((np.mean(subset) * 3 + 50) / 4)

    return high_low, med

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

    #return highest * .1 * (money_over_time.index(highest) / len(money_over_time)) ** 5 + money * .9
    return money

def volatility(prices, sens = 20):

    ema_data = calc_ema(prices, sens)

    volatilities = []

    for index in range(sens, len(prices)):
        volatilities.append(np.mean(abs(prices[index - sens:index] - ema_data[index - sens:index])))

    volatilities = np.concatenate([np.zeros(sens), volatilities]) / prices

    return volatilities

def calc_macd(prices, fast = 12, slow = 26, signal = 9):
    macd = calc_ema(prices, fast) - calc_ema(prices, slow)
    macd_signal = calc_ema(macd, signal)

    return macd, macd_signal

def calc_momentum(graph, sens = 3):
    pre_momentums = (graph[sens:] - graph[:-sens]) / sens

    momentums = [0 for x in range(sens)]
    momentums += list(pre_momentums)

    momentums += graph

    return momentums

def calc_rsis(prices, period = 14, momentum = 10, momentum_multiplier = 0.0):
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

    if momentum:
        for index in range(momentum, len(rsis)):
            maxdif = rsis[index] - np.max(rsis[index - momentum:index])
            mindif = rsis[index] - np.min(rsis[index - momentum:index])

            momentums[index] = maxdif + (mindif - maxdif) * (abs(mindif) > abs(maxdif))

    rsis = rsis + np.array(momentums) * momentum_multiplier

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

def find_best_vars(prices, volume = True, volatile = True, rsi = True, graph = False, rmm = 0.0):
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
        rsis = calc_rsis(prices, momentum_multiplier = rmm)
        settings_code += 1

    possible_values = []

    for test in range(1, 1001):

        buys, sells, filler = model(prices, volumes, sens = test / 100, volatilities = volatilities, rsis = rsis)
        money = simulate(buys, sells, prices)

        possible_values.append(money)

    best_sens = np.max(np.array(possible_values))
    best_sens_value = possible_values.index(best_sens) + 1

    if graph:
        print("Best: ", best_sens, " at a value of ", best_sens_value)

    buys, sells, filler = model(prices, volumes, sens = best_sens_value / 100, volatilities = volatilities, rsis = rsis)
    money = simulate(buys, sells, prices, testing = False)

    return best_sens_value, money, settings_code, possible_values

def big_boy_best_vars(prices, graph = False, rmm = 0.0):
    tests = [
        find_best_vars(prices, False, False, False, graph, rmm), #NOTHING
        find_best_vars(prices, True, False, False, graph, rmm),  #JUST VOLUMES
        find_best_vars(prices, True, True, False, graph, rmm),   #VOLUMES AND VOLATILITIES
        find_best_vars(prices, True, True, True, graph, rmm),    #ALL 3 OF THEM
        find_best_vars(prices, False, True, True, graph, rmm),   #VOLATILITIES AND RSI
        find_best_vars(prices, False, False, True, graph, rmm),  #JUST RSI
        find_best_vars(prices, True, False, True, graph, rmm),   #VOLUMES AND RSI
        find_best_vars(prices, False, True, False, graph, rmm)]  #JUST VOLATILITIES

    tests = np.array(tests, dtype = object)

    vars = tests[:,0]
    money_vals = tests[:,1]
    settings = tests[:,2]
    possible_values = np.array(tests[:,3], dtype = object)
    possible_values = possible_values[0] + possible_values[1] + possible_values[2] + possible_values[3] + possible_values[4] + possible_values[5] + possible_values[6] + possible_values[7]
    
    highest = list(money_vals).index(np.max(money_vals))

    if graph:
        print(money_vals[highest], vars[highest], settings[highest])
        plt.plot(possible_values)
        plt.show()

    return vars[highest], settings[highest], rmm

def find_best_rsi_vars(prices, graph = False):

    possible_values = []

    for rsi_momentum_multiplier in range(-5, 6):
        rsis = calc_rsis(prices, momentum_multiplier = rsi_momentum_multiplier / 10)

        for test in range(1, 1001):

            buys, sells, filler = model(prices, sens = test / 100, rsis = rsis)
            money = simulate(buys, sells, prices)

            possible_values.append(money)

    best_sens = np.max(np.array(possible_values))
    best_sens_value = possible_values.index(best_sens) % 1000 + 1
    best_rsis_multiplier =  (int(possible_values.index(best_sens) / 1000) - 10) / 5

    if graph:
        print(best_sens, best_sens_value, best_rsis_multiplier)
        plt.plot(possible_values)
        plt.show()

    return best_sens_value, 1, best_rsis_multiplier

def trade(prices, sens, settings_code, rmm = 0):
    volumes, volatilities, rsis = [[], [], []]

    stock_object.invested_amount = get_held_value()

    if stock_object.last_signal == 1:
        print("bought", stock_object.invested_amount * float(rh.stocks.get_latest_price("tqqq")[0]), "at price", float(rh.stocks.get_latest_price("tqqq")[0]))
    if stock_object.last_signal == 0:
        print("sold", stock_object.invested_amount * float(rh.stocks.get_latest_price("sqqq")[0]), "at price", float(rh.stocks.get_latest_price("sqqq")[0]))

    if int(settings_code / 100):
        volumes = global_volumes

    if int(settings_code % 100 / 10):
        volatilities = volatility(prices)
        
    if int(settings_code % 100 % 10):
        rsis = calc_rsis(prices, momentum_multiplier = rmm)

    buy, sell, filler = model(prices, sens = sens, volumes = volumes, volatilities = volatilities, rsis = rsis)

    plt.title(dt.datetime.now())
    plt.scatter(range(len(prices[-50:])), buy[-50:], c = "green")
    plt.scatter(range(len(prices[-50:])), sell[-50:], c = "red")
    plt.plot(prices[-50:])
    plt.ylim(np.min(prices[-50:]) - 5, np.max(prices[-50:]) + 5)
    plt.savefig("last50-prices+actions.png")
    plt.close()

    if buy[-1]:
        print("switch to tqqq")
        signal(1, buy, sell)
    elif sell[-1]:
        print("switch to sqqq")
        signal(0, buy, sell)
    elif buy[-2]:
        print("switch to tqqq last hour")
        signal(1, buy, sell)
    elif sell[-2]:
        print("switch to sqqq last hour")
        signal(0, buy, sell)
    else:
        print("hold")

def auto_trade(interval, range):
    rh.authentication.login(username = "REDACTED", password = "REDACTED", expiresIn = 31556926)

    starting = rh.profiles.load_account_profile("buying_power")
    print("Current Buying Power:", float(starting))
    global stock_object
    stock_object = Stock(float(starting), get_held_value())

    while True:
        print("Current time:", dt.datetime.now().strftime("%H:%M"))
        if rh.markets.get_market_hours("XNAS", str(dt.datetime.today())[0:10], info="is_open") and int(dt.datetime.now().strftime("%H%M")) <= 1600 and int(dt.datetime.now().strftime("%H%M")) >= 930:
            print("Trading now:", dt.datetime.now().strftime("%H%M"))
            create_data("tqqq", interval = interval, range = range)
            print(dt.datetime.now(), ":", stock_object.invested_amount)
            
            rmm = find_best_rsi_vars(prices)[2] # <-- USING ONLY RSI AND MACD BUT WITH MODIFIED RSI
            sens, settings_code, rmm = big_boy_best_vars(prices, rmm = rmm) # <-- CHECKING WHICH INDICATORS TO USE
            
            print(sens, settings_code, rmm)

            trade(prices, sens = sens / 100, settings_code = settings_code, rmm = rmm)
            print("-----\n====================================\n-----")

        else:
            print("Couldn't trade now; waiting for next trading period")

        if int(dt.datetime.now().strftime("%H%M")) > 1545 or int(dt.datetime.now().strftime("%H%M")) < 930:
            wait_time = (2010 - int(dt.datetime.now().strftime("%H")) * 60 - int(dt.datetime.now().strftime("%M"))) % 1440
            
            print("Waiting", int(wait_time / 60), "hours and", wait_time % 60, "minutes for 9:30 AM, now it's", dt.datetime.now().strftime("%H:%M"))
            time.sleep(wait_time * 60)
            rh.authentication.login(username = "dipalsk@gmail.com", password = "NyQuil@44$$", expiresIn = 31556926)
        else:
            wait_time = (int(dt.datetime.now().strftime("%H")) * 100 + 30 - int(dt.datetime.now().strftime("%H%M"))) % 60
            if not wait_time:
                wait_time = 60
            print("Waiting", wait_time, "minutes for next half hour (X:30)")
            time.sleep(60 * wait_time)

def show_best_graph(stock = "tqqq", interval = "1d", range = 12, graph = True, backtestamt = 100):
    print("NOW SHOWING GRAPHS FOR", stock, "ON", interval, "INTERVALS FOR THE PAST", range, "MONTHS")

    print(dt.datetime.now())

    create_data(stock, range = range, interval = interval)
    print(dt.datetime.now(), "testing for graphs")

    training_prices = prices[:int(backtestamt / 100 * len(prices))]

    print("rsi based")
    sens, settings_code, rmm = find_best_rsi_vars(training_prices, graph) # <-- USING ONLY RSI AND MACD BUT WITH MODIFIED RSI
    print(sens, settings_code, rmm)

    volumes, volatilities, rsis = [[], [], []]

    if int(settings_code / 100):
        print(settings_code / 100)
        volumes = global_volumes

    if int(settings_code % 100 / 10):
        print(settings_code % 100 / 10)
        volatilities = volatility(prices)
        
    if int(settings_code % 100 % 10):
        rsis1 = calc_rsis(prices, momentum_multiplier = rmm)

    buys, sells, reasons = model(prices, volumes, 20, sens / 100, volatilities, rsis1, graph)
    print(simulate(buys, sells, prices, graph = graph))
    winning_trades(buys, sells, prices)
    if graph:
        plt.plot(reasons)
        plt.show()

    print(dt.datetime.now())

    print("indicator based")
    sens, settings_code, rmm = big_boy_best_vars(training_prices, graph, rmm) # <-- CHECKING WHICH INDICATORS TO USE
    print(sens, settings_code, rmm)

    volumes, volatilities, rsis = [[], [], []]

    if int(settings_code / 100):
        volumes = global_volumes

    if int(settings_code % 100 / 10):
        volatilities = volatility(prices)
        
    if int(settings_code % 100 % 10):
        rsis = calc_rsis(prices, momentum_multiplier = rmm)

    buys, sells, reasons = model(prices, volumes, 20, sens / 100, volatilities, rsis, True)
    print(simulate(buys, sells, prices, 10000, True, False))
    winning_trades(buys, sells, prices)
    if graph:
        plt.plot(reasons)
        plt.show()
    print(dt.datetime.now())

    macd, macd_signal = calc_macd(prices)

    fig = plt.figure()

    ax1 = plt.subplot2grid((8,1), (0,0), rowspan = 5)
    ax2 = plt.subplot2grid((8,1), (5,0), sharex = ax1)
    ax3 = plt.subplot2grid((8,1), (6,0), sharex = ax1)
    ax4 = plt.subplot2grid((8,1), (7,0), sharex = ax1)

    ax1.plot(prices)
    ax2.plot(macd)
    ax2.plot(macd_signal)
    ax3.plot(rsis1)
    ax4.plot(volumes)

    plt.show()

    return sens, settings_code, rmm

def get_held_value():
    invested_amount = 0

    stock_invested = (["Neither Held"] + list(rh.account.build_holdings().keys()))[-1]
    print(list(rh.account.build_holdings().keys()))
    print("Checking quantity of", stock_invested)
    if stock_invested == "TQQQ":
        invested_amount += float(rh.account.build_holdings()[stock_invested]["quantity"])
    if stock_invested == "SQQQ":
        invested_amount -= float(rh.account.build_holdings()[stock_invested]["quantity"])

    return invested_amount

def signal(new_signal, buys, sells):

    plt.title(dt.datetime.now())
    plt.scatter(range(len(prices[-50:])), buys[-50:], c = "green")
    plt.scatter(range(len(prices[-50:])), sells[-50:], c = "red")
    plt.plot(prices[-50:])
    plt.ylim(np.min(prices[-50:]) - 5, np.max(prices[-50:]) + 5)
    plt.savefig("last-action.png")
    plt.close()

    if stock_object.last_signal != new_signal:

        if new_signal:
            if stock_object.invested_amount:
                quantity = get_held_value()
                print("selling SQQQ")
                rh.orders.order_sell_fractional_by_quantity("sqqq", -quantity)

                tries = 0
                while float(get_held_value()):
                    time.sleep(30)
                    tries += 1                
                    if tries == 10:
                        print(rh.orders.cancel_all_stock_orders())
                        print("Canceled...")
                        print("selling SQQQ")
                        rh.orders.order_sell_fractional_by_quantity("sqqq", -quantity)
                        tries = 0
                        print("Reordered...") 
                    print("Rechecking...")
                    pass

                stock_object.buying_power = float(rh.profiles.load_account_profile("buying_power"))
                print("Sold {} {} for {} USD at price {}".format(quantity, "sqqq", stock_object.buying_power * .95, round(float(rh.stocks.get_latest_price("sqqq")[0]), 2)))

                with open("tqsqtradehistory.txt", "a") as txt:
                    txt.write("SOLD | " + str(quantity) + " of " + "sqqq" + " for " + str(stock_object.buying_power * .95) + "USD at a price of " + str( round(float(rh.stocks.get_latest_price("sqqq")[0]), 2)) + " | " + str(dt.datetime.now()) + "\n")

            print("buying TQQQ")
            rh.orders.order_buy_fractional_by_price("tqqq", round(stock_object.buying_power * .95, 2))
            
            tries = 0
            while not float(get_held_value()):
                time.sleep(30)
                tries += 1                
                if tries == 10:
                    print(rh.orders.cancel_all_stock_orders())
                    print("Canceled...")
                    print("buying TQQQ")
                    rh.orders.order_buy_fractional_by_price("tqqq", round(stock_object.buying_power * .95, 2))
                    tries = 0
                    print("Reordered...") 
                print("Rechecking...")
                pass

            quantity = float(get_held_value())
            print("Bought {} {} for {} USD at price {}".format(quantity, "tqqq", stock_object.buying_power * .95, round(float(rh.stocks.get_latest_price("tqqq")[0]), 2)))

            with open("tqsqtradehistory.txt", "a") as txt:
                txt.write("BOUGHT | " + str(quantity) + " of " + "tqqq" + " for " + str(stock_object.buying_power * .95) + "USD at a price of " + str( round(float(rh.stocks.get_latest_price("tqqq")[0]), 2)) + " | " + str(dt.datetime.now()) + "\n")


            stock_object.invested_amount += quantity
            stock_object.buying_power = 0
        else:
            if stock_object.invested_amount:
                quantity = get_held_value()
                print("selling TQQQ")
                rh.orders.order_sell_fractional_by_quantity("tqqq", quantity)

                tries = 0
                while float(get_held_value()):
                    time.sleep(30)
                    tries += 1                
                    if tries == 10:
                        print(rh.orders.cancel_all_stock_orders())
                        print("Canceled...")
                        print("selling TQQQ")
                        rh.orders.order_sell_fractional_by_quantity("tqqq", quantity)
                        tries = 0
                        print("Reordered...") 
                    print("Rechecking...")
                    pass

                stock_object.buying_power = float(rh.profiles.load_account_profile("buying_power"))
                print("Sold {} {} for {} USD at price {}".format(quantity, "tqqq", stock_object.buying_power * .95, round(float(rh.stocks.get_latest_price("tqqq")[0]), 2)))

                with open("tqsqtradehistory.txt", "a") as txt:
                    txt.write("SOLD | " + str(quantity) + " of " + "tqqq" + " for " + str(stock_object.buying_power * .95) + "USD at a price of " + str( round(float(rh.stocks.get_latest_price("tqqq")[0]), 2)) + " | " + str(dt.datetime.now()) + "\n")

            print("buying SQQQ")
            rh.orders.order_buy_fractional_by_price("sqqq", round(stock_object.buying_power * .95, 2))
            
            tries = 0
            while not float(get_held_value()):
                time.sleep(30)
                tries += 1                
                if tries == 10:
                    print(rh.orders.cancel_all_stock_orders())
                    print("Canceled...")
                    print("buying SQQQ")
                    rh.orders.order_buy_fractional_by_price("sqqq", round(stock_object.buying_power * .95, 2))
                    tries = 0
                    print("Reordered...")               
                print("Rechecking...")
                pass

            quantity = float(get_held_value())
            print("Bought {} {} for {} USD at price {}".format(quantity, "sqqq", stock_object.buying_power * .95, round(float(rh.stocks.get_latest_price("sqqq")[0]), 2)))

            with open("tqsqtradehistory.txt", "a") as txt:
                txt.write("BOUGHT | " + str(quantity) + " of " + "sqqq" + " for " + str(stock_object.buying_power * .95) + "USD at a price of " + str( round(float(rh.stocks.get_latest_price("sqqq")[0]), 2)) + " | " + str(dt.datetime.now()) + "\n")


            stock_object.invested_amount += quantity
            stock_object.buying_power = 0

    else:
        print("No change")

    stock_object.last_bs_price = prices[-1]
    stock_object.last_signal = new_signal

class Stock:

    def __init__(self, buying_power = 10000, invested_amount = 0):
        self.buying_power = buying_power
        self.last_bs_price = 0
        #bs meaning buy/sell
        self.last_signal = 2
        self.invested_amount = invested_amount
        self.traded = False

        if self.invested_amount > 0:
            self.last_bs_price = float(rh.stocks.get_latest_price("tqqq")[0]) / self.invested_amount
            self.last_signal = 1
        elif self.invested_amount < 0:
            self.last_bs_price = float(rh.stocks.get_latest_price("sqqq")[0]) / -self.invested_amount
            self.last_signal = 0

        print("Stock created with", self.buying_power, "buying power and", self.invested_amount, "of", names[self.last_signal], "held")

def find_total_possible_gains():

    gains = 0
    for price in range(len(prices) - 1):
        gains += 100 * abs(prices[price] - prices[price - 1]) / prices[price - 1]

    print("A total of", (100 + gains), "% gains were possible")

def real_backtests(stock = "tqqq", interval = "1d", time_range = 12, graph = True, training_range = .25):
    create_data(stock, range = time_range, interval = interval)
    training_period = int(len(prices) * training_range)
    buys = [0 for x in range(training_period)]
    sells = [0 for x in range(training_period)]

    for x in range(training_period, len(prices)):
        print(x, "1")
        backtest_prices = prices[:x + 1]
        print(x, "2")
        sens, settings_code, rmm = find_best_rsi_vars(backtest_prices, graph = False)
        print(x, "3")
        sens, settings_code, rmm = big_boy_best_vars(backtest_prices, graph = False, rmm = rmm)
        print(x, "4")
        
        volumes, volatilities, rsis = [[], [], []]

        if int(settings_code / 100):
            volumes = global_volumes

        if int(settings_code % 100 / 10):
            volatilities = volatility(prices)

        if int(settings_code % 100 % 10):
            rsis = calc_rsis(prices, momentum_multiplier = rmm)
        
        print(x, "5")

        last_buys, last_sells, filler = model(backtest_prices, volumes, 20, sens / 100, volatilities, rsis)
        print(x, "6")
        buys.append(last_buys[-1])
        sells.append(last_sells[-1])
        print(x, "7")

    money = simulate(buys, sells, prices, graph = graph, testing = False)
    winning_trades(buys, sells, prices)
    if graph:
        plt.plot(prices)
        plt.scatter(buys)
        plt.scatter(sells)
        plt.show()

    return money


def mathstrats(prices, backtrack = 0, timesteps = 50, scale = 20, show_initial = False):
    macds, filler = calc_macd(prices)
    rsis = calc_rsis(prices)

    current = len(prices)

    inputmacds = macds.copy()[:current - backtrack]
    inputrsis = rsis.copy()[:current - backtrack]
    inputprices = prices.copy()[:current - backtrack]

    if show_initial:
        vfield.create_vector_field(inputmacds, inputrsis, inputprices)
        vfield.simplify_vector_field(inputmacds, inputrsis, inputprices, scale = scale)
        vfield.create_vector_field(inputmacds, inputrsis)
        vfield.simplify_vector_field(inputmacds, inputrsis, scale = scale)

    prediction, path = vfield.predict(inputmacds, inputrsis, prices, input = [inputmacds[-1], inputrsis[-1], inputprices[-1]], timesteps = timesteps, scale = scale)
    vfield.create_vector_field(*path, show = False)
    vfield.create_vector_field(macds[current - backtrack - 1:], rsis[current - backtrack - 1:], prices[current - backtrack - 1:])

    prediction, path = vfield.predict(inputmacds, inputrsis, input = [macds[-1], inputrsis[-1]], timesteps = timesteps, scale = scale)
    vfield.create_vector_field(*path, show = False)
    vfield.create_vector_field(macds[current - backtrack - 1:], rsis[current - backtrack - 1:])

#auto_trade("1h", 1)
#show_best_graph("tqqq", "1h", 12, False, 100)
#real_backtests(interval = "1h", time_range = 12, training_range = .75)

#find_total_possible_gains()

create_data("tqqq", interval = "1m", range = 6/30)
mathstrats(prices, backtrack = 25, show_initial = True)
#macd_rsi_strat(prices, rsis = calc_rsis(prices), graph = True)