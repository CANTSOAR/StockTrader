import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt

sym = input("Make me money with: ")

global money
money = int(input("How much we starting with: "))

global prices
global macds
prices, macds = [], []

global macd_initiated
global difpos
macd_initiated, difpos = False, False

global rsi_initiated, rsi_parttwo
rsi_initiated, rsi_parttwo = False, False

global holdings_update
holdings_update = False

#backtesting
def backtest(sym, lag):

    today = dt.date.today()

    start = dt.date(day=1, month=1, year=2000)

    end = today - dt.timedelta(days = lag)

    dates = ["M", "T", "W", "Th", "F", "S", "Su"]

    #print("Current Date:", end - dt.timedelta(days = 1), dates[(end - dt.timedelta(days = 1)).weekday()])

    start = str(start)
    end = str(end)

    ticker = yf.Ticker(sym)

    global data
    data = ticker.history(start = start, end = end)

#find out how far back to look at data
def volatility(sym):

    ticker = yf.Ticker(sym)

    closes = ticker.history(period="30d")["Close"]

    today = np.std(closes)

    #lblength = lblength * int(1 + (today - yes) / np.std(closes[1:]))

    global data
    data = ticker.history(interval="1d", period="max")

#getting ema of anything
def ema(data, length):

    weight = 2 / (length + 1)

    ema = data[0]
    for row in range(len(data)-1):
        ema = data[row + 1] * weight + ema * (1-weight)

    return ema

#getting macd
def macd(src = "Close", fast = 12, slow = 26, signal = 9):

    macd = 0

    global macd_initiated, rsi_parttwo, holdings_update

    if not macd_initiated:
        for row in range(slow):
            prices.append(data[src][row])

        macd = ema(prices, fast) - ema(prices, slow)
        macds.append(macd)

        for row in range(slow, data["Open"].size):
            prices.append(data[src][row])
            macd = ema(prices, fast) - ema(prices, slow)
            macds.append(macd)
    else:
        if len(prices) != data[src].size:
            if data[src][-1] == data[src][-1]:
                prices.append(data[src][-1])

            rsi_parttwo = True
            holdings_update = True
    
        if data[src][-1] != prices[-1]:
            prices[-1] = data[src][-1]

            rsi_parttwo = True
            holdings_update = True
            
        macd = ema(prices, fast) - ema(prices, slow)
        macds.append(macd)

    signal = ema(macds, signal)
    
    if not macd_initiated:
        global difpos

        if macd - signal >= 0:
            difpos = True
        else:
            difpos = False

        macd_initiated = True

    return macd, signal

#getting rsi
def rsi(sens = 14):
    
    global avggain, avgloss, rsi_initiated, rsi_parttwo

    if not rsi_initiated:
        avggain = 0
        avgloss = 0

        for row in range(1, sens):
            gainloss = data["Close"][row] - data["Close"][row - 1]
            if gainloss > 0:
                avggain += gainloss
            elif gainloss < 0:
                avgloss += -gainloss

        avggain /= sens
        avgloss /= sens

        for row in range(sens, data["Open"].size):
            gainloss = data["Close"][row] - data["Close"][row - 1]
            if gainloss > 0:
                avggain = (avggain * (sens - 1) + gainloss)/sens
                avgloss = (avgloss * (sens - 1))/sens
            elif gainloss < 0:
                avgloss = (avgloss * (sens - 1) - gainloss)/sens
                avggain = (avggain * (sens - 1))/sens

    if rsi_parttwo and rsi_initiated:
        gainloss = data["Close"][-1] - data["Close"][-2]
        if gainloss > 0:
            avggain = (avggain * (sens - 1) + gainloss)/sens
            avgloss = (avgloss * (sens - 1))/sens
        elif gainloss < 0:
            avgloss = (avgloss * (sens - 1) - gainloss)/sens
            avggain = (avggain * (sens - 1))/sens

        rsi_parttwo = False
        
    rs = avggain/avgloss

    rsi_initiated = True

    return 100 - 100/(1 + rs)

#getting obv
def obv():

    obv = 0

    for row in range(data["Close"].size):
        if data["Close"][row] > data["Close"][row - 1]:
            obv += data["Volume"][row]
        elif data["Close"][row] < data["Close"][row - 1]:
            obv -= data["Volume"][row]

    return obv

#main function
def makememoney(sym, money, wknds):

    lessgo = True

    bought, sold = False, False
    boughtprice = 0
    soldprice = 0

    while lessgo:
        if not wknds and dt.datetime.today().weekday() >= 5:
            continue

        volatility(sym)

        sell, buy = 0, 0

        local_macd = macd()[0]
        macddif = local_macd - macd()[1]

        if macddif > 1000:
            sell += 1
        elif macddif < -1000:
            buy += 1
        elif sell or buy and abs(macddif) == 0.1 * local_macd:
            sell += 1
            buy += 1

        local_rsi = rsi()

        if local_rsi > 70:
            sell += 1
        elif local_rsi < 30:
            buy += 1

        local_obv = obv()

        if buy == 2:
            bought = True
            sold = False
            print("Buy now, your money:", money)
            boughtprice = data["Close"][-1]
        elif sell == 2:
            sold = True
            bought = False
            print("Sell now, your money:", money)
            soldprice = data["Close"][-1]

        if bought:
            print("You bought and have: ", money * (1 + (data["Close"][-1] - boughtprice) / data["Close"][-1]))
        elif sold: 
            print("You sold and have: ", money * (1 + (soldprice - data["Close"][-1]) / data["Close"][-1]))

#backtesting pt 2
def makemebacktest(sym, money, lag):

    bought, sold = False, False
    mostmoney = money

    while lag >= 0:
        global holdings_update, difpos

        backtest(sym, lag)

        sell, buy = 0, 0

        macdO = macd()

        local_macd = macdO[0]
        macddif = local_macd - macdO[1]

        if macddif >= 0:
            newdifpos = True
        else:
            newdifpos = False

        if newdifpos != difpos:
            if difpos:
                sell += 1
            else:
                buy += 1

        difpos = newdifpos

        local_rsi = rsi()

        if local_rsi > 70 or local_rsi < 30:
            sell += 1
            buy += 1

        local_obv = obv()

        if holdings_update:
            if bought:
                money *= (1 + (prices[-1] - prices[-2]) / prices[-2])
                #print("You bought and have: ", money)
            elif sold:
                money *= (1 + (prices[-2] - prices[-1]) / prices[-2])
                #print("You sold and have: ", money)

            if buy == 2:
                bought = True
                sold = False
                print("Buy now, your money:", money, "bought at:", prices[-1])
                print(data.index[-1])
            elif sell == 2:
                sold = True
                bought = False
                print("Sell now, your money:", money, "sold at:", prices[-1])
                print(data.index[-1])

            holdings_update = False

        if money > mostmoney:
            mostmoney = money
        elif money <= 0.95 * mostmoney:
            mostmoney = money
            if bought or sold:
                bought = False
                sold = False
                print("Cutting Losses at: ", money)
                print(data.index[-1])

        #print("Current dif", macddif, "Current RSI", local_rsi, "Current price", prices[-1], "Money:", money, "difpos", difpos)
        #print("Current MACD", local_macd, "Current sig", macdO[1], "Current dif", macddif, "Current RSI", local_rsi, "Current buy and sell", buy, sell)
        lag -= 1

makemebacktest(sym, money, 365)