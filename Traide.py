import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

lag = 15

alldata = {
    "1m":pd.DataFrame(),
    "5m":pd.DataFrame(),
    "15m":pd.DataFrame(),
    "30m":pd.DataFrame(),
    "1h":pd.DataFrame(),
    "1d":pd.DataFrame(),
    "5d":pd.DataFrame(),
    "1wk":pd.DataFrame(),
    "1mo":pd.DataFrame(),
    "3mo":pd.DataFrame()
}

alldate = {
    "1m":[],
    "5m":[],
    "15m":[],
    "30m":[],
    "1h":[],
    "1d":[],
    "5d":[],
    "1wk":[],
    "1mo":[],
    "3mo":[],
    "lag":lag + 1
}

currents = {
    "1m":[],
    "5m":[],
    "15m":[],
    "30m":[],
    "1h":[],
    "1d":[],
    "5d":[],
    "1wk":[],
    "1mo":[],
    "3mo":[]
}

smas = {
    "1m":[],
    "5m":[],
    "15m":[],
    "30m":[],
    "1h":[],
    "1d":[],
    "5d":[],
    "1wk":[],
    "1mo":[],
    "3mo":[]
}

emas = {
    "1m":[],
    "5m":[],
    "15m":[],
    "30m":[],
    "1h":[],
    "1d":[],
    "5d":[],
    "1wk":[],
    "1mo":[],
    "3mo":[]
}

vwaps = {
    "1m":[],
    "5m":[],
    "15m":[],
    "30m":[],
    "1h":[],
    "1d":[],
    "5d":[],
    "1wk":[],
    "1mo":[],
    "3mo":[]
}

macds = {
    "1m":[],
    "5m":[],
    "15m":[],
    "30m":[],
    "1h":[],
    "1d":[],
    "5d":[],
    "1wk":[],
    "1mo":[],
    "3mo":[]
}

signallines = {
    "1m":[],
    "5m":[],
    "15m":[],
    "30m":[],
    "1h":[],
    "1d":[],
    "5d":[],
    "1wk":[],
    "1mo":[],
    "3mo":[]
}

stochks = {
    "1m":[],
    "5m":[],
    "15m":[],
    "30m":[],
    "1h":[],
    "1d":[],
    "5d":[],
    "1wk":[],
    "1mo":[],
    "3mo":[]
}

stochds = {
    "1m":[],
    "5m":[],
    "15m":[],
    "30m":[],
    "1h":[],
    "1d":[],
    "5d":[],
    "1wk":[],
    "1mo":[],
    "3mo":[]
}

rsis = {
    "1m":[],
    "5m":[],
    "15m":[],
    "30m":[],
    "1h":[],
    "1d":[],
    "5d":[],
    "1wk":[],
    "1mo":[],
    "3mo":[]
}

atrs = {
    "1m":[],
    "5m":[],
    "15m":[],
    "30m":[],
    "1h":[],
    "1d":[],
    "5d":[],
    "1wk":[],
    "1mo":[],
    "3mo":[]
}

adxs = {
    "1m":[],
    "5m":[],
    "15m":[],
    "30m":[],
    "1h":[],
    "1d":[],
    "5d":[],
    "1wk":[],
    "1mo":[],
    "3mo":[]
}

ccis = {
    "1m":[],
    "5m":[],
    "15m":[],
    "30m":[],
    "1h":[],
    "1d":[],
    "5d":[],
    "1wk":[],
    "1mo":[],
    "3mo":[]
}

aroons = {
    "1m":[],
    "5m":[],
    "15m":[],
    "30m":[],
    "1h":[],
    "1d":[],
    "5d":[],
    "1wk":[],
    "1mo":[],
    "3mo":[]
}

bbandss = {
    "1m":[],
    "5m":[],
    "15m":[],
    "30m":[],
    "1h":[],
    "1d":[],
    "5d":[],
    "1wk":[],
    "1mo":[],
    "3mo":[]
}

chaikins = {
    "1m":[],
    "5m":[],
    "15m":[],
    "30m":[],
    "1h":[],
    "1d":[],
    "5d":[],
    "1wk":[],
    "1mo":[],
    "3mo":[]
}

obvs = {
    "1m":[],
    "5m":[],
    "15m":[],
    "30m":[],
    "1h":[],
    "1d":[],
    "5d":[],
    "1wk":[],
    "1mo":[],
    "3mo":[]
}


#getting dates to get maximum info for ticker
def date(interval):

    today = dt.date.today()

    if interval == "1m":
        start1 = today - dt.timedelta(days = 29)
        end1 = today - dt.timedelta(days = 22)

        start2 = today - dt.timedelta(days = 21)
        end2 = today - dt.timedelta(days = 14)

        start3 = today - dt.timedelta(days = 13)
        end3 = today - dt.timedelta(days = 6)

        start4 = today - dt.timedelta(days = 5)

        return start1, end1, start2, end2, start3, end3, start4, today

    elif interval[-1] == "m": 
        start = today - dt.timedelta(days = 59)
    elif interval[-1] == "h":
        start = today - dt.timedelta(days = 729)
    else:
        start = dt.date(year = 1970, month = 1, day = 1)

    start = str(start.year) + "-" + str(start.month) + "-" + str(start.day)
    end = str(today.year) + "-" + str(today.month) + "-" + str(today.day)

    return start, end

#use lag to simulate an end date
def lagenddate(data, interval):

    locallag = lag

    if len(alldate[interval]) != alldate["lag"] - lag:
        lagamount = data["Open"].size - 1

        print(lagamount)

        testdate = str(data.index[lagamount].year) + "-" + str(data.index[lagamount].month) + "-" + str(data.index[lagamount].day)

        while locallag != 0:
            print(interval, lagamount, str(data.index[lagamount].year) + "-" + str(data.index[lagamount].month) + "-" + str(data.index[lagamount].day), testdate, locallag)
            lagamount -= 1

            if str(data.index[lagamount].year) + "-" + str(data.index[lagamount].month) + "-" + str(data.index[lagamount].day) != testdate:
                locallag -= 1
                testdate = str(data.index[lagamount].year) + "-" + str(data.index[lagamount].month) + "-" + str(data.index[lagamount].day)

        alldate[interval].append(lagamount)
        print(alldate[interval])
        
    return data[:alldate[interval][len(alldate[interval]) - 1]]

#grabbing data of a particular stock over a period of time
def stock_data(sym, interval):

    if alldata[interval].empty:
        ticker = yf.Ticker(sym)

        dates = date(interval)

        if interval == "1m":
            data = ticker.history(start=dates[0], end=dates[1], interval=interval)
            data = data.append(ticker.history(start=dates[2], end=dates[3], interval=interval))
            data = data.append(ticker.history(start=dates[4], end=dates[5], interval=interval))
            data = data.append(ticker.history(start=dates[6], end=dates[7], interval=interval))
            
            alldata["1m"] = data
            
            return lagenddate(alldata["1m"], interval)

        data = ticker.history(start=dates[0], end=dates[1], interval=interval)
        alldata[interval] = data

        print(data)

    return lagenddate(alldata[interval], interval)

#grabbing current stock price
def current(sym, interval):
    
    if len(currents[interval]) != alldate["lag"] - lag:
        data = stock_data(sym, interval)
    
        currents[interval].append(data["Close"][-1])

    return currents[interval][-1]

#calculating the simple moving average SMA
def sma(sym, interval, length, src):
    
    if len(smas[interval]) != alldate["lag"] - lag:
        data = stock_data(sym, interval)

        count = 0
        for price in data[src][-length:]:
            count += price
        
        smas[interval].append(count/length)
    
    return smas[interval][-1]

#calculating the exponential moving average
def ema(data, length):

    weight = 2 / (length + 1)

    ema = data[0]
    for row in range(data.size-1):
        ema = data[row + 1] * weight + ema * (1-weight)

    return ema

#calculating the exponential moving average of a stock EMA
def ema_s(sym, interval, length, src):

    if len(emas[interval]) != alldate["lag"] - lag:
        data = stock_data(sym, interval)
        
        emas[interval].append(ema(data[src][-length:], length))

    return emas[interval][-1]

#calculating the volume-weighted average price WVAP
def vwap(sym, interval):

    if len(vwaps[interval]) != alldate["lag"] - lag:
        ticker = yf.Ticker(sym)

        start = dt.date.today() - dt.timedelta(days = lag + 1)
        end = dt.date.today() - dt.timedelta(days = lag)

        start = str(start.year) + "-" + str(start.month) + "-" + str(start.day)
        end = str(end.year) + "-" + str(end.month) + "-" + str(end.day)

        data = ticker.history(start=start, end=end, interval=interval)

        cumulativevol = 0
        cumulativetotal = 0
        for row in range(data["Volume"].size):
            cumulativevol += data["Volume"][row]
            cumulativetotal += (data["High"][row] + data["Low"][row] + data["Close"][row])/3 * data["Volume"][row]

        vwaps[interval].append(cumulativetotal/cumulativevol)

    return vwaps[interval][-1]

#calculating the moving average convergence divergence MACD NOTE ADD SIGNAL LINE LATER: = to 9d ema of macd
def macd(sym, interval, src, fast, slow, signal):

    if len(macds[interval]) != alldate["lag"] - lag:
        data = stock_data(sym, interval)

        prices = []
        macd = 0

        for row in range(slow, data["Open"].size):
            prices.append(data[src][row])

        pricespd = pd.DataFrame(prices)

        macd = ema(pricespd, fast) - ema(pricespd, slow)
        macds[interval].append(macd)

        macdpd = pd.DataFrame(macds[interval])
        signallines[interval].append(ema(macdpd, signal))

    return macds[interval][-1], signallines[interval][-1]

#calculating the stochastic oscillator STOCH
def stoch(sym, interval, sens):

    data = stock_data(sym, interval)

    abshigh = 0
    abslow = 0

    perK = 0
    perKlist = []

    for row in range(sens, data["Open"].size):
        abshigh = 0
        abslow = data["Low"][row - sens]

        for rowin in range(sens):
            if data["Low"][row - rowin] < abslow:
                abslow = data["Low"][row - rowin]

            if data["High"][row - rowin] > abshigh:
                abshigh = data["High"][row - rowin]
    
        perK = 100 * ((data["Close"][row - 1] - abslow)/(abshigh-abslow))
        perKlist.append(perK)

    perKpd = pd.DataFrame(perKlist)
    perD = ema(perKpd, 3)
    
    return perK, perD

#calculating the relative strength index RSI
def rsi(sym, interval, sens):

    data = stock_data(sym, interval)
    
    avggain = 0
    avgloss = 0

    for row in range(sens):
        if data["Close"][row] - data["Open"][row] > 0:
            avggain += data["Close"][row] - data["Open"][row]
        elif data["Close"][row] - data["Open"][row] < 0:
            avgloss += data["Open"][row] - data["Close"][row]

    avggain /= sens
    avgloss /= sens

    for row in range(sens, data["Open"].size - 1):
        if data["Close"][row] - data["Open"][row] > 0:
            avggain = (avggain * (sens - 1) + data["Close"][row] - data["Open"][row])/sens
            avgloss = (avgloss * (sens - 1))/sens
        elif data["Close"][row] - data["Open"][row] < 0:
            avgloss = (avgloss * (sens - 1) + data["Open"][row] - data["Close"][row])/sens
            avggain = (avggain * (sens - 1))/sens
        
    rs = avggain/avgloss

    return 100 - 100/(1 + rs)

#calculating the average true range ATR
def atr(sym, interval, sens):

    data = stock_data(sym, interval)

    atr = 0
    truerange = data["High"][0] - data["Low"][0]
    hilorange = 0
    higap = 0
    logap = 0

    for row in range(1,sens):
        hilorange = data["High"][row] - data["Low"][row]
        higap = abs(data["High"][row] - data["Close"][row - 1])
        logap = abs(data["Low"][row] - data["Close"][row - 1])

        if hilorange > higap and hilorange > logap:
            truerange += hilorange
        elif higap > hilorange and higap > logap:
            truerange += higap
        else:
            truerange += logap

    atr = truerange/sens

    for row in range(sens,data["Open"].size):
        hilorange = data["High"][row] - data["Low"][row]
        higap = abs(data["High"][row] - data["Close"][row - 1])
        logap = abs(data["Low"][row] - data["Close"][row - 1])

        if hilorange > higap and hilorange > logap:
            truerange = hilorange
        elif higap > hilorange and higap > logap:
            truerange = higap
        else:
            truerange = logap

        atr = (atr * (sens - 1) + truerange)/sens

    return atr

#calculating the average directional index ADX
def adx(sym, interval, sens):
    
    data = stock_data(sym, interval)

    truerange = data["High"][0] - data["Low"][0]
    hilorange = 0
    higap = 0
    logap = 0
    plusdm = 0
    minusdm = 0

    for row in range(1,sens):
        hilorange = data["High"][row] - data["Low"][row]
        higap = abs(data["High"][row] - data["Close"][row - 1])
        logap = abs(data["Low"][row] - data["Close"][row - 1])

        plusdm = data["High"][row] - data["High"][row - 1]
        minusdm = data["Low"][row - 1] - data["Low"][row]

        if hilorange > higap and hilorange > logap:
            truerange += hilorange
        elif higap > hilorange and higap > logap:
            truerange += higap
        else:
            truerange += logap

        if plusdm > minusdm and plusdm > 0:
            minusdm += 0
        elif minusdm > plusdm and minusdm > 0:
            plusdm += 0
        else:
            plusdm += 0
            minusdm += 0

    smoothedtruerange = truerange
    smoothedplusdm = plusdm
    smoothedminusdm = minusdm

    plusdmi = 0
    minusdmi = 0
    dmx = []

    for row in range(sens,data["Open"].size):
        hilorange = data["High"][row] - data["Low"][row]
        higap = abs(data["High"][row] - data["Close"][row - 1])
        logap = abs(data["Low"][row] - data["Close"][row - 1])
        
        plusdm = data["High"][row] - data["High"][row - 1]
        minusdm = data["Low"][row - 1] - data["Low"][row]

        if hilorange > higap and hilorange > logap:
            truerange = hilorange
        elif higap > hilorange and higap > logap:
            truerange = higap
        else:
            truerange = logap

        if plusdm > minusdm and plusdm > 0:
            minusdm = 0
        elif minusdm > plusdm and minusdm > 0:
            plusdm = 0
        else:
            plusdm = 0
            minusdm = 0

        smoothedtruerange = smoothedtruerange - smoothedtruerange/sens + truerange
        smoothedplusdm = smoothedplusdm - smoothedplusdm/sens + plusdm
        smoothedminusdm = smoothedminusdm - smoothedminusdm/sens + minusdm

        plusdmi = 100 * smoothedplusdm / smoothedtruerange
        minusdmi = 100 * smoothedminusdm / smoothedtruerange
        dmx.append(100 * abs(plusdmi - minusdmi) / (plusdmi + minusdmi))
    
    adx = 0
    
    for val in dmx[:sens]:
        adx += val

    for val in dmx[sens:]:
        adx = (adx * (sens - 1) + val) / sens

    return adx

#calculating the commodity channel index CCI
def cci(sym, interval, length):

    data = stock_data(sym, interval)

    typical = 0
    devtotal = 0

    for row in range(data["Open"].size):
        typical += (data["High"][row] + data["Close"][row] + data["Low"][row]) / 3

    ma = typical / length

    for row in range(data["Open"].size):
        typical = (data["High"][row] + data["Close"][row] + data["Low"][row]) / 3
        devtotal += abs(typical - ma)

    md = devtotal / length

    return (typical - ma) / (0.015 * md)

#calculating the aroon index AROON
def aroon(sym, interval, sens):

    data = stock_data(sym, interval)

    high = 0
    low = data["Low"][data["Open"].size - sens - 1]
    hicount = data["Open"].size - sens
    locount = data["Open"].size - sens

    for row in range(data["Open"].size - sens - 1, data["Open"].size):
        if data["High"][row] > high:
            high = data["High"][row]
            hicount = row
        if data["Low"][row] < low:
            low = data["Low"][row]
            locount = row

    aroonup = (sens - (data["Open"].size - hicount)) / sens * 100
    aroondown = (sens - (data["Open"].size - locount)) / sens * 100

    return aroonup, aroondown

#calculating the bollinger bands BBANDS
def bbands(sym, interval, sens, devs):

    data = stock_data(sym, interval)

    stddev = 0
    mid = 0

    for row in range(data["Open"].size - sens, data["Open"].size):
        mid += data["Close"][row]

    mid = mid / sens

    for row in range(data["Open"].size - sens, data["Open"].size):
        stddev += (data["Close"][row] - mid) ** 2

    stddev = (stddev / sens) ** (1/2)

    high = mid + stddev * devs
    low = mid - stddev * devs

    return high, mid, low

#calculating the chaikin ocillator AD
def ad(sym, interval, fast, slow):

    data = stock_data(sym, interval)

    moneyflow = 0

    if data["High"][0] - data["Low"][0] != 0:
        moneyflow = (data["Close"][0] * 2 - data["Low"][0] - data["High"][0]) / (data["High"][0] - data["Low"][0]) * data["Volume"][0]

    adl = []

    for row in range(1, data["Open"].size - 1):
        if data["High"][row] - data["Low"][row] != 0:
            adl.append(moneyflow + (data["Close"][row] * 2 - data["Low"][row] - data["High"][row]) / (data["High"][row] - data["Low"][row]) * data["Volume"][row])
            moneyflow = (data["Close"][row] * 2 - data["Low"][row] - data["High"][row]) / (data["High"][row] - data["Low"][row]) * data["Volume"][row]
        else:
            moneyflow = 0

    adlpd = pd.DataFrame(adl)

    return ema(adlpd, fast) - ema(adlpd, slow)

#calculating the on-balance volume OBV
def obv(sym, interval):

    data = stock_data(sym, interval)

    obv = 0

    for row in range(1, data["Close"].size):
        if data["Close"][row] > data["Close"][row - 1]:
            obv += data["Volume"][row]
        elif data["Close"][row] < data["Close"][row - 1]:
            obv -= data["Volume"][row]

    return obv

#quick method to call all functions of a stock
def quickrun(sym, interval):

    funclist = {
        "Current":current(sym, interval),
        "SMA":sma(sym, interval, 7, "Close"),
        "EMA":ema_s(sym, interval, 7, "Close"),
        "VWAP":vwap(sym, "5m"),
        "MACD":macd(sym, interval, "Close", 12, 26, 9)[0],
        "MACD Signal":macd(sym, interval, "Close", 12, 26, 9)[1],
        "STOCH %K":stoch(sym, interval, 14)[0],
        "STOCH %D":stoch(sym, interval, 14)[1],
        "RSI":rsi(sym, interval, 14),
        "ATR":atr(sym, interval, 14),
        "ADX":adx(sym, interval, 14),
        "CCI":cci(sym, interval, 20),
        "Aroon High":aroon(sym, interval, 25)[0],
        "Aroon Low":aroon(sym, interval, 25)[1],
        "BBands High":bbands(sym, interval, 20, 2)[0],
        "BBands Mid":bbands(sym, interval, 20, 2)[1],
        "BBands Low":bbands(sym, interval, 20, 2)[2],
        "Chaikin":ad(sym, interval, 3, 10),
        "OBV":obv(sym, interval),
    }

    return funclist

print(ema_s("tsla", "1d", 9, "Close"))

graphs = {
    "Current":[],
    "SMA":[],
    "EMA":[],
    "VWAP":[],
    "MACD":[],
    "MACD Signal":[],
    "STOCH %K":[],
    "STOCH %D":[],
    "RSI":[],
    "ATR":[],
    "ADX":[],
    "CCI":[],
    "Aroon High":[],
    "Aroon Low":[],
    "BBands High":[],
    "BBands Mid":[],
    "BBands Low":[],
    "Chaikin":[],
    "OBV":[],
}

while lag >= 0:

    print("Approx time left:", lag, "seconds")

    funcslist = quickrun("btc-usd", "5m")

    print(funcslist)

    graphs["Current"].append(funcslist["Current"])
    graphs["SMA"].append(funcslist["SMA"])
    graphs["EMA"].append(funcslist["EMA"])
    graphs["VWAP"].append(funcslist["VWAP"])
    graphs["MACD"].append(funcslist["MACD"])
    graphs["MACD Signal"].append(funcslist["MACD Signal"])
    graphs["STOCH %K"].append(funcslist["STOCH %K"])
    graphs["STOCH %D"].append(funcslist["STOCH %D"])
    graphs["RSI"].append(funcslist["RSI"])
    graphs["ATR"].append(funcslist["ATR"])
    graphs["ADX"].append(funcslist["ADX"])
    graphs["CCI"].append(funcslist["CCI"])
    graphs["Aroon High"].append(funcslist["Aroon High"])
    graphs["Aroon Low"].append(funcslist["Aroon Low"])
    graphs["BBands High"].append(funcslist["BBands High"])
    graphs["BBands Mid"].append(funcslist["BBands Mid"])
    graphs["BBands Low"].append(funcslist["BBands Low"])
    graphs["Chaikin"].append(funcslist["Chaikin"])
    graphs["OBV"].append(funcslist["OBV"])

    lag -= 1

figure1 = plt.figure(1, figsize=(80, 80))

currents_chart = figure1.add_subplot(4,4,1)
smas_chart = figure1.add_subplot(4,4,2)
emas_chart = figure1.add_subplot(4,4,3)
vwaps_chart = figure1.add_subplot(4,4,4)
macds_chart = figure1.add_subplot(4,4,5)
stochs_chart = figure1.add_subplot(4,4,6)
rsis_chart = figure1.add_subplot(4,4,7)
atrs_chart = figure1.add_subplot(4,4,8)
adxs_chart = figure1.add_subplot(4,4,9)
ccis_chart = figure1.add_subplot(4,4,10)
aroons_chart = figure1.add_subplot(4,4,11)
bbandss_chart = figure1.add_subplot(4,4,12)
chaikins_chart = figure1.add_subplot(4,4,13)
obvs_chart = figure1.add_subplot(4,4,14)

currents_chart.plot(graphs["Current"])
smas_chart.plot(graphs["SMA"])
emas_chart.plot(graphs["EMA"])
vwaps_chart.plot(graphs["VWAP"])
macds_chart.plot(graphs["MACD"])
macds_chart.plot(graphs["MACD Signal"])
stochs_chart.plot(graphs["STOCH %K"])
stochs_chart.plot(graphs["STOCH %D"])
rsis_chart.plot(graphs["RSI"])
atrs_chart.plot(graphs["ATR"])
adxs_chart.plot(graphs["ADX"])
ccis_chart.plot(graphs["CCI"])
aroons_chart.plot(graphs["Aroon High"])
aroons_chart.plot(graphs["Aroon Low"])
bbandss_chart.plot(graphs["BBands High"])
bbandss_chart.plot(graphs["BBands Mid"])
bbandss_chart.plot(graphs["BBands Low"])
chaikins_chart.plot(graphs["Chaikin"])
obvs_chart.plot(graphs["OBV"])

currents_chart.set_title("Current")
smas_chart.set_title("SMA")
emas_chart.set_title("EMA")
vwaps_chart.set_title("VWAP")
macds_chart.set_title("MACD")
stochs_chart.set_title("STOCHIASTIC")
rsis_chart.set_title("RSI")
atrs_chart.set_title("ATR")
adxs_chart.set_title("ADX")
ccis_chart.set_title("CCI")
aroons_chart.set_title("Aroons")
bbandss_chart.set_title("BBands")
chaikins_chart.set_title("Chaikin")
obvs_chart.set_title("OBV")

plt.show()