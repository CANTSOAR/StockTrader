import pandas as pd
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

sym = input("Make me money with: ")

#bringing in data
def get_data(sym):

    ticker = yf.Ticker(sym)

    data = ticker.history(interval="1d", period="max")

    return data

#getting ema of anything
def ema(emadata, length):

    weight = 2 / (length + 1)
    sum = 0

    for row in range(length):
        sum += emadata[row]

    ema = sum/length

    for row in range(length, len(emadata)):
        ema = emadata[row] * weight + ema * (1-weight)

    return ema

#getting macd
def macd(src = "Close", fast = 12, slow = 26, signalspeed = 9):

    data = get_data(sym)

    prices = list(data[src])
    macds = list(range(slow, data["Close"].size + 1))
    signals = list(range(slow + signalspeed, data["Close"].size + 2))

    for row in range(slow, data["Open"].size + 1):
        macds[row - slow] = ema(prices[:row], fast) - ema(prices[:row], slow)

        if row >= signalspeed + slow - 1:
            signals[row - signalspeed - slow + 1] = ema(macds[:row - slow + 1], signalspeed)

    for x in range(slow + signalspeed - 2):
        if x < slow - 1:
            macds = ["-"] + macds
        signals = ["-"] + signals

    return macds, signals

#getting rsi
def rsi(sens = 14):

    data = get_data(sym)
    
    avggain = 0
    avgloss = 0
    rsi = 0

    rsis = []

    for row in range(1, sens + 1):
        gainloss = data["Close"][row] - data["Close"][row - 1]

        avggain += max(gainloss, 0)
        avgloss -= min(gainloss, 0)

    avggain /= sens
    avgloss /= sens

    rsi = 100 - 100/(1 + avggain/avgloss)
    rsis.append(rsi)

    for row in range(sens + 1, data["Open"].size):
        gainloss = data["Close"][row] - data["Close"][row - 1]

        avggain = (avggain * (sens - 1) + max(gainloss, 0))/sens
        avgloss = (avgloss * (sens - 1) - min(gainloss, 0))/sens

        rsi = 100 - 100/(1 + avggain/avgloss)
        rsis.append(rsi)

    for x in range(sens):
        rsis = ["-"] + rsis

    return rsis

#getting obv
def klinger(slow = 55, fast = 34, signalspeed = 13):

    data = get_data(sym)

    klingers = list(range(slow, data["Close"].size + 1))
    signals = list(range(slow + signalspeed, data["Close"].size + 2))
    volumeforces = list(data["Volume"])

    volumeforces[0] *= -1

    for row in range(1, data["Close"].size):

        today = data["High"][row] + data["Low"][row] + data["Close"][row]
        yes = data["High"][row - 1] + data["Low"][row - 1] + data["Close"][row - 1]

        if today - yes != 0:
            volumeforces[row] *= (today - yes) / abs(today - yes)

        if volumeforces[row] != volumeforces[row]:
            print(volumeforces[row], today, yes, data["Close"][row], data["Close"][row - 1], row)

    for row in range(slow, data["Close"].size + 1):

        klingers[row - slow] = ema(volumeforces[:row], fast) - ema(volumeforces[:row], slow)

        if row >= signalspeed + slow - 1:
            signals[row - signalspeed - slow + 1] = ema(klingers[:row - slow + 1], signalspeed)
    
    for x in range(slow + signalspeed - 2):
        if x < slow - 1:
            klingers = ["-"] + klingers
        signals = ["-"] + signals

    return klingers, signals

mac = macd()
rs = rsi()
kling = klinger()
prices = get_data(sym)["Close"]

trends = []

for row in range(len(prices) - 1):

    if prices[row + 1] - prices[row] >= 0:
        trends.append("Up")
    else:
        trends.append("Down")

trends.append("-")

main = {
    "MACD": mac[0],
    "MACD Signal": mac[1],
    "RSI": rs,
    "Klinger Osc.": kling[0],
    "Klinger Signal": kling[1],
    "Prices": list(prices),
    "Trends": trends,
}

maindf = pd.DataFrame(main, prices.index.values)

inputs = maindf.drop(columns = "Trends").iloc[67:-1]
outputs = maindf["Trends"][67:-1]

inputs_train, inputs_test, outputs_train, outputs_test = train_test_split(inputs, outputs, test_size=0.05) 

model = DecisionTreeClassifier()
model.fit(inputs, outputs)

predictions = model.predict(inputs_test)

print(model.predict(maindf.drop(columns = "Trends").iloc[-1:]))

print(accuracy_score(outputs_test, predictions))

print()
