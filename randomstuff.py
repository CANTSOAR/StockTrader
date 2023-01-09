from robin_stocks import robinhood as rh
from matplotlib import pyplot as plt
import datetime as dt

"""print(rh.authentication.login(username="dipalsk@gmail.com", password="NyQuil@44$$", expiresIn=31556926))
print("======================================")
print(rh.crypto.get_crypto_positions())
print(rh.account.get_all_positions())
print(rh.profiles.load_account_profile("buying_power"))
print(rh.crypto.get_crypto_quote("BTC", info="ask_price"))
print(rh.crypto.get_crypto_quote("BTC", info="mark_price"))
print(rh.crypto.get_crypto_quote("ETH", info="ask_price"))
print(rh.crypto.get_crypto_quote("BTC", info="bid_price"))
print(rh.crypto.get_crypto_positions("quantity"))
print(rh.crypto.get_crypto_positions("currency"))

def get_held_value(stock):
    held_stocks = [x["code"] for x in rh.crypto.get_crypto_positions("currency")]
    print(held_stocks)
    
    if stock in held_stocks:
        index_of_given_stock = held_stocks.index(stock)
        return rh.crypto.get_crypto_positions("quantity")[index_of_given_stock]

    return 0

print(get_held_value("ETH"))"""

rh.authentication.login(username = "dipalsk@gmail.com", password = "NyQuil@44$$", expiresIn = 31556926)

def get_held_value():
    invested_amount = 0

    stock_invested = list(rh.account.build_holdings().keys())[0]
    if stock_invested == "TQQQ":
        stock_positions = [0] + [rh.account.build_holdings()[stock_invested]["quantity"]]
        invested_amount += float(stock_positions[-1])
    if stock_invested == "SQQQ":
        stock_positions = [0] + [rh.account.build_holdings()[stock_invested]["quantity"]]
        invested_amount -= float(stock_positions[-1])

    return invested_amount

print(rh.stocks.get_latest_price("tqqq"))
print(float(rh.stocks.get_latest_price("tqqq")[0]) / get_held_value())

get_held_value()