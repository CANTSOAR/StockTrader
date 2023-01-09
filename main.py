import requests
from bs4 import BeautifulSoup as BS

zacks_headers = {
    "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36"
}
zacks_login = {
    "force_login":"true",
    "username":"REDACTED",
    "password":"REDACTED"
}

with requests.Session() as s:
    zacks = s.post("https://www.zacks.com/", headers=zacks_headers, data=zacks_login)
    zacks = s.get("https://www.zacks.com/blackboxtrader/?icid=investment-services-ultimate-overview-nav_tracking-zcom-main_menu_wrapper-black_box_trader", headers=zacks_headers)
    zacks_soup = BS(zacks.content, features="html.parser")

zacks_recs = []

for body in zacks_soup.find_all("tbody"):
    if body.tr.td["class"][0] == "symbol":
        for tr in body.find_all("tr"):
            zacks_recs.append(tr.td.button["rel"])

print(zacks_recs)