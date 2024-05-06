import requests
from flask import Flask, make_response
from bs4 import BeautifulSoup as bs

stock_id = input("請輸入股票代碼: ")
print("目前最新到民國112年第4季")
year = input("請輸入民國年份: ")
season = input("請輸入第幾季: ")

data = {
    "ncodeURIComponent": "1",
    "step": "1",
    "firstin": "1",
    "off": "1",
    "queryName": "co_id",
    "inpuType": "co_id",
    "TYPEK": "all",
    "isnew": "false",
    "co_id": stock_id,
    "year": year,
    "season": season
}

response = requests.post("https://mops.twse.com.tw/mops/web/ajax_t164sb03", data = data)

soup = bs(response.text, features = "html.parser")
# print(soup.prettify())
html_content = soup.find_all("table")[1].prettify()

app = Flask(__name__)

@app.route("/get_table", methods = ["GET"])
def get_table():
    response = make_response(html_content)
    response.mimetype = "text/html"
    return response

if __name__ == "__main__":
    app.run()