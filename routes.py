import json
import requests
import numpy as np
from bs4 import BeautifulSoup as bs
from flask_cors import CORS
from flask import Flask, make_response, request


app = Flask(__name__)
CORS(app, supports_credentials = True)

@app.route("/get_table", methods = ["GET"])
def get_table():
    statement = request.args.get("statement")
    stock_id = request.args.get("stock_id")
    year = request.args.get("year")
    season = request.args.get("season")
    # statement = input("請輸入想要查詢的報表: ")
    # stock_id = input("請輸入股票代碼: ")
    # print("目前最新到民國112年第4季")
    # year = input("請輸入民國年份: ")
    # season = input("請輸入第幾季: ")

    links_dict = {"資產負債表": "https://mops.twse.com.tw/mops/web/ajax_t164sb03", 
                "綜合損益表": "https://mops.twse.com.tw/mops/web/ajax_t164sb04", 
                "現金流量表": "https://mops.twse.com.tw/mops/web/ajax_t164sb05", 
                "權益變動表": "https://mops.twse.com.tw/mops/web/ajax_t164sb06"}
    
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
    
    response_code = requests.post(links_dict[statement], data = data)
    soup = bs(response_code.text, features = "html.parser")
    # print(soup.prettify())

    if statement == "權益變動表":
        first_table = soup.find_all("table")[1]
        second_table = soup.find_all("table")[2]
        html_content = str(first_table) + str(second_table)
    else:
        html_content = soup.find_all("table")[1].prettify()

    response = make_response(html_content)
    response.mimetype = "text/html"
    return response

@app.route("/get_ratio", methods = ["GET"])
def get_ratio():
    stock_ids = request.args.get("stock_ids")
    ratio = request.args.get("ratio")
    # stock_ids = np.array(input("請輸入股票代碼，並以空格區分: ").split(), dtype = int) # 
    # ratio = input("請輸入想要查看的比率: ")
    if "," in stock_ids:
        # print(stock_ids.split(","))
        stock_ids = stock_ids.split(",")
    else:
        stock_ids = [int(stock_ids)]

    """
    DebtRatio
    LongTermLiabilitiesRatio
    CurrentRatio
    QuickRatio
    InterestCoverage
    AccountsReceivableTurnover
    AccountsReceivableTurnoverDay
    InventoryTurnover
    InventoryTurnoverDay
    TotalAssetTurnover
    GrossMargin
    OperatingMargin
    NetIncomeMargin
    ROA
    ROE
    OperatingCashflowToCurrentLiability
    OperatingCashflowToLiability
    OperatingCashflowToNetProfit
    """

    data = {
    "compareItem": ratio,
    "quarter": "true",
    "ylabel": "%",
    "ys": "0",
    "revenue": "true",
    "bcodeAvg": "true",
    "companyAvg": "true",
    "companyId": list(stock_ids)
    }

    response = requests.post("https://mopsfin.twse.com.tw/compare/data", data = data)

    data = response.json()

    quarters = data["xaxisList"]
    values = data["graphData"]
    full_company_names = data["displayCompanyId"]
    
    if len(stock_ids) == 1:
        quarters_json = [[] for _ in range(2)]
        full_company_names.remove("公司平均數")
        del values[1]
    else:
        quarters_json = [[] for _ in range(len(values))]
    # print(len(stock_ids))
    # print(len(values), len(quarters))
    # print(values)
    for i in range(len(values)):
        for j in range(len(quarters)):
            try:
                quarters_json[i].append({"quarter": str(quarters[j]), "value": round(values[i]["data"][j][1], 2)})
            except:
                pass

    dict_format = {
        "companies": 
        [
            {} for _ in range(len(values))
        ]
    }

    for i in range(len(values)):
        string_values_array = np.array(values[i]["data"])[: len(quarters), 1]
        float_values_array = np.array(string_values_array, dtype = "float64")
        first_space = full_company_names[i].find(" ")
        second_space = full_company_names[i].rfind(" ")
        if (first_space and second_space) == -1:
            dict_format["companies"][i]["name"] = full_company_names[i]
            dict_format["companies"][i]["stock_id"] = full_company_names[i]
        else:
            dict_format["companies"][i]["name"] = full_company_names[i][first_space + 1: second_space]
            dict_format["companies"][i]["stock_id"] = full_company_names[i][: first_space]
        dict_format["companies"][i]["quarters"] = quarters_json[i]
        dict_format["companies"][i]["historical_data"] = [{
            "name": "歷史最低", 
            "value": round(float_values_array.min(), 2)
        }, {
            "name": "歷史最高", 
            "value": round(float_values_array.max(), 2)
        }, {
            "name": "最新一季", 
            "value": round(quarters_json[i][-1]["value"], 2)
        }, {
            "name": "上一季", 
            "value": round(quarters_json[i][-2]["value"], 2)
        }, {
            "name": "去年同季", 
            "value": round(quarters_json[i][-5]["value"], 2)
        }]
        dict_format["companies"][i]["growth_rate"] = [{
            "name": "較上一季",
            "value": round((quarters_json[i][-1]["value"] - quarters_json[i][-2]["value"]) / quarters_json[i][-2]["value"], 2)
        }, {
            "name": "較去年同季", 
            "value": round((quarters_json[i][-1]["value"] - quarters_json[i][-5]["value"]) / quarters_json[i][-5]["value"], 2)
        }]

    return json.dumps(dict_format, ensure_ascii = False, indent = 2)

if __name__ == "__main__":
    app.run(debug = True)