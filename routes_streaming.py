import re
import json
import requests
# import numpy as np
import pandas as pd
from openai import OpenAI
from bs4 import BeautifulSoup as bs
from flask_cors import CORS
from flask import Flask, make_response, request, Response, stream_with_context
有料

app = Flask(__name__)
CORS(app, supports_credentials = True)

@app.route("/get_table", methods = ["GET"])
def get_table():
    statement = request.args.get("statement")
    stock_id = request.args.get("stock_id")
    year = request.args.get("year")
    # season = request.args.get("season")
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
    "season": 4 # season
    }
    
    response_code = requests.post(links_dict[statement], data = data)
    soup = bs(response_code.text, features = "html.parser")
    # print(soup.prettify())
    description = soup.find("h4").text
    start = description.find("由")
    end = description.find("公")
    company_name = description[start + 1: end]

    if statement == "權益變動表":
        first_table = soup.find_all("table")[1]
        second_table = soup.find_all("table")[2]
        html_content = str(first_table) + str(second_table)
    else:
        html_content = soup.find_all("table")[1].prettify()

    html_content = re.sub(r"民國", company_name + "民國", html_content) # \d{3}年第\d季

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
    data = json.loads(data["json"])
    quarters = data["xaxisList"]
    values = data["graphData"]
    full_company_names = data["checkedNameList"]
    
    if len(stock_ids) == 1:
        quarters_json = [[] for _ in range(2)]
        full_company_names.remove("公司平均數")
        del values[1]
    else:
        quarters_json = [[] for _ in range(len(values))]
    # print(len(stock_ids))
    # print(len(values), len(quarters))
    # print(values)

    df = pd.DataFrame()
    df["Quarters"] = quarters

    company_name_list = ["Quarters"]

    for i in range(len(values)):
        company_data = data["graphData"][i]["data"]
        company_name = data["graphData"][i]["label"]
        # if len(company_data) < quarters_amount:
            # print(company_data)
            # print(company_name)
        company_name_list.append(company_name)
        df2 = pd.DataFrame(company_data)
        df2 = df2.iloc[:, : -1]
        df2 = df2.dropna(axis = 0)
        df2.set_index(0, inplace = True)
        df = pd.concat([df, df2], axis = 1)

    df.columns = company_name_list

    for i in range(len(values)):
        for j in range(len(quarters)):
            try:
                quarters_json[i].append({"quarter": str(quarters[j]), "value": None if pd.isna(df[company_name_list[i + 1]][j]) else round(df[company_name_list[i + 1]][j], 2)})
            except:
                pass

    dict_format = {
        "companies": 
        [
            {} for _ in range(len(values))
        ]
    }

    for i in range(len(values)):

        first_space = full_company_names[i].find(" ")
        second_space = full_company_names[i].rfind(" ")
        if (first_space and second_space) == -1:
            dict_format["companies"][i]["name"] = full_company_names[i]
            dict_format["companies"][i]["stock_id"] = full_company_names[i]
        else:
            dict_format["companies"][i]["name"] = full_company_names[i][first_space + 1: second_space]
            dict_format["companies"][i]["stock_id"] = full_company_names[i][: first_space]
        dict_format["companies"][i]["quarters"] = quarters_json[i]
        lastest_season = df[df[company_name_list[i + 1]].notna()][company_name_list[i + 1]].iloc[-1]
        last_season = df[df[company_name_list[i + 1]].notna()][company_name_list[i + 1]].iloc[-2]
        last_year_same_season = df.iloc[int(df[df[company_name_list[i + 1]].notna()].index[-1] - 4)][company_name_list[i + 1]]
        dict_format["companies"][i]["historical_data"] = [{
            "name": "歷史最低", 
            "value": round(df[company_name_list[i + 1]].min(), 2)
        }, {
            "name": "歷史最高", 
            "value": round(df[company_name_list[i + 1]].max(), 2)
        }, {
            "name": "最新一季", 
            "value": round(lastest_season, 2)
        }, {
            "name": "上一季", 
            "value": round(last_season, 2)
        }, {
            "name": "去年同季", 
            "value": round(last_year_same_season, 2)
        }]
        dict_format["companies"][i]["growth_rate"] = [{
            "name": "較上一季",
            "value": round((lastest_season - last_season) / last_season, 2)
        }, {
            "name": "較去年同季", 
            "value": round((lastest_season - last_year_same_season) / last_year_same_season, 2)
        }]

    return json.dumps(dict_format, ensure_ascii = False, indent = 2)

@app.route("/get_analysis", methods = ["GET"])
def get_analysis():
    # ratio = input("請輸入想要查看的比率: ")
    # start = input("請輸入開始時間(YYYYQQ): ")
    # end = input("請輸入結束時間(YYYYQQ): ")
    stock_ids = request.args.get("stock_ids")
    ratio = request.args.get("ratio")
    start = request.args.get("start")
    end = request.args.get("end")

    if "," in stock_ids:
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
    data = json.loads(data["json"])

    quarters = data["xaxisList"]

    values = data["graphData"]
    full_company_names = data["checkedNameList"]

    if len(stock_ids) == 1:
        full_company_names.remove("公司平均數")
        del values[1]

    df = pd.DataFrame()
    df["Quarters"] = quarters
    company_name_list = ["Quarters"]

    for i in range(len(values)):
        company_data = data["graphData"][i]["data"]
        company_name = data["graphData"][i]["label"]
        company_name_list.append(company_name)
        df2 = pd.DataFrame(company_data)
        df2 = df2.iloc[:, : -1]
        df2 = df2.dropna(axis = 0)
        df2.set_index(0, inplace = True)
        df = pd.concat([df, df2], axis = 1)

    df.columns = company_name_list

    client = OpenAI(api_key = "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")
    df_test = df.set_index("Quarters").loc[start: end]

    content_message = "我會給您一到三家公司、這些公司平均以及產業平均的某項比率，請依據以下資料來進行分析並給出一份完整的分析報告:\n"
    content_message += f"公司以及平均是{company_name_list[1: ]}，比率是{ratio}，時間是{start}至{end}\n"
    content_message += f"以下是相關資料\n{df_test}\n"
    content_message += f'請給我{company_name_list[1: ]}近期的趨勢報告,請以詳細、\
        嚴謹及專業的角度撰寫此報告，並提及重要的數字佐證以及各個欄位比較的分析，\
            不需要加上粗體、額外的符號以及最後不需要講到與我們聯繫和要注意風險等贅字，並請給純文字 reply in 繁體中文'

    messages = [
            {"role": "system", "content":  "你現在是一位專業的證券分析師, 你會統整近期的比率並進行分析, 然後生成一份專業的趨勢分析報告"},
            {"role": "user", "content": content_message} # user就是代表我們
        ]

    def stream_chat_completion(messages):
        response = client.chat.completions.create(
        model = "gpt-3.5-turbo",  # 或 "gpt-4" 根據你的需求
        messages = messages,
        stream = True
    )

        for chunk in response:
            if chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, flush = True)
                yield chunk.choices[0].delta.content.encode("utf-8")



    return Response(stream_with_context(stream_chat_completion(messages)), content_type = "text/plain; charset = utf-8")

if __name__ == "__main__":
    app.run(debug = True)
    