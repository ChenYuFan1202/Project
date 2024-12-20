import re
import os
import json
import time
import requests
# import numpy as np
import pandas as pd
import yfinance as yf
from openai import OpenAI
from flask_cors import CORS
from dotenv import load_dotenv
from bs4 import BeautifulSoup as bs
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from flask import Flask, make_response, request
from langchain.chains import load_summarize_chain
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
# from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.output_parsers import StrOutputParser
from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory

load_dotenv()

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
    "year": int(year) - 1911,
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
    html_content = html_content.replace("第4季", "")

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
    以下的比率會先跟自己比再跟產業平均比。
    DebtRatio 判斷是否大於0.5，大於的話再和產業平均比，再大於產業平均就示警，越小越安全。
    LongTermLiabilitiesRatio 判斷是否小於1，如果再小於產業平均再警示。越小代表有較高的融資風險，但也可能是對未來樂觀；大於1代表財務結構安全性高。
    CurrentRatio 判斷是否小於1，小於的話要再和產業平均比，再小於就警示。大於1是安全的，否則也要看是否大於產業平均。
    QuickRatio 判斷是否小於1，小於的話要再和產業平均比，再小於就警示。大於1是安全的，否則也要看是否大於產業平均。
    InterestCoverage 判斷是否小於5，小於的話要再和產業平均比，再小於就警示。
    AccountsReceivableTurnover 這個不判斷
    AccountsReceivableTurnoverDay 這個不判斷
    InventoryTurnover 這個不判斷
    InventoryTurnoverDay 這個不判斷
    TotalAssetTurnover 判斷是否小於0.5，小於的話要再和產業平均比，再小於就警示。
    GrossMargin 
    OperatingMargin 
    NetIncomeMargin 判斷這一期是否小於去年同期，是的話再和產業衰退幅度比，衰退比較大就警示。
    ROA 判斷是否小於6%，小於的話，再和產業平均比，再小於就警示。
    ROE 判斷是否小於8%，小於的話，再和產業平均比，再小於就警示。
    OperatingCashflowToCurrentLiability 判斷是否小於100%，小於的話，再和產業平均比，再小於就警示。
    OperatingCashflowToLiability 
    OperatingCashflowToNetProfit 判斷是否小於80%，小於的話，再和產業平均比，再小於就警示。
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

    for i in range(len(values)):
        for j in range(len(quarters)):
            try: 
                quarters_json[i].append({"quarter": str(quarters[j]), "value": None if pd.isna(df[company_name_list[i + 1]][j]) else round(df[company_name_list[i + 1]][j], 2)})
            except:
                pass

    # df.tail()

    dict_format = {
        "companies": 
        [
            {} for _ in range(len(values))
        ]
    }

    industry_list = full_company_names.copy()

    for i in range(len(values)):
        first_space = full_company_names[i].find(" ")
        second_space = full_company_names[i].rfind(" ")
        if (first_space and second_space) == -1:
            dict_format["companies"][i]["name"] = full_company_names[i]
            dict_format["companies"][i]["stock_id"] = full_company_names[i]
        else:
            dict_format["companies"][i]["name"] = full_company_names[i][first_space + 1: second_space]
            dict_format["companies"][i]["stock_id"] = full_company_names[i][: first_space]

    for i in range(len(stock_ids)):
        # print(industry_list[i])
        left_parenthesis_index = industry_list[i].find("(")
        right_parenthesis_index = industry_list[i].find(")")
        industry = industry_list[i][left_parenthesis_index + 3: right_parenthesis_index]
        # industry_list[i] = industry_list[i][left_parenthesis_index + 3: right_parenthesis_index]
        # print(industry)

        company_name = company_name_list[i + 1]
        # print(company_name)

        # print(df[company_name].iloc[-1])
        # print(df[industry].iloc[-1])
        if ratio == "DebtRatio":
            if df[company_name].iloc[-1] > 50:
                if df[company_name].iloc[-1] > df[industry].iloc[-1]:
                    dict_format["companies"][i]["warning"] = "Y"
                else:
                    dict_format["companies"][i]["warning"] = "N"
            else:
                dict_format["companies"][i]["warning"] = "N"
        elif ratio == "LongTermLiabilitiesRatio"or ratio == "CurrentRatio" or ratio == "QuickRatio":
            if df[company_name].iloc[-1] < 100:
                if df[company_name].iloc[-1] < df[industry].iloc[-1]:
                    dict_format["companies"][i]["warning"] = "Y"
                else:
                    dict_format["companies"][i]["warning"] = "N"
            else:
                dict_format["companies"][i]["warning"] = "N"
        elif ratio == "InterestCoverage":
            if df[company_name].iloc[-1] < 5:
                if df[company_name].iloc[-1] < df[industry].iloc[-1]:
                    dict_format["companies"][i]["warning"] = "Y"
                else:
                    dict_format["companies"][i]["warning"] = "N"
            else:
                dict_format["companies"][i]["warning"] = "N"
        elif ratio == "TotalAssetTurnover":
            if df[company_name].iloc[-1] < 0.5:
                if df[company_name].iloc[-1] < df[industry].iloc[-1]:
                    dict_format["companies"][i]["warning"] = "Y"
                else:
                    dict_format["companies"][i]["warning"] = "N"
            else:
                dict_format["companies"][i]["warning"] = "N"
        elif ratio == "NetIncomeMargin" or ratio == "GrossMargin":
            if df[company_name].iloc[-1] < df[company_name].iloc[-5]:
                if ((df[company_name].iloc[-1] - df[company_name].iloc[-5]) / df[company_name].iloc[-5]) < ((df[industry].iloc[-1] - df[industry].iloc[-5]) / df[industry].iloc[-5]):
                    dict_format["companies"][i]["warning"] = "Y"
                else:
                    dict_format["companies"][i]["warning"] = "N"
            else:
                dict_format["companies"][i]["warning"] = "N"
        elif ratio == "ROA":
            if df[company_name].iloc[-1] < 0.06:
                if df[company_name].iloc[-1] < df[industry].iloc[-1]:
                    dict_format["companies"][i]["warning"] = "Y"
                else:
                    dict_format["companies"][i]["warning"] = "N"
            else:
                dict_format["companies"][i]["warning"] = "N"
        elif ratio == "ROE":
            if df[company_name].iloc[-1] < 0.08:
                if df[company_name].iloc[-1] < df[industry].iloc[-1]:
                    dict_format["companies"][i]["warning"] = "Y"
                else:
                    dict_format["companies"][i]["warning"] = "N"
            else:
                dict_format["companies"][i]["warning"] = "N"  
        elif ratio == "OperatingCashflowToCurrentLiability":
            if df[company_name].iloc[-1] < 1:
                if df[company_name].iloc[-1] < df[industry].iloc[-1]:
                    dict_format["companies"][i]["warning"] = "Y"
                else:
                    dict_format["companies"][i]["warning"] = "N"
            else:
                dict_format["companies"][i]["warning"] = "N"       
        elif ratio == "OperatingCashflowToNetProfit":
            if df[company_name].iloc[-1] < 0.8:
                if df[company_name].iloc[-1] < df[industry].iloc[-1]:
                    dict_format["companies"][i]["warning"] = "Y"
                else:
                    dict_format["companies"][i]["warning"] = "N"
            else:
                dict_format["companies"][i]["warning"] = "N" 

    for i in range(len(values)):   
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

    df_ratio = df.set_index("Quarters").loc[start: end]

    load_dotenv()
    chat_model = ChatOpenAI(model = "gpt-3.5-turbo", api_key = os.getenv("OPENAI_API_KEY"))

    prompt = ChatPromptTemplate.from_messages([
        ("system", "您現在是一位專業的證券分析師，您會統整近期的比率並進行分析，然後生成一份專業的趨勢分析報告。"),
        ("human", "我會給您一到三家公司的財務比率、這些公司的財務比率平均以及產業的財務比率平均值。"
         "請依據以下資料進行分析並給出一份完整的分析報告：\n"
         "公司以及平均名字是{company_name_list1}，比率是{ratio}，時間是{start}至{end}。\n"
         "以下是相關資料：\n{df_ratio}\n"
         "請用繁體中文給我{company_name_list2}近期的趨勢報告，並以詳細、嚴謹及專業的角度撰寫此報告，並提及重要的數字佐證以及各個欄位比較的分析。")
    ])

    str_parser = StrOutputParser()

    financial_ratio_analysis_chain = prompt | chat_model | str_parser
    response = financial_ratio_analysis_chain.invoke({"company_name_list1": company_name_list[1: ], 
                                                      "ratio": ratio, "start": start, "end": end, "df_ratio": df_ratio, 
                                                      "company_name_list2": company_name_list[1: ]})

    return response

@app.route("/get_three_years_ESG_data_format", methods = ["get"])
def three_years_ESG_data_format():
    # stock_id = input("請輸入股票代碼: ")
    stock_id = request.args.get("stock_id")
    year = "2023"
    payload = {
    "companyCode": stock_id, # 要是字串
    "yearList": [year],
    "year": year
    }

    newest_response = requests.post("https://esggenplus.twse.com.tw/api/api/mopsEsg/singleCompanyData", json = payload)

    newest_fields = ["直接(範疇一)溫室氣體排放量(公噸CO₂e)", "能源間接(範疇二)溫室氣體排放量(公噸CO₂e)", "其他間接(範疇三)溫室氣體排放量(公噸CO₂e)",
                    "溫室氣體排放密集度(公噸CO₂e/百萬元營業額)", "再生能源使用率", "用水量(公噸(t))", "用水密集度", "有害廢棄物(公噸(t))",
                    "非有害廢棄物(公噸(t))", "總重量(有害+非有害)(公噸(t))", "廢棄物密集度", "員工福利平均數(每年6/2起公開)(仟元/人)",
                    "員工薪資平均數(每年6/2起公開)(仟元/人)", "非擔任主管職務之全時員工薪資平均數(每年7/1起公開)(仟元/人)", 
                    "非擔任主管職務之全時員工薪資中位數(每年7/1起公開)(仟元/人)", "管理職女性主管占比", "職業災害人數(人)", 
                    "職業災害人數比率", "董事會席次(席)", "獨立董事席次(席)", "女性董事席次(席)", "女性董事比率", "董事出席董事會出席率", 
                    "董事進修時數符合進修要點比率", "公司年度召開法說會次數(次)"]

    oldest_fields = ["直接溫室氣體排放量", "能源間接", "其他間接", "溫室氣體排放密集度", "再生能源使用率", "用水量", "用水密集度", "有害廢棄物", 
                    "非有害廢棄物", "總重量", "廢棄物密集度", "員工福利平均數", "員工薪資平均數", "非擔任主管職務之全時員工薪資平均數",
                    "非擔任主管職務之全時員工薪資中位數", "管理職女性主管占比", "職業災害人數", "職業災害人數比率", "董事會席次", "獨立董事席次",
                    "女性董事席次及比率", "董事出席董事會出席率", "董監事進修時數符合進修要點比率", "公司年度召開法說會次數"]

    data = newest_response.json()["data"]
    company_name = data[0]["companyName"]
    treeModels = data[0]["treeModels"]

    names = []
    values = []
    temp_list = []
    for i in range(len(treeModels)):
        items = treeModels[i]["items"]
        for j in range(len(items)):
            sections = items[j]["sections"]
            # print(sections)
            for k in range(len(sections)):
                name = sections[k]["name"]
                # print(name)
                if name in newest_fields:
                    # print(name)
                    if name == "女性董事比率":
                        names.extend([name, "董事出席董事會出席率", "董事進修時數符合進修要點比率"])
                        value = sections[k]["controls"][0]["value"]
                        value = value[: -1]
                        value = round(float(value) / 100, 4)
                        temp_list.insert(0, value)
                        values.extend(temp_list)
                    elif name == "董事出席董事會出席率" or name == "董事進修時數符合進修要點比率":
                        value = sections[k]["controls"][0]["value"]
                        # print(value)
                        value = value[: -1]
                        value = round(float(value) / 100, 4)
                        temp_list.append(value)
                    else:
                        value = sections[k]["controls"][0]["value"]
                        value = value.replace(",", "")
                        if value[-1] == "%":
                            value = value[: -1]
                            value = round(float(value) / 100, 4)
                        value = round(float(value), 4)
                        values.append(value)
                        names.append(name)

    def call_oldest_response(stock_id, year):
        data = {
            "encodeURIComponent": 1,
            "step": 2,
            "co_id": stock_id, # 可以是數字或文字
            "YEAR": year # 可以是數字或文字
        }
        oldest_response = requests.post("https://mops.twse.com.tw/mops/web/t214sb01", data = data)

        fields = []
        # values = []
        soup = bs(oldest_response.text, features = "html.parser")
        # print(soup.prettify())
        tables = soup.find_all("table")
        target_table = tables[-2]
        trs = target_table.find_all("tr")
        for tr in trs:
            td = tr.find("td", attrs = {"align": "center"})
            if td is None:
                continue
            text = td.text
            end = text.find("(")
            if end == -1:
                text = text.strip()
            else:    
                text = text[: end].strip()
            # print(text)
            # print("-" * 10)
            if text in oldest_fields:
                field = tr.find("td", attrs = {"align": "center"}).text.strip()
                # print(tr)
                # print(field)
                if field == "女性董事席次及比率":
                    fields.extend(["女性董事席次(席)", "女性董事比率"])
                    temp_list = tr.find_all("td", attrs = {"align": "right"})
                    # print(temp_list)
                    number = round(float(temp_list[0].text.strip()[: -1]), 4)
                    ratio = round(float(temp_list[1].text.strip()[: -1]) / 100, 4)
                    # print(number, ratio)
                    values.extend([number, ratio])
                    # print(f"{temp_list[0].text.strip()}({temp_list[1].text.strip()})")
                else: 
                    fields.append(field)
                    value = tr.find("td", attrs = {"align": "right"})
                    # print(value)
                    if value is None:
                        values.append(0)
                    elif value.text == "":
                        values.append(0)  
                    else:
                        value = value.text.strip().replace(",", "")
                        if value[-1] == "%":
                            value = value[: -1]
                            value = round(float(value) / 100, 4)
                            # print(value)
                        elif value[-1] == "人":
                            value = round(float(value[: -1]), 4)
                            # print(value)
                        else:
                            value = round(float(value), 4)
                        # print(value)
                        values.append(value)

    call_oldest_response(stock_id, int(year) - 1911 - 1)
    time.sleep(1.5)
    call_oldest_response(stock_id, int(year) - 1911 - 2)

    values[: 25], values[50: ] = values[50: ], values[: 25]
    dict_format = {
    "company": {
        "name": company_name,
        "stock_id": stock_id,
        "environmental": {
            "years": ["2021", "2022", "2023"],
            "categories": {
                "溫室氣體排放量": [
                    "直接(範疇一)溫室氣體排放量(公噸CO₂e)",
                    "能源間接(範疇二)溫室氣體排放量(公噸CO₂e)",
                    "其他間接(範疇三)溫室氣體排放量(公噸CO₂e)",
                    "溫室氣體排放密集度(公噸CO₂e/百萬元營業額)"
                ], 
                "再生能源使用率": [
                    "再生能源使用率"
                ], 
                "用水量": [
                    "用水量(公噸(t))",
                    "用水密集度"
                ],   
                "廢棄物重量": [
                    "有害廢棄物(公噸(t))",
                    "非有害廢棄物(公噸(t))",
                    "總重量(有害+非有害)(公噸(t))",
                    "廢棄物密集度"
                ]             
            }, 
            "data": {
                "直接(範疇一)溫室氣體排放量(公噸CO₂e)": [],
                "能源間接(範疇二)溫室氣體排放量(公噸CO₂e)": [],
                "其他間接(範疇三)溫室氣體排放量(公噸CO₂e)": [],
                "溫室氣體排放密集度(公噸CO₂e/百萬元營業額)": [],
                "再生能源使用率": [],
                "用水量(公噸(t))": [],
                "用水密集度": [],
                "有害廢棄物(公噸(t))": [],
                "非有害廢棄物(公噸(t))": [],
                "總重量(有害+非有害)(公噸(t))": [],
                "廢棄物密集度": []
            }
        }, 
        "social": {
            "years": ["2021", "2022", "2023"], 
            "categories": {
                  "員工福利平均數": [
                      "員工福利平均數(每年6/2起公開)(仟元/人)"
                  ], 
                  "員工薪資平均數": [
                      "員工薪資平均數(每年6/2起公開)(仟元/人)"
                  ],
                  "非擔任主管職務之全時員工薪資": [
                      "非擔任主管職務之全時員工薪資平均數(每年7/1起公開)(仟元/人)",
                      "非擔任主管職務之全時員工薪資中位數(每年7/1起公開)(仟元/人)"
                  ],
                  "管理職女性主管占比": [
                      "管理職女性主管占比"
                  ],
                  "職業災害": [
                      "職業災害人數(人)",
                      "職業災害人數比率"
                  ]
            },
            "data": {
              "員工福利平均數(每年6/2起公開)(仟元/人)": [],
              "員工薪資平均數(每年6/2起公開)(仟元/人)": [],
              "非擔任主管職務之全時員工薪資平均數(每年7/1起公開)(仟元/人)": [],
              "非擔任主管職務之全時員工薪資中位數(每年7/1起公開)(仟元/人)": [],
              "管理職女性主管占比": [],
              "職業災害人數(人)": [],
              "職業災害人數比率": []
            }
        },
        "governance": {
            "years": ["2021", "2022", "2023"],
            "categories": {
                "董事會結構與會議": [
                    "董事會席次(席)",
                    "獨立董事席次(席)",
                    "女性董事席次(席)",
                    "公司年度召開法說會次數(次)"
                ],
                "董事參與與能力": [
                    "女性董事比率",
                    "董事出席董事會出席率",
                    "董事進修時數符合進修要點比率"
                ]
            },
            "data": {
                "董事會席次(席)": [],
                "獨立董事席次(席)": [],
                "女性董事席次(席)": [],
                "公司年度召開法說會次數(次)": [],
                "女性董事比率": [],
                "董事出席董事會出席率": [],
                "董事進修時數符合進修要點比率": []
                }
            }
        }
    }

    for i in range(len(values)): # 0 ~ 74
        value = values[i]
        j = i % 25 # 0 ~ 24
        name = names[j]
        # print(name)
        if j <= 10:
            dict_format["company"]["environmental"]["data"][name].append(value)
        elif j <= 17:
            dict_format["company"]["social"]["data"][name].append(value)
        else:
            dict_format["company"]["governance"]["data"][name].append(value)

    return json.dumps(dict_format, ensure_ascii = False, indent = 2)

@app.route("/get_three_years_ESG_data_analysis", methods = ["get"])
def three_years_ESG_data_analysis():
    # stock_id = input("請輸入股票代碼: ")
    stock_id = request.args.get("stock_id")
    year = "2023"
    payload = {
    "companyCode": stock_id, # 要是字串
    "yearList": [year],
    "year": year
    }

    newest_response = requests.post("https://esggenplus.twse.com.tw/api/api/mopsEsg/singleCompanyData", json = payload)

    newest_fields = ["直接(範疇一)溫室氣體排放量(公噸CO₂e)", "能源間接(範疇二)溫室氣體排放量(公噸CO₂e)", "其他間接(範疇三)溫室氣體排放量(公噸CO₂e)",
                    "溫室氣體排放密集度(公噸CO₂e/百萬元營業額)", "再生能源使用率", "用水量(公噸(t))", "用水密集度", "有害廢棄物(公噸(t))",
                    "非有害廢棄物(公噸(t))", "總重量(有害+非有害)(公噸(t))", "廢棄物密集度", "員工福利平均數(每年6/2起公開)(仟元/人)",
                    "員工薪資平均數(每年6/2起公開)(仟元/人)", "非擔任主管職務之全時員工薪資平均數(每年7/1起公開)(仟元/人)", 
                    "非擔任主管職務之全時員工薪資中位數(每年7/1起公開)(仟元/人)", "管理職女性主管占比", "職業災害人數(人)", 
                    "職業災害人數比率", "董事會席次(席)", "獨立董事席次(席)", "女性董事席次(席)", "女性董事比率", "董事出席董事會出席率", 
                    "董事進修時數符合進修要點比率", "公司年度召開法說會次數(次)"] # 抓sections裡的name

    oldest_fields = ["直接溫室氣體排放量", "能源間接", "其他間接", "溫室氣體排放密集度", "再生能源使用率", "用水量", "用水密集度", "有害廢棄物", 
                    "非有害廢棄物", "總重量", "廢棄物密集度", "員工福利平均數", "員工薪資平均數", "非擔任主管職務之全時員工薪資平均數",
                    "非擔任主管職務之全時員工薪資中位數", "管理職女性主管占比", "職業災害人數", "職業災害人數比率", "董事會席次", "獨立董事席次",
                    "女性董事席次及比率", "董事出席董事會出席率", "董監事進修時數符合進修要點比率", "公司年度召開法說會次數"]

    data = newest_response.json()["data"]
    company_name = data[0]["companyName"]
    treeModels = data[0]["treeModels"]

    names = []
    values = []
    temp_list = []
    for i in range(len(treeModels)):
        items = treeModels[i]["items"]
        for j in range(len(items)):
            sections = items[j]["sections"]
            # print(sections)
            for k in range(len(sections)):
                name = sections[k]["name"]
                # print(name)
                if name in newest_fields: # 女性董事席次(席) 女性董事比率 女性董事席次及比率 確定後面有沒有空格
                    # print(name)
                    if name == "女性董事比率":
                        names.extend([name, "董事出席董事會出席率", "董事進修時數符合進修要點比率"])
                        value = sections[k]["controls"][0]["value"]
                        value = value[: -1]
                        value = round(float(value) / 100, 4)
                        temp_list.insert(0, value)
                        values.extend(temp_list)
                    elif name == "董事出席董事會出席率" or name == "董事進修時數符合進修要點比率":
                        value = sections[k]["controls"][0]["value"]
                        # print(value)
                        value = value[: -1]
                        value = round(float(value) / 100, 4)
                        temp_list.append(value)
                    else:
                        value = sections[k]["controls"][0]["value"]
                        value = value.replace(",", "")
                        if value[-1] == "%":
                            value = value[: -1]
                            value = round(float(value) / 100, 4)
                        value = round(float(value), 4)
                        values.append(value)
                        names.append(name)


    def call_oldest_response(stock_id, year):
        data = {
            "encodeURIComponent": 1,
            "step": 2,
            "co_id": stock_id, # 可以是數字或文字
            "YEAR": year # 可以是數字或文字
        }
        oldest_response = requests.post("https://mops.twse.com.tw/mops/web/t214sb01", data = data)

        fields = []
        # values = []
        soup = bs(oldest_response.text, features = "html.parser")
        # print(soup.prettify())
        tables = soup.find_all("table")
        target_table = tables[-2]
        trs = target_table.find_all("tr")
        for tr in trs:
            td = tr.find("td", attrs = {"align": "center"})
            if td is None:
                continue
            text = td.text
            end = text.find("(")
            if end == -1:
                text = text.strip()
            else:    
                text = text[: end].strip()
            # print(text)
            # print("-" * 10)
            if text in oldest_fields:
                field = tr.find("td", attrs = {"align": "center"}).text.strip()
                # print(tr)
                # print(field)
                if field == "女性董事席次及比率": # 女性董事席次(席) 女性董事比率 女性董事席次及比率
                    fields.extend(["女性董事席次(席)", "女性董事比率"])
                    temp_list = tr.find_all("td", attrs = {"align": "right"})
                    # print(temp_list)
                    number = round(float(temp_list[0].text.strip()[: -1]), 4)
                    ratio = round(float(temp_list[1].text.strip()[: -1]) / 100, 4)
                    # print(number, ratio)
                    values.extend([number, ratio])
                    # print(f"{temp_list[0].text.strip()}({temp_list[1].text.strip()})")
                else: 
                    fields.append(field)
                    value = tr.find("td", attrs = {"align": "right"})
                    # print(value)
                    if value is None:
                        values.append(0)
                    elif value.text == "":
                        values.append(0)  
                    else:
                        value = value.text.strip().replace(",", "")
                        if value[-1] == "%":
                            value = value[: -1]
                            value = round(float(value) / 100, 4)
                            # print(value)
                        elif value[-1] == "人":
                            value = round(float(value[: -1]), 4)
                            # print(value)
                        else:
                            value = round(float(value), 4)
                        # print(value)
                        values.append(value)

    call_oldest_response(stock_id, int(year) - 1911 - 1)
    time.sleep(1.5)
    call_oldest_response(stock_id, int(year) - 1911 - 2)
    # print(values)
    values[: 25], values[50: ] = values[50: ], values[: 25]
    # print(values)

    df = pd.DataFrame()
    df.index = names
    df["2021年"] = values[: 25]
    df["2022年"] = values[25: 50]
    df["2023年"] = values[50: ]

    load_dotenv()
    chat_model = ChatOpenAI(model = "gpt-3.5-turbo", api_key = os.getenv("OPENAI_API_KEY"))
    prompt = ChatPromptTemplate.from_messages([
        ("system", "請您扮演一位台灣專業的 ESG 分析師"),
        ("human", "我現在有2021年至2023年的 ESG 資料，這間公司是{company_name}，欄位資訊為{names}。"),
        ("human", "請您根據以下三年的資訊，使用繁體中文各自分析 ESG 三個面向給我，並以嚴謹的角度撰寫近期的趨勢報告。\n"
         "以下為此三年的資訊：{ESG_data}")
    ])
    str_parser = StrOutputParser()

    ESG_analysis_chain = prompt | chat_model | str_parser
    ESG_response = ESG_analysis_chain.invoke({"company_name": company_name, "names": names, "ESG_data": df[["2021年", "2022年", "2023年"]]})
    
    return ESG_response

@app.route("/get_one_year_ESG_data_format", methods = ["get"])
def one_year_ESG_data_format():
    def call_response(stock_id, year, names, values, company_names): # 2023年(民國112年)各公司資料的 names 是一樣的
        if year == "2023":
            payload = {
            "companyCode": stock_id, # 要是字串
            "yearList": [year],
            "year": year
            }

            newest_response = requests.post("https://esggenplus.twse.com.tw/api/api/mopsEsg/singleCompanyData", json = payload)

            newest_fields = ["直接(範疇一)溫室氣體排放量(公噸CO₂e)", "能源間接(範疇二)溫室氣體排放量(公噸CO₂e)", "其他間接(範疇三)溫室氣體排放量(公噸CO₂e)",
                            "溫室氣體排放密集度(公噸CO₂e/百萬元營業額)", "再生能源使用率", "用水量(公噸(t))", "用水密集度", "有害廢棄物(公噸(t))",
                            "非有害廢棄物(公噸(t))", "總重量(有害+非有害)(公噸(t))", "廢棄物密集度", "員工福利平均數(每年6/2起公開)(仟元/人)",
                            "員工薪資平均數(每年6/2起公開)(仟元/人)", "非擔任主管職務之全時員工薪資平均數(每年7/1起公開)(仟元/人)", 
                            "非擔任主管職務之全時員工薪資中位數(每年7/1起公開)(仟元/人)", "管理職女性主管占比", "職業災害人數(人)", 
                            "職業災害人數比率", "董事會席次(席)", "獨立董事席次(席)", "女性董事席次(席)", "女性董事比率", "董事出席董事會出席率", 
                            "董事進修時數符合進修要點比率", "公司年度召開法說會次數(次)"] # 抓sections裡的name

            data = newest_response.json()["data"]
            company_name = data[0]["companyName"]
            treeModels = data[0]["treeModels"]
            company_names.append(company_name)

            temp_list = []
            for i in range(len(treeModels)):
                items = treeModels[i]["items"]
                for j in range(len(items)):
                    sections = items[j]["sections"]
                    # print(sections)
                    for k in range(len(sections)):
                        name = sections[k]["name"]
                        # print(name)
                        if name in newest_fields: # 女性董事席次(席) 女性董事比率 女性董事席次及比率 確定後面有沒有空格
                            # print(name)
                            if name == "女性董事比率":
                                names.extend([name, "董事出席董事會出席率", "董事進修時數符合進修要點比率"])
                                value = sections[k]["controls"][0]["value"]
                                value = value[: -1]
                                value = round(float(value) / 100, 4)
                                temp_list.insert(0, value)
                                values.extend(temp_list)
                            elif name == "董事出席董事會出席率" or name == "董事進修時數符合進修要點比率":
                                value = sections[k]["controls"][0]["value"]
                                # print(value)
                                value = value[: -1]
                                value = round(float(value) / 100, 4)
                                temp_list.append(value)
                            else:
                                value = sections[k]["controls"][0]["value"]
                                value = value.replace(",", "")
                                if value[-1] == "%":
                                    value = value[: -1]
                                    value = round(float(value) / 100, 4)
                                value = round(float(value), 4)
                                values.append(value)
                                names.append(name)
            return names, values, company_names
        else:
            oldest_fields = ["直接溫室氣體排放量", "能源間接", "其他間接", "溫室氣體排放密集度", "再生能源使用率", "用水量", "用水密集度", "有害廢棄物", 
                        "非有害廢棄物", "總重量", "廢棄物密集度", "員工福利平均數", "員工薪資平均數", "非擔任主管職務之全時員工薪資平均數",
                        "非擔任主管職務之全時員工薪資中位數", "管理職女性主管占比", "職業災害人數", "職業災害人數比率", "董事會席次", "獨立董事席次",
                        "女性董事席次及比率", "董事出席董事會出席率", "董監事進修時數符合進修要點比率", "公司年度召開法說會次數"]
            data = {
                "encodeURIComponent": 1,
                "step": 2,
                "co_id": stock_id, # 可以是數字或文字
                "YEAR": str(int(year) - 1911) # 可以是數字或文字
            }
            oldest_response = requests.post("https://mops.twse.com.tw/mops/web/t214sb01", data = data)

            soup = bs(oldest_response.text, features = "html.parser")
            # print(soup.prettify())
            start = soup.prettify().find("本資料由") + 4
            end = soup.prettify().find("公司提供")
            company_name = soup.prettify()[start: end]
            company_names.append(company_name)
            tables = soup.find_all("table")
            target_table = tables[-2]
            trs = target_table.find_all("tr")
            for tr in trs:
                td = tr.find("td", attrs = {"align": "center"})
                if td is None:
                    continue
                text = td.text
                end = text.find("(")
                if end == -1:
                    text = text.strip()
                else:    
                    text = text[: end].strip()
                # print(text)
                # print("-" * 10)
                if text in oldest_fields:
                    field = tr.find("td", attrs = {"align": "center"}).text.strip()
                    # print(tr)
                    # print(field)
                    if field == "女性董事席次及比率": # 女性董事席次(席) 女性董事比率 女性董事席次及比率
                        names.extend(["女性董事席次(席)", "女性董事比率"])
                        temp_list = tr.find_all("td", attrs = {"align": "right"})
                        # print(temp_list)
                        number = round(float(temp_list[0].text.strip()[: -1]), 4)
                        ratio = round(float(temp_list[1].text.strip()[: -1]) / 100, 4)
                        # print(number, ratio)
                        values.extend([number, ratio])
                        # print(f"{temp_list[0].text.strip()}({temp_list[1].text.strip()})")
                    else: 
                        names.append(field)
                        value = tr.find("td", attrs = {"align": "right"})
                        # print(value)
                        if value is None:
                            values.append(0)
                        elif value.text == "":
                            values.append(0)  
                        else:
                            value = value.text.strip().replace(",", "")
                            if value[-1] == "%":
                                value = value[: -1]
                                value = round(float(value) / 100, 4)
                                # print(value)
                            elif value[-1] == "人":
                                value = round(float(value[: -1]), 4)
                                # print(value)
                            else:
                                value = round(float(value), 4)
                            # print(value)
                            values.append(value)
            return names, values, company_names

    # stock_ids = input("請輸入股票代碼，並以空格區分: ").split()
    # year = input("請輸入西元年份: ")
    stock_ids = request.args.get("stock_ids")
    year = request.args.get("year")
    names = []
    values = []
    company_names = []

    if "," in stock_ids:
        stock_ids = stock_ids.split(",")
    else:
        stock_ids = [stock_ids]

    for stock_id in stock_ids:
        time.sleep(1.5)
        names, values, company_names = call_response(stock_id, year, names, values, company_names)

    dict_format = {"companies": [{} for _ in range(len(stock_ids))]}
    for i in range(len(stock_ids)):
        dict_format["companies"][i]["name"] = company_names[i]
        dict_format["companies"][i]["year"] = int(year)
        dict_format["companies"][i]["stock_id"] = stock_ids[i]
        dict_format["companies"][i]["environmental"] = [{} for _ in range(11)]
        dict_format["companies"][i]["social"] = [{} for _ in range(7)]
        dict_format["companies"][i]["governance"] = [{} for _ in range(7)]

    for i in range(len(names)):
        name = names[i]
        value = values[i]
        index = i % 25
        # print(index)
        if index < 11 and i < 25:
            dict_format["companies"][0]["environmental"][index] = {"index": name, "value": value}
        elif index < 18 and i < 25:
            dict_format["companies"][0]["social"][index % 11] = {"index": name, "value": value}
        elif index < 25 and i < 25:
            dict_format["companies"][0]["governance"][index % 18] = {"index": name, "value": value}
        elif index < 11 and i < 50:
            dict_format["companies"][1]["environmental"][index] = {"index": name, "value": value}
        elif index < 18 and i < 50:
            dict_format["companies"][1]["social"][index % 11] = {"index": name, "value": value}
        elif index < 25 and i < 50:
            dict_format["companies"][1]["governance"][index % 18] = {"index": name, "value": value}
        elif index < 11 and i < 75:
            dict_format["companies"][2]["environmental"][index] = {"index": name, "value": value}
        elif index < 18 and i < 75:
            dict_format["companies"][2]["social"][index % 11] = {"index": name, "value": value}
        else:
            dict_format["companies"][2]["governance"][index % 18] = {"index": name, "value": value}

    return json.dumps(dict_format, ensure_ascii = False, indent = 2)

@app.route("/get_one_year_ESG_data_analysis", methods = ["get"])
def one_year_ESG_data_analysis():
    def call_response(stock_id, year, names, values, company_names): # 2023年(民國112年)各公司資料的 names 是一樣的
        if year == "2023":
            payload = {
            "companyCode": stock_id, # 要是字串
            "yearList": [year],
            "year": year
            }

            newest_response = requests.post("https://esggenplus.twse.com.tw/api/api/mopsEsg/singleCompanyData", json = payload)

            newest_fields = ["直接(範疇一)溫室氣體排放量(公噸CO₂e)", "能源間接(範疇二)溫室氣體排放量(公噸CO₂e)", "其他間接(範疇三)溫室氣體排放量(公噸CO₂e)",
                            "溫室氣體排放密集度(公噸CO₂e/百萬元營業額)", "再生能源使用率", "用水量(公噸(t))", "用水密集度", "有害廢棄物(公噸(t))",
                            "非有害廢棄物(公噸(t))", "總重量(有害+非有害)(公噸(t))", "廢棄物密集度", "員工福利平均數(每年6/2起公開)(仟元/人)",
                            "員工薪資平均數(每年6/2起公開)(仟元/人)", "非擔任主管職務之全時員工薪資平均數(每年7/1起公開)(仟元/人)", 
                            "非擔任主管職務之全時員工薪資中位數(每年7/1起公開)(仟元/人)", "管理職女性主管占比", "職業災害人數(人)", 
                            "職業災害人數比率", "董事會席次(席)", "獨立董事席次(席)", "女性董事席次(席)", "女性董事比率", "董事出席董事會出席率", 
                            "董事進修時數符合進修要點比率", "公司年度召開法說會次數(次)"] # 抓sections裡的name

            data = newest_response.json()["data"]
            company_name = data[0]["companyName"]
            treeModels = data[0]["treeModels"]
            company_names.append(company_name)

            temp_list = []
            for i in range(len(treeModels)):
                items = treeModels[i]["items"]
                for j in range(len(items)):
                    sections = items[j]["sections"]
                    # print(sections)
                    for k in range(len(sections)):
                        name = sections[k]["name"]
                        # print(name)
                        if name in newest_fields: # 女性董事席次(席) 女性董事比率 女性董事席次及比率 確定後面有沒有空格
                            # print(name)
                            if name == "女性董事比率":
                                names.extend([name, "董事出席董事會出席率", "董事進修時數符合進修要點比率"])
                                value = sections[k]["controls"][0]["value"]
                                value = value[: -1]
                                value = round(float(value) / 100, 4)
                                temp_list.insert(0, value)
                                values.extend(temp_list)
                            elif name == "董事出席董事會出席率" or name == "董事進修時數符合進修要點比率":
                                value = sections[k]["controls"][0]["value"]
                                # print(value)
                                value = value[: -1]
                                value = round(float(value) / 100, 4)
                                temp_list.append(value)
                            else:
                                value = sections[k]["controls"][0]["value"]
                                value = value.replace(",", "")
                                if value[-1] == "%":
                                    value = value[: -1]
                                    value = round(float(value) / 100, 4)
                                value = round(float(value), 4)
                                values.append(value)
                                names.append(name)
            return names, values, company_names
        else:
            oldest_fields = ["直接溫室氣體排放量", "能源間接", "其他間接", "溫室氣體排放密集度", "再生能源使用率", "用水量", "用水密集度", "有害廢棄物", 
                        "非有害廢棄物", "總重量", "廢棄物密集度", "員工福利平均數", "員工薪資平均數", "非擔任主管職務之全時員工薪資平均數",
                        "非擔任主管職務之全時員工薪資中位數", "管理職女性主管占比", "職業災害人數", "職業災害人數比率", "董事會席次", "獨立董事席次",
                        "女性董事席次及比率", "董事出席董事會出席率", "董監事進修時數符合進修要點比率", "公司年度召開法說會次數"]
            data = {
                "encodeURIComponent": 1,
                "step": 2,
                "co_id": stock_id, # 可以是數字或文字
                "YEAR": str(int(year) - 1911) # 可以是數字或文字
            }
            oldest_response = requests.post("https://mops.twse.com.tw/mops/web/t214sb01", data = data)

            soup = bs(oldest_response.text, features = "html.parser")
            # print(soup.prettify())
            start = soup.prettify().find("本資料由") + 4
            end = soup.prettify().find("公司提供")
            company_name = soup.prettify()[start: end]
            company_names.append(company_name)
            tables = soup.find_all("table")
            target_table = tables[-2]
            trs = target_table.find_all("tr")
            for tr in trs:
                td = tr.find("td", attrs = {"align": "center"})
                if td is None:
                    continue
                text = td.text
                end = text.find("(")
                if end == -1:
                    text = text.strip()
                else:    
                    text = text[: end].strip()
                # print(text)
                # print("-" * 10)
                if text in oldest_fields:
                    field = tr.find("td", attrs = {"align": "center"}).text.strip()
                    # print(tr)
                    # print(field)
                    if field == "女性董事席次及比率": # 女性董事席次(席) 女性董事比率 女性董事席次及比率
                        names.extend(["女性董事席次(席)", "女性董事比率"])
                        temp_list = tr.find_all("td", attrs = {"align": "right"})
                        # print(temp_list)
                        number = round(float(temp_list[0].text.strip()[: -1]), 4)
                        ratio = round(float(temp_list[1].text.strip()[: -1]) / 100, 4)
                        # print(number, ratio)
                        values.extend([number, ratio])
                        # print(f"{temp_list[0].text.strip()}({temp_list[1].text.strip()})")
                    else: 
                        names.append(field)
                        value = tr.find("td", attrs = {"align": "right"})
                        # print(value)
                        if value is None:
                            values.append(0)
                        elif value.text == "":
                            values.append(0)  
                        else:
                            value = value.text.strip().replace(",", "")
                            if value[-1] == "%":
                                value = value[: -1]
                                value = round(float(value) / 100, 4)
                                # print(value)
                            elif value[-1] == "人":
                                value = round(float(value[: -1]), 4)
                                # print(value)
                            else:
                                value = round(float(value), 4)
                            # print(value)
                            values.append(value)
            return names, values, company_names

    stock_ids = request.args.get("stock_ids")
    year = request.args.get("year")
    # print(stock_ids)
    
    if "," in stock_ids:
        stock_ids = stock_ids.split(",")
    else:
        stock_ids = [stock_ids]
    # print(stock_ids)
    # return "over"
    # stock_ids = input("請輸入股票代碼，並以空格區分: ").split() # 3413 5347 6770
    # year = input("請輸入西元年份: ")
    names = []
    values = []
    company_names = []

    for stock_id in stock_ids:
        time.sleep(1.5)
        names, values, company_names = call_response(stock_id, year, names, values, company_names)

    # return names, values, company_names

    df = pd.DataFrame()
    for i in range(len(company_names)):
        df[company_names[i]] = values[i * 25: (i + 1) * 25]
    df.index = names[: 25]

    load_dotenv()
    chat_model = ChatOpenAI(model = "gpt-3.5-turbo", api_key = os.getenv("OPENAI_API_KEY"))
    prompt = ChatPromptTemplate.from_messages([
        ("system", "請您扮演一位台灣專業的 ESG 分析師"),
        ("human", "我現在有{year}年的 ESG 資訊，有{company_amount}間公司，包含{company_names}，欄位資訊為{column_names}。"),
        ("human", "請您根據以下公司的 ESG 資訊，使用繁體中文比較並分析以下公司 ESG 的三個面向給我，請以嚴謹的角度撰寫近期的趨勢報告。\n"
         "以下為公司的資訊：{ESG_data}")
    ])
    str_parser = StrOutputParser()

    ESG_analysis_chain = prompt | chat_model | str_parser
    ESG_response = ESG_analysis_chain.invoke({"year": year, "company_amount": len(company_names), "company_names": company_names,
                                              "column_names": df.index, "ESG_data": df[company_names]})
    return ESG_response

@app.route("/get_annual_report_summary", methods = ["get"])
def annual_report_summary():
    def annual_report(stock_id, year):
        url = "https://doc.twse.com.tw/server-java/t57sb01"

      # 建立 POST 請求的表單
        data = {
          "id": "",
          "key": "",
          "step": "1",
          "co_id": stock_id, # 可以是數字
          "year": int(year) - 1911 + 1, # 要輸入民國年份 # 可以是數字
          "seamon": "",
          "mtype": "F",
          "dtype": "F04"
    }
        # # 這兩行為測試
        # response = requests.post(url, data = data)
        # return response

        try:
          # 發送 POST 請求
          response = requests.post(url, data = data)

          # 取得回應後擷取檔案名稱
          soup = bs(response.text, "html.parser")
          link1 = soup.find("a").text
          # print(link1)
        except Exception as e:
          print(f"發生{e}錯誤")

        # 建立第二個 POST 請求的表單
        data2 = {
            "step": "9",
            "kind": "F",
            "co_id": stock_id,
            "filename": link1 # 檔案名稱
        }

        # 這兩行為測試
        # response = requests.post(url, data = data2)
        # return response

        try:
          # 發送 POST 請求
          response = requests.post(url, data = data2)
          soup = bs(response.text, "html.parser")
          link2 = soup.find("a")
          # 取得 PDF 連結
          link2 = link2.get("href")
        #   print(link2)
        except Exception as e:
          print(f"發生{e}錯誤")

        # # 這兩行為測試
        # response = requests.get("https://doc.twse.com.tw" + link2)
        # return response

        return "https://doc.twse.com.tw" + link2

    # stock_id = input("請輸入想要查詢的股票代碼: ")
    # year = input("請輸入西元年度: ")
    stock_id = request.args.get("stock_id")
    year = request.args.get("year")

    # 取得目前程式碼所在的目錄
    current_directory = os.getcwd()

    # 定義你要檢查的資料夾名稱
    folder_name = "{}_{}_small_chunks_db".format(year, stock_id)

    # 檢查資料夾是否存在
    folder_path = os.path.join(current_directory, folder_name)

    embeddings_model = OpenAIEmbeddings(model = "text-embedding-3-small")

    if os.path.isdir(folder_path):
        # print(f"資料夾 '{folder_name}' 存在於當前目錄。")
        pass
    else:
        # print(f"資料夾 '{folder_name}' 不存在於當前目錄。")
        url = annual_report(stock_id, year)
        loader = PyPDFLoader(file_path = url)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(separators = ["\n \n", "\n"], 
                                                      chunk_size = 500, chunk_overlap = 100) # 原本用 1000 和 100
        splits = text_splitter.split_documents(docs)
        FAISS_db = FAISS.from_documents(splits, embeddings_model)
        FAISS_db.save_local(f"{year}_{stock_id}_small_chunks_db") # 這裡有改

    Load_FAISS_db = FAISS.load_local(
        folder_path = f"{year}_{stock_id}_small_chunks_db", # 這裡有改
        embeddings = embeddings_model,
        allow_dangerous_deserialization = True
    )

    chat_model = ChatOpenAI(model = "gpt-3.5-turbo", api_key = os.getenv("OPENAI_API_KEY")) # 要錄影再改比較好的模型 這裡有改

    key_words = ["公司的業務範疇、主要產品及服務、市場概述",
                "資本支出重點",
                "主要財務風險",
                "市場競爭策略",
                "技術創新、數位轉型的重要成果",
                "在環境、社會和治理(ESG)方面的表現",
                "供應鏈管理上的挑戰",
                "未來發展計劃",
                "致股東報告書的摘要",
                "會計師查核意見"]

    data_list = []
    for key_word in key_words:
        data = Load_FAISS_db.max_marginal_relevance_search(key_word, search_type = "mmr", k = 2) # 這裡有改成2，因為 chunk 變小 
        data_list += data

    language_prompt = "請使用繁體中文和台灣用詞輸出報告"

    report_template = [("system", "你的任務是生成年報摘要，"
                    "請務必保留重點如營收漲跌、開發項目等，生成的內容不應該太攏統，也不是解釋問題，"
                    "{language}。\n\n"
                    "輸出的格式請針對各個問題：\n"
                    "{questions}，列點生成相對應的年報內容，例如：1. 關鍵字: 回覆，關鍵字請利用重點表示而不是疑問句。\n\n"
                    "請你要用分段的形式好讓使用者方便閱讀，並且不需要在開頭和結尾生成不必要的文字，只要關鍵字和內容就好。\n"
                    "此外，也請不要將生成內容用括號包起來，直接輸出條列式的文即可。\n\n"
                    "以下為年報內容：\n{content}")]

    prompt = ChatPromptTemplate.from_messages(messages = report_template).partial(language = language_prompt, 
                                                                                  questions = key_words)

    str_output_parser = StrOutputParser()

    # stuff 預設是限定變數名為 text，但可以改
    summarize_chain = load_summarize_chain(llm = chat_model, prompt = prompt, chain_type = "stuff", document_variable_name = "content")
    str_result = summarize_chain.invoke({"input_documents": data_list})
    str_result = str_output_parser.invoke(str_result["output_text"])
    # print(str_result)
    return str_result

@app.route("/get_QA_response", methods = ["get"])
def QA_response():
    # stock_id = "2330" # input("請輸入您想查詢的台灣公司股票代碼: ")
    # year = "2023" # input("請輸入您想查詢的西元年度: ")
    stock_id = request.args.get("stock_id")
    year = request.args.get("year")
    msg = request.args.get("msg")
    # msg = input("請輸入您想問的問題: ")
    # print("我說:", msg)

    embeddings_model = OpenAIEmbeddings(model = "text-embedding-3-small")
    Load_FAISS_db = FAISS.load_local(
        folder_path = f"{year}_{stock_id}_small_chunks_db", # 這裡有改
        embeddings = embeddings_model,
        allow_dangerous_deserialization = True
    )

    retriever = Load_FAISS_db.as_retriever(search_type = "similarity", search_kwargs = {"k": 5}) # k 可以設大一點

    tool = create_retriever_tool(
        retriever = retriever,
        name = "retriever_by_company_annual_report",
        description = "搜尋並返回公司年報內容"
    )
    tools = [tool]

    first_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位善用工具的好助理，請回答與公司年報內容相關的問題，這間台灣公司的股票代號是{stock_id}、此年報的年份是{year}。\n"
        "請自己判斷上下文來回答問題，請確定公司的名字，並且使用繁體中文和台灣用語。"),
        MessagesPlaceholder(variable_name = "chat_history"),
        ("human", "{input}。"),
        MessagesPlaceholder(variable_name = "agent_scratchpad")
    ])
    first_prompt = first_prompt.partial(year = year, stock_id = stock_id)
    chat_model = ChatOpenAI(model = "gpt-3.5-turbo") # 這裡模型可以改
    first_agent = create_openai_tools_agent(chat_model, tools, first_prompt)
    first_agent_executor = AgentExecutor(agent = first_agent, tools = tools) # , verbose = True 

    memory = SQLChatMessageHistory(
        session_id = "test_id",
        connection = "sqlite:///history_db"
    )

    def window_messages(chain_input):
        if len(memory.messages) > 6:
            cur_messages = memory.messages[-6: ]
            memory.clear()
            memory.add_messages(cur_messages)
        return

    def add_history(first_agent_executor):
        agent_with_chat_history = RunnableWithMessageHistory(
            first_agent_executor,
            lambda session_id: memory,
            input_messages_key = "input",
            history_messages_key = "chat_history"
        )
        memory_chain = (
            RunnablePassthrough.assign(messages = window_messages)
            | agent_with_chat_history
        )
        return memory_chain

    first_agent = add_history(first_agent_executor)
    first_response = first_agent.invoke({"input": msg}, config = {"configurable": {"session_id": "test_id"}})["output"]

    return first_response

@app.route("/get_stock_price", methods = ["get"])
def stock_price():
    # stock_id = input("請輸入公司股票代號: ") # 2330
    stock_id = request.args.get("stock_id")
    stock_id = f"{stock_id}.TW"
    # maximum_expected_return = int(input("請輸入最大整數預期報酬率: ")) / 100 # 5
    # minumum_expected_return = int(input("請輸入最小整數預期報酬率: ")) / 100 # 3
    maximum_expected_return = float(request.args.get("maximum_expected_return")) / 100
    minumum_expected_return = float(request.args.get("minumum_expected_return")) / 100
    stock = yf.Ticker(stock_id)
    df = pd.DataFrame(stock.dividends).reset_index()
    # df["Dividends"] = round(df["Dividends"], 2)
    df["Year"] = df["Date"].dt.year # year 的 type 是 int32
    df_current_year_dividends = df[df["Year"] == 2024]
    dividends = df_current_year_dividends["Dividends"].sum()#.round(2)
    expensive_stock_price = round(dividends / minumum_expected_return)
    cheap_stock_price = round(dividends / maximum_expected_return)
    # print("昂貴價:", expensive_stock_price)
    # print("便宜價:", cheap_stock_price)

    dot_index = stock_id.find(".")

    response = requests.get(f"https://ws.api.cnyes.com/ws/api/v1/charting/history?resolution=1&symbol=TWS:{stock_id[: dot_index]}:STOCK&quote=1")

    stock_price = str(response.json()["data"]["quote"]["6"])
    stock_price_increase_or_decrease = str(response.json()["data"]["quote"]["220027"])
    stock_price_increase_or_decrease_percentage = str(response.json()["data"]["quote"]["56"])
    # print("股價:", stock_price)
    # print("股價漲跌:", stock_price_increase_or_decrease)
    # print("股價漲跌幅:", f"{stock_price_increase_or_decrease_percentage}%")

    def get_warning(stock_id, ratio):
        data = {
            "compareItem": ratio,
            "quarter": "true",
            "ylabel": "%",
            "ys": "0",
            "revenue": "true",
            "bcodeAvg": "true",
            "companyAvg": "false",
            "companyId": stock_id
        }

        response = requests.post("https://mopsfin.twse.com.tw/compare/data", data = data)

        data = response.json()
        data = json.loads(data["json"])

        quarters = data["xaxisList"]

        values = data["graphData"]
        full_company_names = data["checkedNameList"]

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
        industry_list = full_company_names.copy()
        for i in range(1):
            # print(industry_list[i])
            left_parenthesis_index = industry_list[i].find("(")
            right_parenthesis_index = industry_list[i].find(")")
            industry = industry_list[i][left_parenthesis_index + 3: right_parenthesis_index]
            # industry_list[i] = industry_list[i][left_parenthesis_index + 3: right_parenthesis_index]
            # print(industry)

            company_name = company_name_list[i + 1]
            # print(company_name)

            # print(df[company_name].iloc[-1])
            # print(df[industry].iloc[-1])
            if ratio == "DebtRatio":
                if df[company_name].iloc[-1] > 50:
                    if df[company_name].iloc[-1] > df[industry].iloc[-1]:
                        ratios_values.append("Y")
                    else:
                        ratios_values.append("N")
                else:
                    ratios_values.append("N")
            elif ratio == "LongTermLiabilitiesRatio"or ratio == "CurrentRatio" or ratio == "QuickRatio":
                if df[company_name].iloc[-1] < 100:
                    if df[company_name].iloc[-1] < df[industry].iloc[-1]:
                        ratios_values.append("Y")
                    else:
                        ratios_values.append("N")
                else:
                    ratios_values.append("N")
            elif ratio == "InterestCoverage":
                if df[company_name].iloc[-1] < 5:
                    if df[company_name].iloc[-1] < df[industry].iloc[-1]:
                        ratios_values.append("Y")
                    else:
                        ratios_values.append("N")
                else:
                    ratios_values.append("N")
            elif ratio == "TotalAssetTurnover":
                if df[company_name].iloc[-1] < 0.5:
                    if df[company_name].iloc[-1] < df[industry].iloc[-1]:
                        ratios_values.append("Y")
                    else:
                        ratios_values.append("N")
                else:
                    ratios_values.append("N")
            elif ratio == "NetIncomeMargin" or ratio == "GrossMargin":
                if df[company_name].iloc[-1] < df[company_name].iloc[-5]:
                    if ((df[company_name].iloc[-1] - df[company_name].iloc[-5]) / df[company_name].iloc[-5]) < ((df[industry].iloc[-1] - df[industry].iloc[-5]) / df[industry].iloc[-5]):
                        ratios_values.append("Y")
                    else:
                        ratios_values.append("N")
                else:
                    ratios_values.append("N")
            elif ratio == "ROA":
                if df[company_name].iloc[-1] < 0.06:
                    if df[company_name].iloc[-1] < df[industry].iloc[-1]:
                        ratios_values.append("Y")
                    else:
                        ratios_values.append("N")
                else:
                    ratios_values.append("N")
            elif ratio == "ROE":
                if df[company_name].iloc[-1] < 0.08:
                    if df[company_name].iloc[-1] < df[industry].iloc[-1]:
                        ratios_values.append("Y")
                    else:
                        ratios_values.append("N")
                else:
                    ratios_values.append("N") 
            elif ratio == "OperatingCashflowToCurrentLiability":
                if df[company_name].iloc[-1] < 1:
                    if df[company_name].iloc[-1] < df[industry].iloc[-1]:
                        ratios_values.append("Y")
                    else:
                        ratios_values.append("N")
                else:
                    ratios_values.append("N")      
            elif ratio == "OperatingCashflowToNetProfit":
                if df[company_name].iloc[-1] < 0.8:
                    if df[company_name].iloc[-1] < df[industry].iloc[-1]:
                        ratios_values.append("Y")
                    else:
                        ratios_values.append("N")
                else:
                    ratios_values.append("N")

    ratios = ["DebtRatio", "LongTermLiabilitiesRatio", "CurrentRatio", "QuickRatio", "InterestCoverage", "TotalAssetTurnover",
            "GrossMargin", "NetIncomeMargin", "ROA", "ROE", "OperatingCashflowToCurrentLiability", "OperatingCashflowToNetProfit"]

    ratios_values = []

    for i in range(len(ratios)):
        # print("ratio:", ratios[i])
        get_warning(stock_id[: dot_index], ratios[i])
        # time.sleep(1)

    # Period must be one of ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
    stock_data = yf.download(stock_id, period = "max").reset_index()

    df_dividends_date = df.drop("Dividends", axis = 1)
    df_dividends_date["Date_without_time"] = pd.to_datetime(df_dividends_date["Date"].dt.date) # 原本是文字

    the_date_before_dividends = []

    for i in range(len(stock_data["Date"])):
        # print(stock_data["Date"].iloc[i]) 
        if stock_data["Date"].iloc[i] in list(df_dividends_date["Date_without_time"]):
            # print(stock_data["Date"].iloc[i - 1])
            the_date_before_dividends.append(stock_data["Date"].iloc[i - 1])

    df_dividends_date["the_date_before_dividends"] = the_date_before_dividends
    df_dividends_date = df_dividends_date.drop(["Date"], axis = 1) # "Date_without_time", 
    df_dividends_date["Year"] = df_dividends_date["Date_without_time"].dt.year
    df_dividends_date["Quarter"] = df_dividends_date["Date_without_time"].dt.quarter

    # print(df_dividends_date)
    stock_data = stock_data.reset_index(drop = True)
    stock_data.columns = stock_data.columns.droplevel(1)  # 僅保留第二層
    # print(stock_data)
    
    df_before_dividends_date_and_stock_price = pd.merge(df_dividends_date, stock_data, left_on = "the_date_before_dividends", 
                                                        right_on = "Date")[["Date_without_time", "the_date_before_dividends", 
                                                                            "Year", "Quarter", "Close"]]
    df_before_dividends_date_and_stock_price = df_before_dividends_date_and_stock_price[df_before_dividends_date_and_stock_price["Year"] 
                                                                                        >= 2015]
    df_before_dividends_date_and_stock_price = df_before_dividends_date_and_stock_price\
        .sort_values("the_date_before_dividends", ascending = False)\
        .reset_index()\
        .drop("index", axis = 1)

    beginning_year = "104"
    ending_year = "113"

    data = {
        "encodeURIComponent": 1,
        "step": 1,
        "firstin": 1,
        "off": 1,
        "queryName": "co_id",
        "inpuType": "co_id",
        "TYPEK": "all",
        "isnew": "false",
        "co_id": stock_id[: dot_index],
        "date1": beginning_year,
        "date2": ending_year,
        "qryType": "1"
    }
    # 你的程式碼中使用的是 requests.get，表示你執行的是 GET 請求，而 data 是用於 POST 的，因此需要改為 params。
    response = requests.get("https://mops.twse.com.tw/mops/web/t05st09_2", params = data) 

    df_cash_and_stock_dividends = pd.DataFrame()

    years_list = list(df_before_dividends_date_and_stock_price["Date_without_time"])[: : -1]
    cash_dividends = []
    stock_dividends = []

    soup = bs(response.text, "html.parser")
    # print(soup.prettify())
    table = soup.find("table", attrs = {"class": "hasBorder"})
    # table 
    trs = table.find_all(name = "tr", attrs = {"class": ["odd", "even"]})
    for tr in trs:
        tds = tr.find_all(name = "td", attrs = {"align": "right"})
        for i in range(len(tds)):
            if i == 0:
                cash_dividends.append(float(tds[i].text))
            elif i == 4:
                stock_dividends.append(float(tds[i].text))

    df_cash_and_stock_dividends["現金股利"] = cash_dividends[: : -1]
    df_cash_and_stock_dividends["股票股利"] = stock_dividends[: : -1]
    df_cash_and_stock_dividends = df_cash_and_stock_dividends.iloc[: len(years_list)]
    df_cash_and_stock_dividends.index = years_list
    df_cash_and_stock_dividends = df_cash_and_stock_dividends\
        .reset_index()\
        .rename(columns = {"index": "the_date_of_dividends"})\
        .sort_values("the_date_of_dividends", ascending = False)
    ten_years_average_dividends = round(df_cash_and_stock_dividends["現金股利"].sum() / 10, 2)

    df_dividends = pd.merge(df_before_dividends_date_and_stock_price, df_cash_and_stock_dividends, left_on = "Date_without_time", 
                            right_on = "the_date_of_dividends")
    df_dividends["殖利率"] = round(df_dividends["現金股利"] / df_dividends["Close"], 4)
    df_dividends["現金股利"] = round(df_dividends["現金股利"], 2)
    dividends_yield = round(df_dividends["殖利率"].sum() / 10, 4)

    # print("近10年平均股利:", ten_years_average_dividends)
    # print("近10年平均殖利率:", dividends_yield)
    # df_dividends
    # ratios_values

    dict_format = {
    "stockCode": stock_id, 
    "stockPrice": {
        "currentPrice": stock_price,         
        "priceChange": stock_price_increase_or_decrease,          
        "priceChangePercent": stock_price_increase_or_decrease_percentage,   
        "expensivePrice": expensive_stock_price,       
        "cheapPrice": cheap_stock_price           
    }
    }
    dict_format["warning"] = {}
    for i in range(len(ratios)):
        dict_format["warning"][ratios[i]] = ratios_values[i]

    dict_format["dividendHistory"] = {
        "years": [],
        "last10YearsDividends": [],
        "last10YearsDividendYield": []
    }

    for i in range(len(df_dividends)):
        row = df_dividends.iloc[i]
        # print(f"{row['Year']}Q{row['Quarter']}", row["現金股利"], row["殖利率"])
        dict_format["dividendHistory"]["years"].append(f"{row['Year']}Q{row['Quarter']}")
        dict_format["dividendHistory"]["last10YearsDividends"].append(row["現金股利"])
        dict_format["dividendHistory"]["last10YearsDividendYield"].append(row["殖利率"])

    dict_format["averages"] = {
    "averageDividend": ten_years_average_dividends,
    "averageDividendYield": dividends_yield
    }

    return json.dumps(dict_format, ensure_ascii = False, indent = 2)

if __name__ == "__main__":
    app.run(debug = True)