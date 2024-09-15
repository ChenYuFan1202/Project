import re
import os
import json
import time
import requests
import pandas as pd
# import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from bs4 import BeautifulSoup as bs
from flask_cors import CORS
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from flask import Flask, make_response, request
from langchain.chains import load_summarize_chain
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from pydantic import BaseModel, Field
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

    reply = client.chat.completions.create(
        model = "gpt-3.5-turbo",
        # model = "gpt-4",
        messages = [
            {"role": "system", "content":  "你現在是一位專業的證券分析師, 你會統整近期的比率並進行分析, 然後生成一份專業的趨勢分析報告"},
            {"role": "user", "content": content_message} # user就是代表我們
        ]
    )
    return reply.choices[0].message.content # json.dumps(reply.choices[0].message.content, ensure_ascii = False, indent = 2)

@app.route("/get_three_years_data", methods = ["get"])
def three_years_data():
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
                        values.append("Undefined")
                    elif value.text == "":
                        values.append("Undefined")  
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

    dict_format = {"company": {"name": company_name,
                            "stock_id": stock_id,
                            "environmental": [{} for _ in range(33)],
                            "social": [{} for _ in range(21)],
                            "governance": [{} for _ in range(21)]}

    } # 11 7 7 11 7 7 11 7 7

    for i in range(len(values)): # 0 ~ 74
        epoch = i // 25 + 1
        j = i % 25 # 0 ~ 24
        value = values[i]
        name = names[j]
        if j // 11 < 1:
            if epoch == 1:
                dict_format["company"]["environmental"][j + (epoch - 1) * 11] = {"year": year, "index": name, "value": value}
            elif epoch == 2:
                dict_format["company"]["environmental"][j + (epoch - 1) * 11] = {"year": "2022", "index": name, "value": value}
            else:
                dict_format["company"]["environmental"][j + (epoch - 1) * 11] = {"year": "2021", "index": name, "value": value}
        elif j // 18 < 1:
            if epoch == 1:
                dict_format["company"]["social"][j % 11 + (epoch - 1) * 7] = {"year": year, "index": name, "value": value}
            elif epoch == 2:
                dict_format["company"]["social"][j % 11 + (epoch - 1) * 7] = {"year": "2022", "index": name, "value": value}
            else:
                dict_format["company"]["social"][j % 11 + (epoch - 1) * 7] = {"year": "2021", "index": name, "value": value}
        elif j // 25 < 1:
            if epoch == 1:
                dict_format["company"]["governance"][j % 18 + (epoch - 1) * 7] = {"year": year, "index": name, "value": value}
            elif epoch == 2:
                dict_format["company"]["governance"][j % 18 + (epoch - 1) * 7] = {"year": "2022", "index": name, "value": value}
            else:
                dict_format["company"]["governance"][j % 18 + (epoch - 1) * 7] = {"year": "2021", "index": name, "value": value}
    return json.dumps(dict_format, ensure_ascii = False, indent = 2)

@app.route("/get_one_year_data", methods = ["get"])
def one_year_data():
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
                            values.append("Undefined")
                        elif value.text == "":
                            values.append("Undefined")  
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

    dict_format = {"companies": [{} for _ in range(len(stock_ids))]
    }
    for i in range(len(stock_ids)):
        dict_format["companies"][i]["name"] = company_names[i]
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
            dict_format["companies"][0]["environmental"][index] = {"year": year, "index": name, "value": value}
        elif index < 18 and i < 25:
            dict_format["companies"][0]["social"][index % 11] = {"year": year, "index": name, "value": value}
        elif index < 25 and i < 25:
            dict_format["companies"][0]["governance"][index % 18] = {"year": year, "index": name, "value": value}
        elif index < 11 and i < 50:
            dict_format["companies"][1]["environmental"][index] = {"year": year, "index": name, "value": value}
        elif index < 18 and i < 50:
            dict_format["companies"][1]["social"][index % 11] = {"year": year, "index": name, "value": value}
        elif index < 25 and i < 50:
            dict_format["companies"][1]["governance"][index % 18] = {"year": year, "index": name, "value": value}
        elif index < 11 and i < 75:
            dict_format["companies"][2]["environmental"][index] = {"year": year, "index": name, "value": value}
        elif index < 18 and i < 75:
            dict_format["companies"][2]["social"][index % 11] = {"year": year, "index": name, "value": value}
        else:
            dict_format["companies"][2]["governance"][index % 18] = {"year": year, "index": name, "value": value}

    return json.dumps(dict_format, ensure_ascii = False, indent = 2)

@app.route("/get_annual_report_summary", methods = ["get"])
def annual_report_summary():
    def annual_report(id, y):
        url = "https://doc.twse.com.tw/server-java/t57sb01"

      # 建立 POST 請求的表單
        data = {
          "id": "",
          "key": "",
          "step": "1",
          "co_id": id, # 可以是數字
          "year": int(y) - 1911 + 1, # 要輸入民國年份 # 可以是數字
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
            "co_id": id,
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

    retriever = Load_FAISS_db.as_retriever(search_type = "similarity", search_kwargs = {"k": 3})

    tool = create_retriever_tool(
        retriever = retriever,
        name = "retriever_by_company_annual_report",
        description = "搜尋並返回公司年報內容"
    )
    tools = [tool]

    first_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位善用工具的好助理，請勿回答與公司年報內容無關的問題，這間台灣公司的股票代號是{stock_id}、此年報的年份是{year}。\n"
        "請自己判斷上下文來回答問題，如果不確定公司名字就不要亂掰，並且使用繁體中文和台灣用語，不要盲目地使用工具。"),
        MessagesPlaceholder(variable_name = "chat_history"),
        ("human", "{input}。"),
        MessagesPlaceholder(variable_name = "agent_scratchpad")
    ])
    first_prompt = first_prompt.partial(year = year, stock_id = stock_id)
    chat_model = ChatOpenAI(model = "gpt-3.5-turbo")
    first_agent = create_openai_tools_agent(chat_model, tools, first_prompt)
    first_agent_executor = AgentExecutor(agent = first_agent, tools = tools) # , verbose = True 



    memory = SQLChatMessageHistory(
        session_id = "test_id",
        connection = "sqlite:///history_db"
    )

    def window_messages(chain_input):
        # print(len(memory.messages))
        if len(memory.messages) > 4:
            cur_messages = memory.messages[-4: ]
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


    search_run = DuckDuckGoSearchRun()


    class SearchRun(BaseModel):
        query: str = Field(description = "給搜尋引擎的搜尋關鍵字, 請使用繁體中文")

    search_run = DuckDuckGoSearchRun(
        name = "ddg-search", 
        description = "使用網路搜尋你不知道的事物", 
        args_schema = SearchRun
    )

    second_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位善用工具的好助理，"
        "請利用關鍵字搜尋台灣公司的年報資訊，並且使用繁體中文和台灣用語，不要盲目地使用工具。"),
        ("human", "此為我感興趣的台灣公司股票代碼{stock_id}，請您幫我搜尋{year}年的年報資訊。"),
        MessagesPlaceholder(variable_name = "agent_scratchpad")
    ])

    search_tool = [search_run]

    second_agent = create_openai_tools_agent(chat_model, search_tool, second_prompt)
    second_agent_executor = AgentExecutor(agent = second_agent, tools = search_tool) # agent 連續使用工具會出事 # , verbose = True

    if "counter" not in globals():
        # print("執行了")
        globals()["counter"] = 0
        globals()["temp_stock_id"] = stock_id
        globals()["temp_year"] = year
        globals()["second_response"] = second_agent_executor.invoke({"stock_id": stock_id, "year": year})["output"]
        # counter += 1
    # if "counter" not in globals():
    #     counter = 0

    # if counter == 0:
    #     # print("執行了")
    #     temp_stock_id = stock_id
    #     temp_year = year
    #     second_response = second_agent_executor.invoke({"stock_id": stock_id, "year": year})["output"]
    #     counter += 1

    if globals()["temp_stock_id"] != stock_id or globals()["temp_year"] != year:
        # print("執行了!!")
        globals()["second_response"] = second_agent_executor.invoke({"stock_id": stock_id, "year": year})["output"]
        globals()["temp_stock_id"] = stock_id
        globals()["temp_year"] = year

    str_parser = StrOutputParser()

    third_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位台灣專業的財務分析師，"
        "你會根據此台灣公司的年報資訊來思考第一個 GenAI 所生出的回覆**是否正確**，第一個 GenAI 生出的回覆可能會錯，所以請務必注意!\n"
        "請你使用**繁體中文和台灣用語**來回答使用者。\n"
        "注意，內容要**以回答使用者所問的問題為主**，此外，不要提到資料來源。\n"
        "請務必遵照以下內容：\n 請**不要在開頭加入公司介紹與重複問題以及不要在結尾加入總結**，只需要**輸出回覆即可**。\n"
        "最後，請輸出方便使用者閱讀的格式，例如有標題等。"),
        ("human", "此為使用者問的問題{question}和第一個 GenAI 生出的內容{first_response}，不需要重複輸出問題。"),
        ("human", "以下為股票代碼為{stock_id}的台灣公司的{year}年報資訊:\n"
        "{second_response}")
    ])

    third_agent = third_prompt | chat_model | str_parser
    third_response = third_agent.invoke({"stock_id": stock_id, "year": year, 
                                        "first_response": first_response, "second_response": globals()["second_response"], "question": msg})
    # print(third_response)
    return third_response

if __name__ == "__main__":
    app.run(debug = True)
    