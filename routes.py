import re
import json
import time
import requests
# import numpy as np
import pandas as pd
from openai import OpenAI
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

if __name__ == "__main__":
    app.run(debug = True)
    