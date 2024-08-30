import bs4
import requests
import re
import json
import os
import pandas as pd
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import time

news_data_path = "./news_data/"
if not os.path.exists(news_data_path):
    os.makedirs(news_data_path)


def send_get(date, headers, page=0):
    flag = 0
    skip = 0
    resp = requests.Response
    try:
        time.sleep(0.5)
        resp = get_response(date, headers, page)
        if resp.status_code >= 500:
            skip = 1
            return skip, flag, resp
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(e)
        print(f"Prepare for another try.")
        try:
            time.sleep(10)
            resp = get_response(date, headers, page)
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(e)
            print("Eliminate.")
            flag = 1
    return skip, flag, resp


def get_response(date, headers, page=0):
    if page == 0:
        url = f"https://www.wsj.com/news/archive/{date}"
        resp = requests.get(url, headers=headers, verify=True)
        return resp
    else:
        url = f"https://www.wsj.com/news/archive/{date}?page={page}"
        resp = requests.get(url, headers=headers, verify=True)
        return resp


def time_text_format(time_: str):
    if time_[1] == ":":
        time_ = "0" + time_
    if "PM" in time_:
        hr = int(time_[:2])
        hr += 12
        str_hr = str(hr)
        str_time = str_hr + time_[2: 5]
    else:
        str_time = time_[:5]
    return str_time


def get_data(soup: bs4.BeautifulSoup):
    page_max = soup.find(class_="WSJTheme--pagepicker-total--Kl350I1l")
    page_cnt = 0
    if page_max is not None:
        page_num = page_max.get_text(strip=True)
        page_cnt = int(page_num[3])
    headlines, times = [], []
    articles = soup.find_all(
        class_="WSJTheme--story--XB4V2mLz WSJTheme--padding-top-large--2v7uyj-o styles--padding-top-large--3rrHKJPO WSJTheme--padding-bottom-large--2lt6ga_1 styles--padding-bottom-large--2vWCTk2s WSJTheme--border-bottom--s4hYCt0s")
    for article in articles:
        if article.find(class_='WSJTheme--headlineText--He1ANr9C'):
            headline = article.find(class_='WSJTheme--headlineText--He1ANr9C')
            headlines.append(headline.get_text())
        else:
            headlines.append("Notable &amp; Quotable")
        time_ = article.find(class_="WSJTheme--timestamp--22sfkNDv")
        times.append(time_text_format(time_.get_text()))
    return page_cnt, headlines, times


def get_headlines():
    start_date = datetime(2012, 10, 5)
    end_date = datetime(2020, 12, 31)
    delta = timedelta(days=1)
    date_list = []
    current_date = start_date
    while current_date <= end_date:
        date_list.append(current_date.strftime('%Y/%m/%d'))
        current_date += delta

    headers = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "User-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
        "Cookie": "MicrosoftApplicationsTelemetryDeviceId=410753dc-ba2c-3caa-1e04-212b902d4a9e; MicrosoftApplicationsTelemetryFirstLaunchTime=1723638071924; _sp_su=false; _ncg_domain_id_=fe612a1f-1488-40c2-8b1a-a3de9aca41b1.1.1720362108.1751898108; _pcid=%7B%22browserId%22%3A%22lybn7fm61l8ywruk%22%7D; cX_P=lybn7fm61l8ywruk; _pctx=%7Bu%7DN4IgrgzgpgThIC4B2YA2qA05owMoBcBDfSREQpAeyRCwgEt8oBJAEzIE4AmHgZgEYAHADZ%2BvQQFYuAFhEcADPJABfIA; _scid=2b7bbc84-5fe1-47e0-aec4-9ce8116eae85; _ncg_g_id_=2ad6d9e2-f451-48e7-b2ab-ac9190bbb65f.3.1720362132.1751898108; ajs_anonymous_id=fb7c171d-ed59-4b98-b468-c61bd6f01565; _fbp=fb.1.1720362168578.1223668280; _meta_facebookTag_sync=1720362168578; _meta_cross_domain_id=171f474d-2ecb-4854-9e34-8a5a7e38f010; _scor_uid=d24ee57c854e44e28c794938a6865d06; _fbp=fb.1.1720362168578.1223668280; _pin_unauth=dWlkPU9UaGlOekEyTmpZdE5UVmxPQzAwWldWbUxXSTFNbUV0TkRZMk1UUXdNVGhrWVRRMw; _gcl_au=1.1.760706240.1720362194; cX_G=cx%3Ak23ub7fokjxz1ql2d3mgoj05v%3A20qsec27jduq9; permutive-id=c518f217-cd33-49bd-9b32-8f3ae35c202a; wsjregion=na%2Cus; ab_uuid=cab709e1-de10-4e41-b4c6-da17344c2001; _ga=GA1.1.1787966660.1723134918; _dj_sp_id=532b3859-ed45-4e9d-88d9-02b173afafe1; _ScCbts=%5B%5D; _ncg_id_=fe612a1f-1488-40c2-8b1a-a3de9aca41b1; _meta_cross_domain_recheck=1723134926827; _sctr=1%7C1723132800000; _pubcid=42ec14e8-4ad3-4f6c-949f-64a83d9dbaa3; _lr_env_src_ats=false; gdprApplies=false; ccpaApplies=true; vcdpaApplies=true; regulationApplies=gdpr%3Afalse%2Ccpra%3Atrue%2Cvcdpa%3Atrue; AMCVS_CB68E4BA55144CAA0A4C98A5%40AdobeOrg=1; s_cc=true; _lr_geo_location_state=WA; _lr_geo_location=US; _lr_sampling_rate=0; __gads=ID=5331b77d4c5a0e09:T=1720362107:RT=1723637976:S=ALNI_MbVn6yYZrOPvaZXIklRKGsY6yxd7g; __gpi=UID=00000e85cb9cc93b:T=1720362107:RT=1723637976:S=ALNI_MZrTTWWq-PM176u-peySLpxDCP9zw; __eoi=ID=6de56c1df97e6aa7:T=1720362107:RT=1723637976:S=AA-AfjbVjhFsxRX5zASIW7D0i5MY; AMCV_CB68E4BA55144CAA0A4C98A5%40AdobeOrg=1585540135%7CMCIDTS%7C19950%7CMCMID%7C83887209895423079864309010065544947139%7CMCAAMLH-1724242774%7C9%7CMCAAMB-1724242774%7CRKhpRz8krg2tLO6pguXWp5olkAcUniQYPHaMWWgdJ3xzPWQmdj0y%7CMCOPTOUT-1723645174s%7CNONE%7CMCAID%7CNONE%7CvVersion%7C4.4.0; _pubcid_cst=CyzZLLwsaQ%3D%3D; utag_main=v_id:01908d928485001d7a4124e93ba505075002b06d00942$_sn:14$_se:3$_ss:0$_st:1723639841146$vapi_domain:wsj.com$ses_id:1723637970006%3Bexp-session$_pn:3%3Bexp-session$_prevpage:WSJ_Summaries_Archive_NewsArchive%3Bexp-1723641641151; _dj_id.9183=.1723556684.3.1723638054.1723597250.d60f8437-4f73-4c88-a92b-6201ad4bc677.cf81ab76-8257-476b-8f55-3c1be62bd37a.cbeedd5a-d993-4762-a020-e2fabd2e4156.1723638023461.2; _ga_K2H7B9JRSS=GS1.1.1723637589.13.1.1723638055.60.0.0; _scid_r=2b7bbc84-5fe1-47e0-aec4-9ce8116eae85; _rdt_uuid=1720362168366.51f184fe-60c4-4dd3-9291-d1c2865da75a; _uetsid=872e886057e011efbe14459a2cc39406; _uetvid=5cbc9e003c6c11efb42579a89649ab86; _ncg_sp_id.5378=fe612a1f-1488-40c2-8b1a-a3de9aca41b1.1720362161.14.1723638070.1723597261.e95ce494-a379-4a08-9541-0fabc313c7ed; cto_bundle=U6w2yF8wcFl5dFdEWUxmellhJTJCbTdzamdpR05CQnlybEd4QnRiaUNwbWVxWlFLJTJCWHNsJTJCY2ZHSFJFUnlEdnJsUzZJMjQzaEt1ViUyRmtKRXFtZmFZUHBDU1NVS0ElMkZPNkZxTW1iSk03c2JXcEtHVTcwSGVRVkYlMkIxZFBZZFpaN0NYUjFTbGZtR082MmRyaEZ0U1pxRzIzeFYlMkZ3dnM5cjFKck5UQXJ5Y0Z4RGNLZXFSSzlEY0pCM0QlMkJaRGRqdDFsbzl2YzBycSUyRlQ; datadome=0Aw1PEw6l30BJ0F4S0Z3oCT~IUSwEEaeoiyXEq_BjDI1ePGqeC3oYQhbseH8W~PkruJS1iPTHoDWLh9eBkDbBKgnvutCaa7z0rIPsKTGUMD8ZTOqSS2sS_21bFXBq6gw; s_tp=6093; s_ppv=WSJ_Summaries_Archive_NewsArchive%2C90%2C90%2C5480.5"
    }

    df_list = []

    for i, date in enumerate(date_list):
        print(date)
        flag, skip = 0, 0
        skip, flag, resp = send_get(date, headers)
        if skip:
            continue
        if flag:
            break
        headline_all = []
        time_all = []
        soup = BeautifulSoup(resp.text, "html.parser")
        page_cnt, headlines, times = get_data(soup)
        headline_all.extend(headlines)
        time_all.extend(times)
        if len(headline_all) != len(time_all):
            print(len(headline_all), len(time_all))
            print(date)
            break
        for page in range(2, page_cnt + 1):
            skip, flag, resp = send_get(date, headers, page)
            if skip:
                continue
            if flag:
                break
            headline_all = []
            time_all = []
            soup = BeautifulSoup(resp.text, "html.parser")
            _, headlines, times = get_data(soup)
            headline_all.extend(headlines)
            time_all.extend(times)
            if len(headline_all) != len(time_all):
                print(date, page)
                print(len(headline_all), len(time_all))
                flag = 1
                break
        if flag:
            break

        num_headlines = len(headline_all)
        date_all = [date for i in range(num_headlines)]

        df = pd.DataFrame({
            "headline": headline_all,
            "date": date_all,
            "time": time_all
        })

        df_list.append(df)

    final_df = pd.concat(df_list, ignore_index=True)

    final_df.to_csv(os.path.join(news_data_path, "./wsj_headlines_h.csv"), index=False)


if __name__ == "__main__":
    get_headlines()
