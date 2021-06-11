#!/usr/bin/env python
# coding: utf-8

# # Pulling from the UMD / Facebook API
# 
# https://covidmap.umd.edu/api.html
# 
# Silently truncates if response rows go above 3600. I ensure a full scrape by raising an exception if it does.

import requests
import json
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Countries in the set
COUNTRY_API = "https://covidmap.umd.edu/api/country"
countries = requests.get(COUNTRY_API).text
jsonData = json.loads(countries)
countries = [c['country'] for c in jsonData["data"]]



def build_all_request(start, end) :
    BASE = "https://covidmap.umd.edu/api/resources?"
    IND = "indicator=mask"
    typ = "type=daily" 
    dates = f"&daterange={start}-{end}"
    country = "country=all"
    region = "region=all"
    
    fields = [BASE, IND, country, typ, region, dates]
    
    return "&".join(fields)

    
def build_country_request(country, start, end, typ="daily", isRegion=False) :
    BASE = "https://covidmap.umd.edu/api/resources?"
    IND = "indicator=mask"
    typ = f"type={typ}" 
    # This doesn't work, capped if more than a month
    #country = "country=all"
    country = f"country={country}"
    dates = f"&daterange={start}-{end}"
    fields = [BASE, IND, country, typ, dates]
    
    if isRegion :
        region = "region=all"
        fields += [region]
    
    return "&".join(fields) 


def build_dataframe(req) :
    response = requests.get(req)
    if response.status_code != 500 :
        jsonData = json.loads(response.text)
        if jsonData["status"] != 'success' :
            raise ValueError("Truncated, abort")
        cdf = pd.DataFrame.from_records(jsonData["data"])
    else :
        cdf = pd.DataFrame()
        
    return cdf


def increment_date(dstr, fmt="%Y%m%d", days=6) :
    d = datetime.datetime.strptime(dstr, fmt)
    td = datetime.timedelta(days=days)
    
    return (d + td).strftime(fmt)


def get_all(country=None, isRegionalRequest=True) :
    if isRegionalRequest :
        days_per_iter = 5
    else :
        days_per_iter = 10
    start = "2020101"
    end = increment_date(start, days=days_per_iter)
    df = pd.DataFrame()
    
    its = 365 // days_per_iter
    
    for i in range(its) :
        if isRegionalRequest :
            req = build_all_request(start, end)
        else :
            req = build_country_request(country, start, end)
        idf = build_dataframe(req)
        df = pd.concat([df,idf])
        
        start = increment_date(start, days=days_per_iter)
        end = increment_date(end, days=days_per_iter)
        
    return df


df = pd.DataFrame()
for c in countries :
    print(c)
    cdf = get_all(country=c, isRegionalRequest=False)
    df = pd.concat([df, cdf])


df.to_csv("umd/umd_national_wearing.csv", index=False)



