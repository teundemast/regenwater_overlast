from osgeo import gdal

# this script generates the dataset which we can apply machine learning to
# 1. get all positive examples
# 2. enrich positive examples with rain information
# 3. keep only the examples with enough precipitation
# 4. find n negative examples in a certain range for each positive example using the bag list
# 5. enrich negative examples with precipitation data
# 6. enrich all examples with height, water, infiltration, non-infiltration, house, center-house


import numpy as np
import pandas as pd
import random

begin = '2016-01-02 00:00:00'
# begin = '2022-01-02 00:00:00'
end = '2022-01-30 23:59:59'
strfformat = "%Y-%m-%d %H:%M:%S"

from helpers import rdconverter
from neerslag import precipitation_nl
from slak import ahn_layer
from depression import calcDepression, scalar

from datetime import datetime
import time
import os

total = 0
count = 0
btime = time.time()
rain_treshold = 500

# helper functions
def get_data(data, a):
    d = data
    # print(d)
    if len(d) > 0 and d[0]['geometry']['location_type'] != 'APPROXIMATE':
        return data[0]['geometry']['location'][a]
    else:
        return np.nan

def parse_datetime(data):
    return datetime.strptime(str(data['date']).strip(), strfformat)

def get_precipitation_data(row):
    global count
    count +=1
    now = time.time()
    avg_time = (now-btime)/count
    left = total-count
    if count % 10 == 0:
        print('====== rain ')
        print('time spent', now-btime)
        print('did', count, 'examples')
        print('avg', avg_time)
        print('left', left)
        print('time left', left*avg_time)
        print('======')

    date = row['date']
    date = datetime.strptime(str(row['date']), strfformat)
    lat = row['lat']
    lon = row['lng']
    rain = PNL.get_precipation_data_past_hours_list(date.year, date.month, date.day, date.hour, date.minute, lat, lon, 12)
    
    return rain

def get_past3hours(date, lat, lon):
    rain = PNL.get_precipation_data_past_hours_list(date.year, date.month, date.day, date.hour, date.minute, lat, lon, 3)
    return sum(rain)
    
def calc_sums(row):
    data = []
    try:
        lst = list(row['prec12'])
        
        for j in range(12):
            d = np.asarray(lst[j*12:j*12+12])
            d = np.where(d == 65535, 0, d)
            d = np.where(d == np.nan, 0, d)
            d = np.where(d < 0, 0, d)
            data.append(np.sum(d))
            
    except:
        data = [0]*12
        
    return data

def same_day(datetime1, datetime2):
    return datetime1.year == datetime2.year and datetime1.month == datetime2.month and datetime1.day == datetime2.day

#https://stackoverflow.com/questions/50559078/generating-random-dates-within-a-given-range-in-pandas
def random_datetimes_or_dates(start, end, out_format='datetime', n=10): 

    '''   
    unix timestamp is in ns by default. 
    I divide the unix time value by 10**9 to make it seconds (or 24*60*60*10**9 to make it days).
    The corresponding unit variable is passed to the pd.to_datetime function. 
    Values for the (divide_by, unit) pair to select is defined by the out_format parameter.
    for 1 -> out_format='datetime'
    for 2 -> out_format=anything else
    '''
    (divide_by, unit) = (10**9, 's') if out_format=='datetime' else (24*60*60*10**9, 'D')

    start_u = start.value//divide_by
    end_u = end.value//divide_by

    return pd.to_datetime(np.random.randint(start_u, end_u, n), unit=unit).to_pydatetime()[0] 


# step 1
df1 = pd.read_json('src/data/merged_precise_coord.json')
# df1 = df1.head()
df1 = df1.dropna()
df1['lat'] = df1['google_results'].apply(get_data, args=('lat',))
df1['lng'] = df1['google_results'].apply(get_data, args=('lng',))
df1['target'] = 1
df1 = df1[['lat','lng','target','date']]
df1 = df1.dropna()

df1['date'] = df1.apply(parse_datetime, axis=1)
df1 = df1[(begin < df1['date']) & (df1['date'] < end)]
# df1 = df1[150:200]

# step 2
PNL = precipitation_nl.PrecipitationNL(queue_size=300)
total = len(df1)
count = 0
btime = time.time()
df1 = df1.sort_values(by=['date'])
df1['prec12'] = df1.apply(get_precipitation_data, axis=1)


# step 3
df1['prec_sums'] = df1.apply(calc_sums, axis=1)
cols = ['sum'+str(i+1) for i in range(12)]
df1[cols] = pd.DataFrame(df1.prec_sums.tolist(), index=df1.index)
df1['past3hours'] = df1['sum1'] + df1['sum2'] + df1['sum3']


df1 = df1[(df1.past3hours > rain_treshold)]
df1 = df1.drop(columns=cols)
df1 = df1.drop(columns=['prec12', 'prec_sums'])

# print(df1)
# step 4
data0 = {'lat':[], 'lng':[], 'date':[], 'target': [], 'past3hours': []}

def sample_dependent(data):
    lat = data["lat"]
    lng = data["lng"]
    data0["lat"].append(lat)
    data0["lng"].append(lng)
    data0['target'].append(0)
    start_date = pd.to_datetime(begin)
    end_date = pd.to_datetime(end)
    random_date = random_datetimes_or_dates(start_date, end_date,n=1, out_format="datetime")
    data0["date"].append(random_date)
    max_rain = data["past3hours"]
    data0["past3hours"].append(random.randint(rain_treshold, max_rain))
    print("Negative sample added")

df1.apply(sample_dependent, axis=1)
df0 = pd.DataFrame(data0)

df = pd.concat([df0, df1], ignore_index=True)


# # step 6
total = len(df)
count = 0
btime = time.time()
ahn = ahn_layer.AHNLayer()

def add_layers(data):
    try:
        global count
        count +=1
        now = time.time()
        avg_time = (now-btime)/count
        left = total-count
        if count % 10 == 0:
            print('====== layers ')
            print('time spent', now-btime)
            print('did', count, 'examples')
            print('avg', avg_time)
            print('left', left)
            print('time left', left*avg_time)
            print('======')

        lat = data['lat']
        lng = data['lng']
        rdx = rdconverter.gps2X(lat,lng)
        rdy = rdconverter.gps2Y(lat,lng)

        x, y = round(rdx, 2), round(rdy, 2)
        d = 150
        arr = ahn.get_gdal_dataset(x-d, x+d, y-d, y+d)
        arr = arr.ReadAsArray()
        arr1 = scalar(arr, 0.2)
        arr2 = scalar(arr, 0.4)
        arr3 = scalar(arr, 0.6)
        arr4 = scalar(arr, 0.8)
        score = calcDepression(arr)
        score1 = calcDepression(arr1)
        score2 = calcDepression(arr2)
        score3 = calcDepression(arr3)
        score4 = calcDepression(arr4)
        
        array = np.array([score, score1, score2, score3, score4])
        return array
    except Exception as e:
        print(e)
        return None

df['layers'] = df.apply(add_layers, axis=1)

dir_ = '/local/s2656566/wateroverlast/regenwater_overlast/src/data/'


df.to_pickle(dir_ + 'dataset_depression.pkl', protocol=4)

print(df)
