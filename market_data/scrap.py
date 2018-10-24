#!/usr/bin/python


from urllib.request import urlopen
import json
import pandas as pd

#ret = json.load(urllib2.urlopen(r"https://api.gdax.com/products/BTC-EUR/candles?start=2017-08-16T12:00:00&end=2017-08-16T13:00:00&granularity=600"))

import datetime
import time

# timepoint = datetime.datetime(2015, 3, 1, 0, 0, 0, 0)
#timepoint = datetime.datetime.fromtimestamp(1491321600)
granularity = datetime.timedelta(minutes=5)

def candle_gen(timepoint, granularity):

    gran_str = str(granularity.seconds)

    end = timepoint + granularity * 192
    start = timepoint - granularity * 96
    timepoint = start
    url = "https://api.gdax.com/products/BTC-USD/candles?start=" + start.isoformat()\
          + "&end=" + end.isoformat() + "&granularity=" + gran_str
    resp = json.load(urlopen(url))
    resp.reverse()
    for row in resp:
        yield row
    pass
    time.sleep(0.5)

"""
g = candle_gen(timepoint, granularity)

for timestamp, low, high, open_, close_, vol in g:
    iso = datetime.datetime.fromtimestamp(timestamp).isoformat()
    print("{},{},{},{},{},{},{}".format(iso,timestamp, low, high, open_, close_, vol))
"""



# [1502840700, 3551.41, 3573.03, 3567.2, 3566.81, 5.052059229999997]
# [ time,      low,     high,    open,   close,   volume ]

# write data to a csv data
days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
year_list = [2015, 2016, 2017, 2018]

for year in year_list:
    for month in range(1, 13):
        print(year, month)
        if year == 2015 and month in [1, 2]:
            continue
        if year == 2018 and month > 4:
            continue
        day_list = range(1, days[month - 1] + 1)
        for day in day_list:
            timepoint = datetime.datetime(year, month, day, 0, 0, 0, 0)
            g = candle_gen(timepoint, granularity)
            temp_df = pd.DataFrame.from_records(g, columns= ['timestamp', 'low', 'high', 'open', 'close', 'vol'])
            iso_list = [datetime.datetime.fromtimestamp(temp).isoformat() for temp in temp_df['timestamp']]
            temp_df['time'] = iso_list
            temp_df.to_csv('gdax_data.csv', mode='a', header=None, index=0)

