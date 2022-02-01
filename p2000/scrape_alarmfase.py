import requests

from datetime import datetime

import time
import json

url = "http://api.alarmfase1.nl/3.2/calls/fire.json?&end={}"

strfformat = "%Y-%m-%d %H:%M:%S"
enddate = datetime(year=2016, month=1, day=1, hour=0, minute=0, second=0)
prevdate = datetime.now()

while prevdate > enddate:
    data = requests.get(url.format(prevdate.strftime(strfformat)))

    if data.status_code != 200:
        data = requests.get(url.format(prevdate.strftime(strfformat)))

    calls = data.json()['calls']
    for c in calls:
        mess = c['message'].lower()
        if 'wateroverlast' in mess:
            print(json.dumps(c)+ ",")
            
    prevdate = datetime.strptime(calls[-1]['date'], strfformat)

    time.sleep(0.1)
