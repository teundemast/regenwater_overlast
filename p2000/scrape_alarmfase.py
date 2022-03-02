import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from datetime import datetime, timedelta

import time
import json

# The URL for API calls is the following: it ends with a timestamp and will return a number of p2000 which are the closest but lower than the given time
url = "http://api.alarmfase1.nl/3.2/calls/fire.json?&end={}"


strfformat = "%Y-%m-%d %H:%M:%S" # Format of the timestamp which is formatted into the URL
enddate = datetime(year=2021, month=3, day=5, hour=0, minute=0, second=0)
prevdate = datetime.now()

out_file = open("final.json", "w")

list_wateroverlast_calls = [] # List of p2000 calls which contain the word 'wateroverlast'

while prevdate > enddate:
    # A session is made so it possible to give a number of max retries for an API call (to avoid errored API calls)
    session = requests.Session()
    retry = Retry(connect=5, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    data = session.get(url.format(prevdate.strftime(strfformat)))
    
    try:
        calls = data.json()["calls"]
    except ValueError: # This happens when the return value of the API call cannot be formatted into a json object 
        prevdate = prevdate - timedelta(hours=5)
        continue

    for c in calls:
        if 'wateroverlast' in c['message'].lower():
            list_wateroverlast_calls.append(c) # Only calls which are possibly relevant are saved 

    prevdate = datetime.strptime(calls[-1]["date"], strfformat) # This becomes the timestamp of the earliest p2000 call. Note: this does not have to be a call which is a 'wateroverlast' call. 

    time.sleep(0.1)

data = {"data": list_wateroverlast_calls}

json.dump(data, out_file, indent=6)
out_file.close()
