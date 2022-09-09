"""
CPH Security Queue
"""
import matplotlib.pyplot as plt
import requests
import json
import pandas as pd
from datetime import datetime, timezone, timedelta
import numpy as np
import streamlit as st
import urllib.request
import datetime
import calendar
import datetime
from lightgbm import LGBMRegressor
import time as t

@st.experimental_memo
def load_data():
    data = urllib.request.urlopen("https://cphapi.simonottosen.dk/waitingtime?select=id,t2waitingtime,deliveryid").read()
    output = json.loads(data)
    dataframe = pd.DataFrame(output)
    StartTime = dataframe["deliveryid"]
    StartTime = pd.to_datetime(StartTime)
    StartTime = StartTime.apply(lambda t: t.replace(tzinfo=None))
    StartTime = StartTime + pd.DateOffset(hours=2)
    dataframe["Date and time"] = StartTime
    dataframe["Queue"] = dataframe["t2waitingtime"]
    dataframe["Time"] = StartTime.dt.time
    dataframe["Date"] = StartTime.dt.date
    return dataframe

def load_latest():
    data = urllib.request.urlopen("https://cphapi.simonottosen.dk/waitingtime?select=id,t2waitingtime,deliveryid&order=id.desc&limit=2").read()
    output = json.loads(data)
    dataframe = pd.DataFrame(output)
    delta = dataframe["t2waitingtime"][0] - dataframe["t2waitingtime"][1]
    latest = dataframe["t2waitingtime"][0]
    delta = np.int16(delta).item()
    deltapure = np.int16(delta).item()
    latest = np.int16(latest).item()
    M = ' minute'
    S = 's'
    if latest == 1:
        latest = (str(latest) + M)
    else:
        latest = (str(latest) + M + S)
    if delta == 1:
        delta = (str(delta) + M)
    else:
        delta = (str(delta) + M + S)
    latest_update = dataframe["deliveryid"]
    return latest, delta, deltapure, latest_update


def load_last_two_hours():
    data = urllib.request.urlopen("https://cphapi.simonottosen.dk/waitingtime?select=t2waitingtime&order=id.desc&limit=24").read()
    output = json.loads(data)
    dataframe = pd.DataFrame(output)
    two_hours_avg = dataframe['t2waitingtime'].to_list()
    def Average(l): 
        avg = sum(l) / len(l) 
        return avg
    average = round(Average(two_hours_avg))
    M = ' minute'
    S = 's'
    if average == 1:
        average = (str(average) + M)
    else:
        average = (str(average) + M + S)
    return average    


def findDay(date):
    date = str(date)
    born = datetime.datetime.strptime(date, '%Y-%m-%d').weekday()
    return (calendar.day_name[born])

def new_model(test):
    modeldatetime = test["deliveryid"]
    modeldatetime = pd.to_datetime(modeldatetime)
    test["deliveryid"] = modeldatetime
    test = test.set_index(test.deliveryid)
    test.drop('deliveryid', axis=1, inplace=True)
    test['hour'] = test.index.hour
    test['day'] = test.index.day
    test['month'] = test.index.month
    test['weekday'] = test.index.weekday
    data = urllib.request.urlopen("https://cphapi.simonottosen.dk/waitingtime?select=id,t2waitingtime,deliveryid").read()
    output = json.loads(data)
    dataframe = pd.DataFrame(output)
    StartTime = dataframe["deliveryid"]
    StartTime = pd.to_datetime(StartTime)
    StartTime = StartTime.apply(lambda t: t.replace(tzinfo=None))
    StartTime = StartTime + pd.DateOffset(hours=2)
    dataframe["deliveryid"] = StartTime
    df = dataframe.set_index(dataframe.deliveryid)
    df.drop('deliveryid', axis=1, inplace=True)
    df['hour'] = df.index.hour
    df['day'] = df.index.day
    df['month'] = df.index.month
    df['weekday'] = df.index.weekday
    df.drop(['id'], axis=1, inplace=True)
    X = df.drop('t2waitingtime', axis=1)
    y = df['t2waitingtime']
    X_train = X.iloc[:]
    y_train = y.iloc[:]    
    model = LGBMRegressor(random_state=42)
    model.fit(X_train, y_train)
    predict = model.predict(test)
    return round(predict[0])


st.set_page_config(page_icon="👮🏼", page_title="CPH Security Queue")



hide_streamlit_style = """
            <style>
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
#root > div:nth-child(1) > div > div > div > div > section > div {padding-top: 1rem;}

footer {
	
	visibility: hidden;
	
	}
footer:after {
	content:'Made with ❤️ by Simon Ottosen'; 
	visibility: visible;
	display: block;
	position: relative;
	#background-color: red;
	padding: 5px;
	top: 2px;
}
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

# Using object notation
M = " minute"
S = "s"


currenttime = load_latest()
latest_update = load_latest()
average = load_last_two_hours()
st.title("CPH Security Queue ✈️ 👮🏼 ")
#in2hours = datetime.datetime.now() + timedelta(hours=2)
#in2hours = pd.DataFrame({'deliveryid': [in2hours]}) 
M = " minute"
S = "s"

col1, col2 = st.columns(2)
if currenttime[2] == 0:
    col1.metric(label="Current waiting time ⏰", value=currenttime[0])
else:
    col1.metric(label="Current waiting time ⏰", value=currenttime[0], delta=currenttime[1], delta_color="inverse")
col2.metric(label="Average waiting time in last 2 hours 🕵🏼", value=average)
#col3.metric(label="Expected waiting time in 2 hours 🔮", value=in2hours)

with st.form("Input"):
    st.subheader("Get queing time for your upcoming flight")
    date = st.date_input(
        "On what date is your flight?")
    time = st.time_input('At what time would you expect to arrive at the airport?', help="Usually you should arrive approx. 2 hours before your flight if bringing luggage and 1 hour before if you are only bringing carry-on")
    datetime_queue_input_text = ('You will be flying out at ' + str(time.strftime("%H:%M")) + ' on a ' + str(findDay(date)))
    datetime_queue_input = (str(date) + ' ' + str(time))
    test = pd.DataFrame({'deliveryid': [datetime_queue_input]}) 
    st.write('Calculation might take a few seconds')
    btnResult = st.form_submit_button('Calculate')
    if btnResult:
        with st.spinner('Estimating queue...'):
            prediction = new_model(test)
        if prediction == 1:
            prediction = (str(prediction) + M)
        else:
            prediction = (str(prediction) + M + S)
        st.write('Expected queue at ' + str(time.strftime("%H:%M")) + ' on ' + str(findDay(date)) + ' is ' + str(prediction) + ' ✈️')
