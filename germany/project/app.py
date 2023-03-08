import psycopg2
import requests
import json
from datetime import datetime
import os
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import urllib.request
import re

def frankfurt():
    POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD")
    POSTGRES_DB = os.environ.get("POSTGRES_DB")
    POSTGRES_USER = os.environ.get("POSTGRES_USER")
    POSTGRES_PORT = os.environ.get("POSTGRES_PORT")
    POSTGRES_HOST = os.environ.get("POSTGRES_HOST")
    CPHAPI_HOST = os.environ.get("CPHAPI_HOST")
    HEALTHCHECK = os.environ.get("FRA_HEALTHCHECK")
    GOOGLE_APPLICATION_CREDENTIALS = '/home/user/app/keyfile.json'

    response = requests.get('https://www.frankfurt-airport.com/wartezeiten/rest/waz?lang=en')
    waitingtime = json.loads(response.text)

    for item in waitingtime["data"]:
        if item["ps"] == "Security checkpoint, Concourse A\r\nDeparture gates: A1 - A69":
            status = item["status"]
            break

    numbers = list(map(int,re.findall('\d+', str(status))))
    print(numbers[0])


    if isinstance(numbers[0], int):
        queue = numbers[0]
    else:
        print("Error: Information not found in the JSON data.")

    now_utc = datetime.utcnow()
    timestamp = now_utc.strftime('%Y-%m-%dT%H:%M:%S')
    airport = "FRA"
    print(queue, timestamp, airport)

    cred = credentials.Certificate(GOOGLE_APPLICATION_CREDENTIALS)
    firebase_admin.initialize_app(cred)
    db = firestore.client()

    try:
        connection = psycopg2.connect(user=POSTGRES_USER,
                                      password=POSTGRES_PASSWORD,
                                      host=POSTGRES_HOST,
                                      port=POSTGRES_PORT,
                                      database=POSTGRES_DB)
        cursor = connection.cursor()

        postgres_insert_query = """ INSERT INTO waitingtime (queue, timestamp, airport) VALUES (%s,%s,%s)"""
        record_to_insert = (queue, timestamp, airport)
        cursor.execute(postgres_insert_query, record_to_insert)
        connection.commit()
        count = cursor.rowcount
        print(count, "Record inserted successfully into waiting time table")
        print("PostgreSQL connection is closed")

        # Delete the below code to remove support for Google Firebase
        print("Attempting to get data from public URL")
        apiurl=(str(CPHAPI_HOST)+str("/waitingtime?&order=id.desc&limit=1"))
        data = urllib.request.urlopen(apiurl).read()
        output = json.loads(data)
        aDict = output[0]
        print(aDict)
        data = {
        u'id': str(aDict['id']),
        u'queue': str(aDict['queue']),
        u'timestamp': str(aDict['timestamp']),
        u'airport': str(aDict['airport'])
        }
        db.collection(u'waitingtimetest').document(str(aDict['id'])).set(data)
        if HEALTHCHECK == "NULL":
            print("Skipping. Healthcheck not configured")
        else:
            print("Attempting to do healthcheck")
            healthcheckurl=(str("https://hc-ping.com/")+str(HEALTHCHECK))
            requests.get(healthcheckurl, timeout=10)
    except (Exception, psycopg2.Error) as error:
        print("Failed to synchronize data with Firebase due to ", error)

    finally:
        # closing database connection.
        if connection:
            cursor.close()
            connection.close()


def dusseldorf():
    POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD")
    POSTGRES_DB = os.environ.get("POSTGRES_DB")
    POSTGRES_USER = os.environ.get("POSTGRES_USER")
    POSTGRES_PORT = os.environ.get("POSTGRES_PORT")
    POSTGRES_HOST = os.environ.get("POSTGRES_HOST")
    CPHAPI_HOST = os.environ.get("CPHAPI_HOST")
    HEALTHCHECK = os.environ.get("DUS_HEALTHCHECK")
    GOOGLE_APPLICATION_CREDENTIALS = '/home/user/app/keyfile.json'

    url = "https://www.dus.com/api/sitecore/flightapi/WaitingTimes?lang=en"
    headers = {"X-Requested-With": "XMLHttpRequest"}
    response = requests.get(url, headers=headers)
    waitingtime = json.loads(response.text)

    for item in waitingtime['data']:
        # check if the name in the current item equals 'Sicherheitskontrolle A'
        if item['name'] == 'Sicherheitskontrolle A':
            numbers = item['waitingTime']
            if isinstance(numbers, int):
                queue = numbers   # Assign a value to queue
            else:
                print("Error: Information not found in the JSON data.")

    if queue is None:
        print("Error: Value of 'queue' was not set. Message from website was ", waitingtime)
    else:
        now_utc = datetime.utcnow()
        timestamp = now_utc.strftime('%Y-%m-%dT%H:%M:%S')
        airport = "DUS"
        print(queue, timestamp, airport)

    cred = credentials.Certificate(GOOGLE_APPLICATION_CREDENTIALS)
    firebase_admin.initialize_app(cred)
    db = firestore.client()

    try:
        connection = psycopg2.connect(user=POSTGRES_USER,
                                      password=POSTGRES_PASSWORD,
                                      host=POSTGRES_HOST,
                                      port=POSTGRES_PORT,
                                      database=POSTGRES_DB)
        cursor = connection.cursor()

        postgres_insert_query = """ INSERT INTO waitingtime (queue, timestamp, airport) VALUES (%s,%s,%s)"""
        record_to_insert = (queue, timestamp, airport)
        cursor.execute(postgres_insert_query, record_to_insert)
        connection.commit()
        count = cursor.rowcount
        print(count, "Record inserted successfully into waiting time table")
        print("PostgreSQL connection is closed")

        # Delete the below code to remove support for Google Firebase
        print("Attempting to get data from public URL")
        apiurl=(str(CPHAPI_HOST)+str("/waitingtime?&order=id.desc&limit=1"))
        data = urllib.request.urlopen(apiurl).read()
        output = json.loads(data)
        aDict = output[0]
        print(aDict)
        data = {
        u'id': str(aDict['id']),
        u'queue': str(aDict['queue']),
        u'timestamp': str(aDict['timestamp']),
        u'airport': str(aDict['airport'])
        }
        db.collection(u'waitingtimetest').document(str(aDict['id'])).set(data)
        if HEALTHCHECK == "NULL":
            print("Skipping. Healthcheck not configured")
        else:
            print("Attempting to do healthcheck")
            healthcheckurl=(str("https://hc-ping.com/")+str(HEALTHCHECK))
            requests.get(healthcheckurl, timeout=10)
    except (Exception, psycopg2.Error) as error:
        print("Failed to synchronize data with Firebase due to ", error)

    finally:
        # closing database connection.
        if connection:
            cursor.close()
            connection.close()


