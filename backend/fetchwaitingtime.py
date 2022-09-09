import psycopg2
import requests
import json
from datetime import datetime
import os
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import urllib.request

POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD")
POSTGRES_DB = os.environ.get("POSTGRES_DB")
POSTGRES_USER = os.environ.get("POSTGRES_USER")
POSTGRES_PORT = os.environ.get("POSTGRES_PORT")

response = requests.get('https://cph-flightinfo-prod.azurewebsites.net//api/v1/waiting/get?type=ventetid')
waitingtime = json.loads(response.text)
waitingtime_json = json.dumps(waitingtime)
t2WaitingTime = (waitingtime["t2WaitingTime"])
t2WaitingTimeInterval = (waitingtime["t2WaitingTimeInterval"])
deliveryId = (waitingtime["deliveryId"])
deliveryId = (deliveryId.replace("T", " ")) 
print(t2WaitingTime, deliveryId)

GOOGLE_APPLICATION_CREDENTIALS = '/code/cphairportqueue-firebase-key.json'
cred = credentials.Certificate("/code/cphairportqueue-firebase-key.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

try:
    connection = psycopg2.connect(user=POSTGRES_USER,
                                  password=POSTGRES_PASSWORD,
                                  host="cph_postgres_db",
                                  port=POSTGRES_PORT,
                                  database=POSTGRES_DB)
    cursor = connection.cursor()

    postgres_insert_query = """ INSERT INTO waitingtime (t2WaitingTime, t2WaitingTimeInterval, deliveryId) VALUES (%s,%s,%s)"""
    record_to_insert = (t2WaitingTime, t2WaitingTimeInterval, deliveryId)
    cursor.execute(postgres_insert_query, record_to_insert)

    connection.commit()
    count = cursor.rowcount
    print(count, "Record inserted successfully into CPH Waiting Time table")
    requests.get("https://hc-ping.com/443ecacf-ec17-4912-91e5-183957ba5e07", timeout=10)
    data = urllib.request.urlopen("https://cphapi.simonottosen.dk/waitingtime?&order=id.desc&limit=1").read()
    output = json.loads(data)
    aDict = output[0]
    print(aDict)
    data = {
    u'id': str(aDict['id']),
    u't2waitingtime': str(aDict['t2waitingtime']),
    u't2waitingtimeinterval': str(aDict['t2waitingtimeinterval']),
    u'deliveryid': str(aDict['deliveryid'])
    }
    db.collection(u'waitingtime').document(str(aDict['id'])).set(data)
    print("PostgreSQL connection is closed")


except (Exception, psycopg2.Error) as error:
    print("Failed to insert record into CPH Waiting Time table", error)

finally:
    # closing database connection.
    if connection:
        cursor.close()
        connection.close()


