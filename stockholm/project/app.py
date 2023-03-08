# from selenium.webdriver import Chrome
# from selenium.webdriver.chrome.service import Service as ChromeService

# url = 'https://www.swedavia.com/arlanda/security-control/'
# service = ChromeService() 
# with Chrome(service=service) as driver:
#     driver.get(url)

#     element = driver.find_element(by='css selector', value='#waitingtimes > div > div > div:nth-child(3) > div > div > div > div:nth-child(3) > div.terminalQueueTime')
#     waiting_time = element.text.strip()

# print(waiting_time)

# Will also need standalone Selenium Container - docker run -d -p 4444:4444 -p 7900:7900 --shm-size="2g" selenium/standalone-firefox:4.8.1-20230306

import psycopg2
import requests
import json
from datetime import datetime
import os
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import urllib.request
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver import Firefox, FirefoxOptions




def main():
    def select_first_char(string):
        if string[0].isdigit():
            return int(string[0])
        else:
            raise ValueError("First character is not an integer.")

    POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD")
    POSTGRES_DB = os.environ.get("POSTGRES_DB")
    POSTGRES_USER = os.environ.get("POSTGRES_USER")
    POSTGRES_PORT = os.environ.get("POSTGRES_PORT")
    POSTGRES_HOST = os.environ.get("POSTGRES_HOST")
    CPHAPI_HOST = os.environ.get("CPHAPI_HOST")
    HEALTHCHECK = os.environ.get("HEALTHCHECK")
    GOOGLE_APPLICATION_CREDENTIALS = '/home/user/app/keyfile.json'



    options = FirefoxOptions()

    browser = webdriver.Remote(command_executor='http://selenium_firefox:4444/wd/hub', options=options)
    browser.get('https://www.swedavia.com/arlanda/security-control/')

    element = browser.find_element(by='css selector', value='#waitingtimes > div > div > div:nth-child(3) > div > div > div > div:nth-child(3) > div.terminalQueueTime')
    waiting_time = element.text.strip()
    browser.quit()

    try:
        queue = select_first_char(waiting_time)
    except ValueError as e:
        print(f"Error: {e}")

    airport = "ARN"
    now_utc = datetime.utcnow()
    timestamp = now_utc.strftime('%Y-%m-%dT%H:%M:%S')
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
