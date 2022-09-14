import psycopg2
import requests
import json
from datetime import datetime
import os


def main():
    POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD")
    POSTGRES_DB = os.environ.get("POSTGRES_DB")
    POSTGRES_USER = os.environ.get("POSTGRES_USER")
    POSTGRES_PORT = os.environ.get("POSTGRES_PORT")
    POSTGRES_HOST = os.environ.get("POSTGRES_HOST")
    
    response = requests.get('https://cph-flightinfo-prod.azurewebsites.net//api/v1/waiting/get?type=ventetid')
    waitingtime = json.loads(response.text)
    queue = (waitingtime["t2WaitingTime"])
    timestamp = (waitingtime["deliveryId"])
    timestamp = (timestamp.replace("T", " ")) 
    airport = "CPH"
    print(queue, timestamp, airport)

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


    except (Exception, psycopg2.Error) as error:
        print("Failed to insert record into CPH Waiting Time table", error)

    finally:
        # closing database connection.
        if connection:
            cursor.close()
            connection.close()


