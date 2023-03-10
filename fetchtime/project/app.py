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

# This function performs a healthcheck.
def healthcheck_perform(HEALTHCHECK):
    if HEALTHCHECK == "NULL":
        # If no healthcheck is provided, skip it and return a message
        print("Skipping. Healthcheck not configured")
        healthcheck_result = "Skipping. Healthcheck not configured"
    else:
        # Otherwise attempt to do the healthcheck by sending an HTTP GET request to the healthcheckurl
        print("Attempting to do healthcheck")
        healthcheckurl=(str("https://hc-ping.com/")+str(HEALTHCHECK))
        requests.get(healthcheckurl, timeout=10)
        # Return a message indicating the healthcheck was submitted successfully
        healthcheck_result = "Healthcheck submitted"
    # Return the result of the function: whether the healthcheck was successful or not
    return healthcheck_result

# This function writes data to a Postgres database.
def database_write(queue, timestamp, airport):
    # Get environment variables
    POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD")
    POSTGRES_DB = os.environ.get("POSTGRES_DB")
    POSTGRES_USER = os.environ.get("POSTGRES_USER")
    POSTGRES_PORT = os.environ.get("POSTGRES_PORT")
    POSTGRES_HOST = os.environ.get("POSTGRES_HOST")
    
    database_write_result = "Failed to write to database. No error registered."
    
    try:
        # Create a connection to the database
        connection = psycopg2.connect(user=POSTGRES_USER,
                                      password=POSTGRES_PASSWORD,
                                      host=POSTGRES_HOST,
                                      port=POSTGRES_PORT,
                                      database=POSTGRES_DB)
        cursor = connection.cursor()

        # Define query and values to be inserted
        postgres_insert_query = """ INSERT INTO waitingtime (queue, timestamp, airport) VALUES (%s,%s,%s)"""
        record_to_insert = (queue, timestamp, airport)
        
        # Execute and commit
        cursor.execute(postgres_insert_query, record_to_insert)
        connection.commit()
        
        database_write_result = "Database write complete"
        
    except (Exception, psycopg2.Error) as error:
        # Catch exceptions if any
        database_write_result = ("Failed to write to the database: ", error)
        
    finally:
        # Closing the database connection
        if connection:
            cursor.close()
            connection.close()
            
    return database_write_result

# This function writes data to Firebase database.
def firebase_write(airport):
    CPHAPI_HOST = os.environ.get("CPHAPI_HOST") 
    db = firestore.client()
    GOOGLE_APPLICATION_CREDENTIALS = '/home/user/app/keyfile.json'
    
    # authenticate the credential with Firebase
    cred = credentials.Certificate(GOOGLE_APPLICATION_CREDENTIALS)
    firebase_admin.initialize_app(cred)
    
    # create url for getting data from CPH API by appending airport code (three-letter code)
    apiurl=(str(CPHAPI_HOST)+str("/waitingtime?&order=id.desc&limit=1&airport=eq.")+str(airport))
    
    # read data fetched from API endpoint using urllib.request.urlopen and store it in data
    data = urllib.request.urlopen(apiurl).read()
    output = json.loads(data)
    aDict = output[0]
        
    # create a dictionary object with keys 'id', 'queue', 'timestamp' and 'airport' and their respective values
    data = {
            u'id': str(aDict['id']),
            u'queue': str(aDict['queue']),
            u'timestamp': str(aDict['timestamp']),
            u'airport': str(aDict['airport'])
        }
    
    try:
        # write the document to Firestore with id being the value associated with key id in dictionary data
        db.collection(u'waitingtimetest').document(str(aDict['id'])).set(data)
        firebase_write_status = "Firebase write completed"
    
    except Exception as e:
        firebase_write_status = f"Firebase write failed with error: {e}"
    
    return firebase_write_status

# This function retrieves the waiting time at Frankfurt airport
def frankfurt():
    # Define initial values
    healthcheck = os.environ.get("FRA_HEALTHCHECK") 
    airport = "FRA"
    airport_api = "https://www.frankfurt-airport.com/wartezeiten/rest/waz?lang=en"
    
    # Use requests module to send a GET request to the airport API and retrieve waiting time information as JSON
    response = requests.get(airport_api)
    waitingtime = json.loads(response.text)
    
    # Loop through data in the waiting time JSON until you find the relevant security checkpoint waiting time
    for item in waitingtime["data"]:
        if item["ps"] == "Security checkpoint, Concourse A\r\nDeparture gates: A1 - A69":
            status = item["status"]
            break
    
    # Using regular expressions, extract the first number found in the checkpoint waiting time status string and convert it from a string to an integer
    numbers = list(map(int,re.findall('\d+', str(status))))
    
    # If the first element of the extracted number list is an integer, assign its value to a queue variable. Otherwise print an error message
    if isinstance(numbers[0], int):
        queue = numbers[0]
    else:
        print("Error: Information not found in the JSON data.")
    
    # Get current UTC datetime and format as string
    now_utc = datetime.utcnow()
    timestamp = now_utc.strftime('%Y-%m-%dT%H:%M:%S')
    
    # Call three other functions to write the retrieved waiting time data to a database, firebase, and to perform a healthcheck. Store the result of each function into corresponding variables
    database_write_status = database_write(queue, timestamp, airport)
    firebase_write_status = firebase_write(airport)
    healthcheck_perform_status = healthcheck_perform(healthcheck)
    
    # Print a completion message with the status results from the three previous function calls
    print("Airport "+str(airport)+" was completed with the following status. Database: "+str(database_write_status)+". Firebase: "+str(firebase_write_status)+". Healthcheck: "+str(healthcheck_perform_status))

