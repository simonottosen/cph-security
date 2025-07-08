import psycopg2
import requests
import json
from datetime import datetime
import os
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import re
from supabase import create_client, Client
import gzip
import io
import http.client

from typing import Optional


# This function performs a healthcheck.
def healthcheck_perform(HEALTHCHECK):
    if HEALTHCHECK == "NULL":
        # If no healthcheck is provided, skip it and return a message
        healthcheck_result = "Skipping. Healthcheck not configured"
    else:
        # Otherwise attempt to do the healthcheck by sending an HTTP GET request to the healthcheckurl
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
    connection = None

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
        
    except Exception as e:
        # Catch exceptions if any
        database_write_result = (f"An error occurred: {e}")

    finally:
        # Closing the database connection
        if connection is not None:
            connection.close()

            
    return database_write_result

# ---------------------------------------------------------------------------
#                    Common post‑processing helper
# ---------------------------------------------------------------------------
def process_airport_result(
    queue: int,
    airport: str,
    healthcheck_id: str,
    timestamp: Optional[str] = None,
) -> None:
    """
    Centralised post‑processing for every airport function.

    It writes results to Postgres, Firebase and Supabase, pings the
    health‑check endpoint and reports a concise summary to stdout.

    Parameters
    ----------
    queue : int
        Waiting time in minutes.
    airport : str
        Three‑letter IATA airport code.
    healthcheck_id : str
        HealthChecks.io ping ID, or ``"NULL"`` to skip the ping.
    timestamp : str, optional
        ISO‑8601 timestamp.  If omitted, the current UTC time is used.
    """
    if timestamp is None:
        timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")

    database_status = database_write(queue, timestamp, airport)
    firebase_status = firebase_write(airport)
    supabase_status = supabase_write(queue, timestamp, airport)
    healthcheck_status = healthcheck_perform(healthcheck_id)

    print(
        f"Airport {airport} was completed with the following status. "
        f"Database: {database_status}. Firebase: {firebase_status}. "
        f"Supabase: {supabase_status}. Healthcheck: {healthcheck_status}. "
        f"Queue is {queue} at {timestamp}"
    )

def supabase_write(queue, timestamp, airport):
    url: str = os.environ.get("SUPABASE_URL")
    key: str = os.environ.get("SUPABASE_KEY")
    email: str = os.environ.get("SUPABASE_EMAIL")
    password: str = os.environ.get("SUPABASE_PASSWORD")

    supabase: Client = create_client(url, key)
    data = supabase.auth.sign_in_with_password({"email": email, "password": password})
    
    try:
        data, count = supabase.table('waitingtime') \
            .insert({"queue": queue, "timestamp": timestamp, "airport": airport}) \
            .execute()
        supabase_write_status = "Supabase write completed"
        supabase.auth.sign_out()
    except Exception as e:
        supabase_write_status = f"Supabase write failed with error: {e}"
    
    return supabase_write_status

    
# This function writes data to Firebase database.
def firebase_write(airport):
    CPHAPI_HOST = os.environ.get("CPHAPI_HOST") 
    GOOGLE_APPLICATION_CREDENTIALS = '/home/user/app/keyfile.json'
    
    # authenticate the credential with Firebase
    cred = credentials.Certificate(GOOGLE_APPLICATION_CREDENTIALS)
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    # create url for getting data from CPH API by appending airport code (three-letter code)
    apiurl=(str(CPHAPI_HOST)+str("?order=id.desc&limit=1&airport=eq.")+str(airport))
    
    # read data fetched from API endpoint using urllib.request.urlopen and store it in data    
    response = requests.get(apiurl)
    data = response.json()
    aDict = data[0]
        
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

def munich():
    healthcheck = os.environ.get("MUC_HEALTHCHECK") 
    airport = "MUC"
    airport_api = "https://www.passngr.de/info/generic/data/QueueWaitingTimeEDDM_en.json"
    
    # Use requests module to send a GET request to the airport API and retrieve waiting time information as JSON
    response = requests.get(airport_api)
    waitingtime = json.loads(response.text)
    
    for datas in waitingtime["queueTimes"]["current"]:
        if datas["queueId"] == "T2_Abflug_SIKO_ECO_NORD":
            numbers = datas["projectedWaitTime"]
            if numbers < 0 or numbers == 0.0:
                numbers = numbers = 0
            numbers = round(numbers)
            if isinstance(numbers, int):
                queue = numbers   # Assign a value to queue
            else:
                print("Error: Information not found in the JSON data.")
    
    process_airport_result(queue, airport, healthcheck)



def istanbul():
    healthcheck = os.environ.get("IST_HEALTHCHECK") 
    airport = "IST"
    airport_api = "https://www.istairport.com/umbraco/api/Checkpoint/GetWaitingTimes?culture=en-US"
    headers = {"Referer": "https://www.istairport.com/en/?locale=en"}

    
    # Use requests module to send a GET request to the airport API and retrieve waiting time information as JSON
    response = requests.get(airport_api, headers=headers)
    waitingtime = json.loads(response.text)
    numbers = waitingtime['result']['gateWaitTime']
    
    # Loop through data in the waiting time JSON until you find the relevant security checkpoint waiting time
    if isinstance(numbers, int):
        queue = numbers   # Assign a value to queue
    else:
        print("Error: Information not found in the JSON data.")
    process_airport_result(queue, airport, healthcheck)


def heathrow():
    healthcheck = os.environ.get("LHR_HEALTHCHECK") 
    airport = "LHR"
    conn = http.client.HTTPSConnection("api-dp-prod.dp.heathrow.com")
    payload = ''
    headers = {
      'Origin': 'https://www.heathrow.com'
    }
    conn.request("GET", "/pihub/securitywaittime/ByTerminal/2?=null", payload, headers)
    res = conn.getresponse()
    data = res.read()
    
    with gzip.GzipFile(fileobj=io.BytesIO(data)) as f:
        decompressed_data = f.read()
    
    # Print the decompressed data
    response = (decompressed_data.decode('utf-8'))  # Assuming the decompressed data is a UTF-8 encoded string
    
    waitingtime = json.loads(response)
    for entry in waitingtime:
        for measurement in entry['queueMeasurements']:
            if measurement['name'] == 'maximumWaitTime':
                numbers = measurement['value']
                if isinstance(numbers, int):
                    queue = numbers   # Assign a value to queue
                else:
                    print("Error: Information not found in the JSON data.")
    process_airport_result(queue, airport, healthcheck)

# This function retrieves the waiting time at Frankfurt airport
def frankfurt():
    # Define initial values
    healthcheck = os.environ.get("FRA_HEALTHCHECK") 
    airport = "FRA"
    airport_api = "https://www.frankfurt-airport.com/wartezeiten/appres/rest/waz?lang=en"
    
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
    
    process_airport_result(queue, airport, healthcheck)

# This function retrieves the waiting time at Dusseldorf airport
def dusseldorf():
    # Define initial values
    healthcheck = os.environ.get("DUS_HEALTHCHECK") 
    airport = "DUS"
    airport_api = "https://www.dus.com/api/sitecore/flightapi/WaitingTimes?lang=en"
    headers = {"X-Requested-With": "XMLHttpRequest"}

    # Use requests module to send a GET request to the airport API and retrieve waiting time information as JSON
    response = requests.get(airport_api, headers=headers)
    waitingtime = json.loads(response.text)
    
    # Loop through data in the waiting time JSON until you find the relevant security checkpoint waiting time
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
        process_airport_result(queue, airport, healthcheck)

# This function retrieves the waiting time at Copenhagen airport
def copenhagen():
    healthcheck = os.environ.get("CPH_HEALTHCHECK")
    airport = "CPH"
    api_url = "https://cphwaitingtime.z6.web.core.windows.net/waitingtime.json"

    # Fetch data
    resp = requests.get(api_url)
    resp.raise_for_status()
    data = resp.json()

    # Raw strings (or None if missing)
    t2_raw = data.get("t2WaitingTimeInterval")
    t3_raw = data.get("t3WaitingTimeInterval")

    # Parse any string like "10-15", "> 20 min", "< 5", etc. → list of ints
    def parse_interval(s: str) -> list[int]:
        nums = re.findall(r"\d+", s)
        return [int(n) for n in nums]

    # Build lists (empty if that terminal is missing)
    t2_times = parse_interval(t2_raw) if t2_raw else []
    t3_times = parse_interval(t3_raw) if t3_raw else []

    # Fail if neither terminal has data
    if not t2_times and not t3_times:
        raise RuntimeError("No waiting-time data available for Terminal 2 or Terminal 3")

    # Compute average over whatever values we got
    all_times = t2_times + t3_times
    average = sum(all_times) / len(all_times)
    queue = int(round(average))

    # Use deliveryId timestamp if present, otherwise now
    timestamp = data.get("deliveryId") or datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")

    # Final reporting
    process_airport_result(queue, airport, healthcheck, timestamp)
    
# This function retrieves the waiting time at Arlanda airport
def arlanda():
    # Define initial values
    healthcheck = os.environ.get("ARN_HEALTHCHECK") 
    airport = "ARN"
    airport_api = "https://www.swedavia.com/services/queuetimes/v2/airport/en/ARN/true"
    
    # Use requests module to send a GET request to the airport API and retrieve waiting time information as JSON
    response = requests.get(airport_api)
    waitingtime = json.loads(response.text)
    for queue_time in waitingtime['queueTimesList']:
        if queue_time['locationName'] == 'Terminal 5':
            timeinput = {queue_time["interval"]}
    
    # Loop through data in the waiting time JSON until you find the relevant security checkpoint waiting time
    numbers = list(map(int,re.findall('\d+', str(timeinput))))

    if isinstance(numbers[0], int):
        queue = numbers[0]
    else:
        print("Error: Could not get waitingtime for Arlanda.")
    
    process_airport_result(queue, airport, healthcheck)

# This function retrieves the waiting time at Dublin airport
def dublin():
    # Define initial values
    healthcheck = os.environ.get("DUB_HEALTHCHECK") 
    airport = "DUB"
    airport_api = "https://www.dublinairport.com/upi/SecurityTimes/GetTimes"
    
    # Use requests module to send a GET request to the airport API and retrieve waiting time information as JSON
    response = requests.get(airport_api)
    waitingtime = json.loads(response.text)
    t1_wait_time = None
    for key, value in waitingtime.items():
        if key == 'T1':
            t1_wait_time = int(value.strip('= '))

    # If wf1_wait_time is not None, assign its value to a queue variable. Otherwise print an error message
    if t1_wait_time is not None:
        queue = t1_wait_time 
        if queue < 0:
            queue = 0
    else:
        print("Waiting time not found for Dublin Airport.")
    
    process_airport_result(queue, airport, healthcheck)




# This function retrieves the waiting time at Oslo airport
def oslo():
    # Define initial values
    healthcheck = os.environ.get("OSL_HEALTHCHECK") 
    airport = "OSL"
    airport_api = "https://avinor.no/Api/QueueTime/OSL"
    
    # Use requests module to send a GET request to the airport API and retrieve waiting time information as JSON
    response = requests.get(airport_api)
    waitingtime = json.loads(response.text)
    time_minutes_rounded = waitingtime.get("timeMinutesRounded")

    # If the first element of the extracted number list is an integer, assign its value to a queue variable. Otherwise print an error message
    if isinstance(time_minutes_rounded, int):
        queue = time_minutes_rounded
        if queue < 0:
            queue = 0
    else:
        print("Waiting time not found in the JSON data.")
    
    process_airport_result(queue, airport, healthcheck)


# # This function retrieves the waiting time at Berlin airport. Currently removed as BER has blocked the server.
# def berlin():
#     # Define initial values
#     healthcheck = os.environ.get("BER_HEALTHCHECK") 
#     airport = "BER"
#     airport_api = "https://ber.berlin-airport.de/api.aplsv2.json?lang=en"

#     # Use requests module to send a GET request to the airport API and retrieve waiting time information as JSON
#     response = requests.get(airport_api)
#     waitingtime = json.loads(response.text)
#     for i in waitingtime["data"]["apls-data"]:
#         if i["terminal"] == "T1" and i["security_control"] == "1":
#             low = i["low_minutes"]

#     for i in waitingtime["data"]["apls-data"]:
#             if i["terminal"] == "T1" and i["security_control"] == "1":
#                 high = i["high_minutes"]

#     time_minutes_rounded = high+low
#     time_minutes_rounded = time_minutes_rounded / 2
#     time_minutes_rounded = int(round(time_minutes_rounded, 1))

#     # If the first element of the extracted number list is an integer, assign its value to a queue variable. Otherwise print an error message
#     if type(time_minutes_rounded) == int:
#         queue = time_minutes_rounded
#     else:
#         print("Waiting time not found in the JSON data.")    
#     # Get current UTC datetime and format as string
#     now_utc = datetime.utcnow()
#     timestamp = now_utc.strftime('%Y-%m-%dT%H:%M:%S')
    
#     # Call three other functions to write the retrieved waiting time data to a database, firebase, and to perform a healthcheck. Store the result of each function into corresponding variables
#     database_write_status = database_write(queue, timestamp, airport)
#     firebase_write_status = firebase_write(airport)
#     supabase_write_status = supabase_write(queue, timestamp, airport)
#     healthcheck_perform_status = healthcheck_perform(healthcheck)
    
#     # Print a completion message with the status results from the three previous function calls
#     print("Airport "+str(airport)+" was completed with the following status. Database: "+str(database_write_status)+". Firebase: "+str(firebase_write_status)+". Healthcheck: "+str(healthcheck_perform_status)+". Queue is "+str(queue)+" at "+str(timestamp))


def amsterdam():
    # Define initial values
    healthcheck = os.environ.get("AMS_HEALTHCHECK") 
    airport = "AMS"
    airport_api = "https://www.schiphol.nl/api/proxy/v3/waittimes/security-filters"
    
    # Use requests module to send a GET request to the airport API and retrieve waiting time information as JSON
    response = requests.get(airport_api)
    waitingtime = json.loads(response.text)
    vf1_wait_time = waitingtime.get("VF1", {}).get("waitTimeInSeconds")

    # If wf1_wait_time is not None, assign its value to a queue variable. Otherwise print an error message
    if vf1_wait_time is not None:
        queue = vf1_wait_time // 60  # Convert wait time in seconds to minutes
        if queue < 0:
            queue = 0
    else:
        print("Waiting time not found for Schipol.")
    
    process_airport_result(queue, airport, healthcheck)

