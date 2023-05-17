

# Airport Security Queue
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/simonottosen/cph-security/blob/main/LICENSE)
[![CodeFactor](https://www.codefactor.io/repository/github/simonottosen/cph-security/badge)](https://www.codefactor.io/repository/github/simonottosen/cph-security)

This tool is designed to provide estimates of the security queue in the CPH airport at a user-defined point in time in the future. The tool uses Machine Learning based on historical data to estimate the queue.

Once the system is running you can access the frontend on [http://localhost:8501](http://localhost:8501)

### To-Do
Implement data on amount of flights in the airport and understand correlation
```
https://www.cph.dk/api/FlightInformation/GetFlightInfoTable?direction=D&userQuery=*:*&startDateTime=2023-05-17T00:00:00.000Z&endDateTime=2023-05-17T23:59:00.000Z&language=da
```

# Server configuration

The below guide will set up the server on your local environment with minimal configuration.

## Configuration of the server

Before the server will be able to run, you will need to set up Google Firebase and the relevant environment variables. If you do not wish to run with Google Firebase, delete the relevant code in /backend/app/fetch.py

### Set up environment

The .env file present in the repository will be able to work OOTB. It is recommended to set up a reverse proxy such as Traefik or similar and edit the environment variable accordingly. 

You will need to save your Firebase key as keyfile.json in the root folder of the project or edit the environment variable $GCP_KEY_PATH. 
[Firebase getting started](https://cloud.google.com/firestore/docs/client/get-firebase)
[Setting up Firebase API key](https://firebase.google.com/docs/projects/api-keys)

## Deployment

Following the configuration of the environment variables and setting up Firebase, the server can be run with docker-compose. 
```
$ source .env
$ docker-compose up -d
```

## Ofelia scheduler configuration

Ofelia is utilized for orchestrating the fetchtime container and avoid using Cron through Docker, which can often be a challenging task when using environment variables. To change the time between each fetch of data, adjust the docker-compose file. 
Documentation for Ofelia can be found here [mcuadros/ofelia](https://github.com/mcuadros/ofelia).

```
labels:
    ofelia.enabled: "true"
    ofelia.job-exec.app.schedule: "@every 1m"
    ofelia.job-exec.app.command: "fetch"
```


## Usage of the API

The API is driven by [PostgREST/postgrest](https://github.com/PostgREST/postgrest). PostgREST serves a fully RESTful API from any existing PostgreSQL database. It provides a cleaner, more standards-compliant, faster API than you are likely to write from scratch.

Latest documentation is at [postgrest.org](http://postgrest.org). You can contribute to the docs in [PostgREST/postgrest-docs](https://github.com/PostgREST/postgrest-docs).



Get all
```
http://localhost:3000/waitingtime
```

Get all times with a waiting time less that 4 minutes
```
http://localhost:3000/waitingtime?queue=lt.4
```

Select only a subset of the data
```
http://localhost:3000/waitingtime?select=queue,timestamp
```


Get the current waiting time
```
http://localhost:3000/waitingtime?select=queue&order=id.desc&limit=1
```


## API-documentation

PostgREST uses the [OpenAPI](https://openapis.org/) standard to generate up-to-date documentation for APIs. This is being consumed by Swagger. It is available on localhost:8080 after you have run the project. 

[Swagger UI](https://swagger.io/tools/swagger-ui/) allows us to visualize and interact with the API’s resources without having any of the implementation logic in place. It’s automatically generated from the OpenAPI Specification, with the visual documentation making it easy for back end implementation and client side consumption.

##  Frontend

This code is the frontend part of the Waitport service, a web application that allows users to track waiting times at various European airports. The frontend is built using React and includes features such as selecting an airport to view current queue times and selecting a specific date and time to see predicted queue lengths.

### Dependencies
- React
- axios
- react-datetime
- react-bootstrap

### How to Use
1. Clone the repository to your local machine.
2. Install dependencies by running `npm install`.
3. Start the development server by running `npm start`.
4. Open the web application in your browser at `http://localhost:3000/`.

#### Features
- Select an airport from the dropdown menu to view current queue times and average queue times for the past two hours.
- Select a specific date and time to view predicted queue lengths for the selected airport.
- API endpoint available at `https://waitport.com/api/v1/:airport` where `:airport` is the airport code (e.g. `cph` for Copenhagen Airport). 
