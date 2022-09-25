# Airport Security Queue

This tool is designed to provide estimates of the security queue in the CPH airport at a user-defined point in time in the future. The tool uses Machine Learning based on historical data to estimate the queue.

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

# Front-end usage

The frontend system will be running on localhost:8501. The system is running on [Streamlit](https://github.com/streamlit/streamlit). 
The system will show the latest security waiting time from your database and will also enable you to utilize a simple machine learning model to calculate the future waiting time. The model is being trained every time you request a new estimate. 


