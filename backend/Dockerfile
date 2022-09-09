# Pull base image
FROM python:3.7
# Set environment varibles
WORKDIR /code/
# Install dependencies
RUN pip install pipenv
COPY Pipfile /code/
COPY env.txt /code/
RUN pipenv lock
RUN pipenv install --system --dev
COPY fetchwaitingtime.py /code/fetchwaitingtime.py
COPY cphairportqueue-firebase-key.json /code/cphairportqueue-firebase-key.json
RUN apt-get update && apt-get -y install cron nano
COPY crontab /etc/cron.d/crontab
RUN chmod 0644 /etc/cron.d/crontab
RUN /usr/bin/crontab /etc/cron.d/crontab
RUN crontab /code/env.txt
CMD ["cron", "-f"]