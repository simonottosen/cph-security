# Pull base image
FROM python:3.14.0a4
ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1
# Set environment varibles
WORKDIR /project/
# Install dependencies
RUN pip install --no-cache-dir pipenv
COPY Pipfile /project/
RUN pipenv lock
RUN pipenv install --system --dev
COPY project/app.py /project/app.py
#CMD python /project/app.py
CMD [ "python", "/project/app.py", "-h", "0.0.0.0"]
