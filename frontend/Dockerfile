# Pull base image
FROM python:3.8
ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1
# Set environment varibles
WORKDIR /project/
# Install dependencies
RUN pip install pipenv
COPY Pipfile /project/
RUN pipenv lock
RUN pipenv install --system --dev
COPY project/app.py /project/app.py
CMD streamlit run /project/app.py
