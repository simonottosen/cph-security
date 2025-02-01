# Pull base image
FROM python:3.8

# Use the recommended "key=value" format for ENV
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set the working directory
WORKDIR /project

# Install pipenv
RUN pip install --no-cache-dir pipenv

# Copy your Pipfile
COPY Pipfile /project/

# Lock dependencies (this creates/updates Pipfile.lock in /project)
RUN pipenv lock

# Install dependencies into the system (includes dev packages)
RUN pipenv install --system --dev

# Copy your application code
COPY project/app.py /project/app.py

# Set your default command
CMD ["python", "/project/app.py", "-h", "0.0.0.0"]