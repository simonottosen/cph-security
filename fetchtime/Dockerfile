# Will also need standalone Selenium Container - docker run -d -p 4444:4444 -p 7900:7900 --shm-size="2g" selenium/standalone-firefox:4.8.1-20230306
FROM python:3.13.3
RUN groupadd --gid 1000 user && \
    useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash user
RUN mkdir /home/user/app
ADD setup.py /home/user/app
ADD project /home/user/app/app
WORKDIR /home/user/app/
RUN pip install --no-cache-dir pipenv
COPY Pipfile /home/user/app
RUN pipenv lock
RUN pipenv install --system --dev
RUN cd /home/user/app && \
    pip install --no-cache-dir .
RUN chown -R "1000:1000" /home/user
USER user
WORKDIR /home/user/app

CMD tail -f /dev/null
