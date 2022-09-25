FROM python:3.7
RUN groupadd --gid 1000 user && \
    useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash user
RUN mkdir /home/user/app
ADD setup.py /home/user/app
ADD app /home/user/app/app
WORKDIR /home/user/app/
RUN pip install pipenv
COPY Pipfile /home/user/app
RUN pipenv lock
RUN pipenv install --system --dev
RUN cd /home/user/app && \
    pip install --no-cache-dir .
RUN chown -R "1000:1000" /home/user
USER user
WORKDIR /home/user/app

CMD tail -f /dev/null
