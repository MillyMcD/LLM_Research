#syntax=docker/dockerfile:1.2
FROM python:3.12

ADD requirements.txt /usr/local/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /usr/local/requirements.txt

RUN curl -fsSL https://ollama.com/install.sh | sh

ADD entrypoint.sh entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

CMD tail -f /dev/null