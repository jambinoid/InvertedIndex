FROM python:3.9.7-slim-buster
WORKDIR /inverted_index
COPY ./requirements.txt .
RUN pip install -r requirements.txt
COPY . .
RUN python3 setup.py install && python3 tools/create_index.py --config config.json
CMD ["python3", "api.py", "--config", "config.json"]
