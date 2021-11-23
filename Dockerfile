FROM python:3.9

RUN pip3 install spacy

RUN pythom -m spacy download ru_core_news