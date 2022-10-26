FROM ghcr.io/luxonis/robothub-app:2022.269.1517-ubuntu22.04

RUN pip3 install -U numpy opencv-contrib-python-headless

ADD requirements.txt requirements.txt
RUN pip install -r requirements.txt

ARG FILE=app.py

ADD MultiMsgSync.py .

ADD $FILE run.py
