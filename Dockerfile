FROM ghcr.io/luxonis/robothub-app:2022.269.1517-ubuntu22.04

RUN pip3 install -U numpy opencv-contrib-python-headless

ARG FILE=app.py

ADD MultiMsgSync.py .

ADD $FILE run.py
