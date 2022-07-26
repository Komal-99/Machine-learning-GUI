#Python's Alpine Base Image
FROM python:3.7
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE $PORT
#Start Flask Server
CMD python --bind 0.0.0.0:$PORT app:uicode
#Expose server port