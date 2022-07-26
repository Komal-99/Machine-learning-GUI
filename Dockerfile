#Python's Alpine Base Image
FROM python:3.6-alpine3.6

#Installing all python modules specified
ADD requirements.txt requirements.txt
RUN pip install -r requirements.txt

#Copy App Contents
ADD . /app
WORKDIR /app

#Start Flask Server
CMD [ "python","uicode.py"]
#Expose server port
EXPOSE 8080